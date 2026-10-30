"""Quality checks on a finished build. Plans 3 and 4 refuse a dataset whose QA did not pass."""

from datetime import date
from pathlib import Path

import polars as pl

from alpha_factory.contracts.data import QACheck, QAReport
from alpha_factory.data.adjust import valid_splits
from alpha_factory.data.build import read_manifest
from alpha_factory.data.calendar import TradingCalendar
from alpha_factory.data.config import DataConfig
from alpha_factory.data.layout import DataLayout, Kind

MAX_LISTED = 50


def _scan(build_dir: Path, table: str) -> pl.LazyFrame:
    return pl.scan_parquet(build_dir / table / "*" / "part-0.parquet", hive_partitioning=False)


def _check(name: str, passed: bool, value: float, threshold: float, detail: str) -> QACheck:
    return QACheck(
        name=name, passed=passed, value=float(value), threshold=float(threshold), detail=detail
    )


def _equal_check(dev_dir: Path | None, full_dir: Path | None, dev_end: date) -> QACheck:
    """Every dev row must equal the matching full row. Mismatches mean something read the future."""
    name = "dev_full_equal"
    if dev_dir is None or full_dir is None:
        return _check(
            name, False, 1, 0, "needs both a dev and a full build; run `af data build --kind both`"
        )
    mismatches = 0
    detail: list[str] = []
    dev_names = (
        _scan(dev_dir, "universe").select("ticker_id").unique().collect()["ticker_id"].to_list()
    )
    for table in ("bars_30m", "universe"):
        dev = _scan(dev_dir, table)
        full = _scan(full_dir, table).filter(pl.col("session_date") <= dev_end)
        if table == "bars_30m":
            full = full.filter(pl.col("ticker_id").is_in(dev_names))
        cols = dev.collect_schema().names()
        only_dev = (
            dev.join(full, on=cols, how="anti", nulls_equal=True).select(pl.len()).collect().item()
        )
        only_full = (
            full.join(dev, on=cols, how="anti", nulls_equal=True).select(pl.len()).collect().item()
        )
        mismatches += only_dev + only_full
        detail.append(f"{table}: {only_dev} dev-only, {only_full} full-only rows")
    for path in ("adj_factors.parquet",):
        dev = pl.scan_parquet(dev_dir / path)
        full = pl.scan_parquet(full_dir / path).filter(
            (pl.col("session_date") <= dev_end) & pl.col("ticker_id").is_in(dev_names)
        )
        cols = dev.collect_schema().names()
        a = dev.join(full, on=cols, how="anti", nulls_equal=True).select(pl.len()).collect().item()
        b = full.join(dev, on=cols, how="anti", nulls_equal=True).select(pl.len()).collect().item()
        mismatches += a + b
        detail.append(f"adj_factors: {a} dev-only, {b} full-only rows")
    return _check(name, mismatches == 0, mismatches, 0, "; ".join(detail))


def run_qa(layout: DataLayout, cfg: DataConfig, kind: Kind, write: bool = True) -> QAReport:
    build_dir = layout.current_build(kind)
    if build_dir is None:
        raise FileNotFoundError(f"no {kind} build; run `af data build --kind {kind}`")
    manifest = read_manifest(build_dir)
    last = cfg.end if kind == "full" else cfg.dev_end
    cal = TradingCalendar(cfg.start, cfg.end)
    expected_sessions = cal.sessions(cfg.start, last)
    q = cfg.qa
    checks: list[QACheck] = []

    bars = _scan(build_dir, "bars_30m")
    universe = _scan(build_dir, "universe")

    # calendar_complete
    have = {
        date.fromisoformat(p.name.split("=")[1])
        for p in (build_dir / "bars_30m").glob("session_date=*")
    }
    missing = [d for d in expected_sessions if d not in have]
    checks.append(
        _check(
            "calendar_complete",
            not missing,
            len(missing),
            0,
            f"missing sessions: {[str(d) for d in missing[:10]]}"
            if missing
            else "all sessions present",
        )
    )

    # member-bar completeness and thin bars
    members = universe.select("session_date", "ticker_id")
    per_session = members.group_by("session_date").len().collect()
    expected_bars = sum(
        n * cal.n_bars(d)
        for d, n in zip(per_session["session_date"], per_session["len"], strict=True)
    )
    member_bars = (
        bars.join(members, on=["session_date", "ticker_id"], how="inner")
        .select(
            pl.len().alias("n"),
            (pl.col("n_minutes") < cfg.bars.thin_bar_minutes).sum().alias("thin"),
        )
        .collect()
    )
    n_present, n_thin = int(member_bars["n"][0]), int(member_bars["thin"][0])
    rate = 1 - n_present / expected_bars if expected_bars else 1.0
    checks.append(
        _check(
            "missing_bar_rate",
            rate <= q.max_missing_bar_rate,
            rate,
            q.max_missing_bar_rate,
            f"{expected_bars - n_present} of {expected_bars} member-bars absent",
        )
    )
    thin = n_thin / n_present if n_present else 1.0
    checks.append(
        _check(
            "thin_bar_share",
            thin <= q.max_thin_bar_share,
            thin,
            q.max_thin_bar_share,
            f"{n_thin} of {n_present} member-bars under {cfg.bars.thin_bar_minutes} minutes",
        )
    )

    # price sanity
    price_cols = ["open", "high", "low", "close", "raw_close", "vwap"]
    sane = bars.select(
        pl.len().alias("n"),
        (
            ~(
                (pl.col("low") <= pl.col("open"))
                & (pl.col("open") <= pl.col("high"))
                & (pl.col("low") <= pl.col("close"))
                & (pl.col("close") <= pl.col("high"))
            )
        )
        .sum()
        .alias("bad_ohlc"),
        pl.any_horizontal(*[pl.col(c) <= 0 for c in price_cols]).sum().alias("bad_price"),
    ).collect()
    n_bars_total = int(sane["n"][0])
    bad_ohlc, bad_price = int(sane["bad_ohlc"][0]), int(sane["bad_price"][0])
    checks.append(
        _check(
            "ohlc_consistent",
            bad_ohlc == 0,
            bad_ohlc,
            0,
            f"{bad_ohlc} of {n_bars_total} bars violate low <= open, close <= high",
        )
    )
    checks.append(
        _check(
            "positive_prices",
            bad_price == 0,
            bad_price,
            0,
            f"{bad_price} bars with a zero or negative price",
        )
    )

    # extreme returns (bar to bar, adjusted)
    extremes = (
        bars.sort("ticker_id", "ts")
        .with_columns(
            (pl.col("close") / pl.col("close").shift(1).over("ticker_id") - 1).alias("ret")
        )
        .filter(pl.col("ret").abs() > q.extreme_return_abs)
        .select("ticker_id", "ts", "ret")
        .collect()
    )
    share = extremes.height / n_bars_total if n_bars_total else 0.0
    listed = [
        f"{r['ticker_id']}@{r['ts']:%Y-%m-%d %H:%M}:{r['ret']:+.2f}"
        for r in extremes.head(MAX_LISTED).to_dicts()
    ]
    checks.append(
        _check(
            "extreme_returns",
            share <= q.max_extreme_return_share,
            share,
            q.max_extreme_return_share,
            f"{extremes.height} bars above {q.extreme_return_abs:.0%}: {listed}",
        )
    )

    # split continuity
    splits = pl.read_parquet(layout.ref_table("splits")).filter(
        (pl.col("execution_date") > expected_sessions[0]) & (pl.col("execution_date") <= last)
    )
    gaps: list[tuple[float, str]] = []
    rejected = 0
    if splits.height:
        sb = (
            bars.filter(pl.col("ticker_id").is_in(splits["ticker_id"].to_list()))
            .select("ticker_id", "session_date", "bar", "open", "close", "raw_close")
            .collect()
        )
        daily = (
            sb.sort("ticker_id", "session_date", "bar")
            .group_by("ticker_id", "session_date", maintain_order=True)
            .agg(pl.col("raw_close").last())
        )
        # Only splits the raw prices confirm are adjusted, so only those can be checked; the
        # rejected ones (another company that used the symbol, bad vendor rows) are counted.
        confirmed = valid_splits(daily, splits)
        rejected = splits.height - confirmed.height
        for row in confirmed.to_dicts():
            t, d = row["ticker_id"], row["execution_date"]
            tb = sb.filter(pl.col("ticker_id") == t).sort("session_date", "bar")
            before, after = (
                tb.filter(pl.col("session_date") < d),
                tb.filter(pl.col("session_date") == d),
            )
            if before.height and after.height:
                gaps.append((abs(after["open"][0] / before["close"][-1] - 1), f"{t}@{d}"))
    worst = max(gaps, default=(0.0, ""))
    checks.append(
        _check(
            "split_continuity",
            worst[0] <= q.max_split_gap_abs,
            worst[0],
            q.max_split_gap_abs,
            f"{len(gaps)} splits checked, {rejected} rejected by the raw-price check; "
            f"worst {worst[1] or 'n/a'}",
        )
    )

    # universe size
    window = cfg.universe.adv_window_days
    sizes = dict(zip(per_session["session_date"], per_session["len"], strict=True))
    bad_sizes = [d for d in expected_sessions[window:] if sizes.get(d, 0) != cfg.universe.top_n]
    first_bad = [str(d) for d in bad_sizes[:10]]
    checks.append(
        _check(
            "universe_size",
            not bad_sizes,
            len(bad_sizes),
            0,
            f"sessions without exactly {cfg.universe.top_n} members: {first_bad}"
            if bad_sizes
            else f"{cfg.universe.top_n} members every session",
        )
    )

    # survivorship
    inst = pl.read_parquet(layout.ref_table("instruments"))
    member_ids = universe.select("ticker_id").unique().collect()["ticker_id"].to_list()
    delisted = inst.filter(
        pl.col("ticker_id").is_in(member_ids)
        & pl.col("delisted_date").is_not_null()
        & (pl.col("delisted_date") <= last)
    )
    checks.append(
        _check(
            "has_delisted_members",
            delisted.height >= 1,
            delisted.height,
            1,
            f"{delisted.height} members later delisted",
        )
    )

    # identity gaps
    stats = pl.read_parquet(build_dir / "session_stats.parquet")
    cs, un = float(stats["cs_dollar_volume"].sum()), float(stats["unmapped_dollar_volume"].sum())
    un_share = un / (cs + un) if cs + un else 0.0
    checks.append(
        _check(
            "unmapped_symbols",
            un_share <= q.max_unmapped_dollar_share,
            un_share,
            q.max_unmapped_dollar_share,
            "share of dollar volume from symbols with no instrument",
        )
    )

    # leakage
    dev_dir = build_dir if kind == "dev" else layout.current_build("dev")
    full_dir = build_dir if kind == "full" else layout.current_build("full")
    checks.append(_equal_check(dev_dir, full_dir, cfg.dev_end))
    if kind == "dev":
        late = max(have, default=cfg.start)
        checks.append(
            _check(
                "dev_end_respected",
                late <= cfg.dev_end and manifest.end <= cfg.dev_end,
                0 if late <= cfg.dev_end else 1,
                0,
                f"last session {late}, dev_end {cfg.dev_end} (inclusive)",
            )
        )
    else:
        checks.append(_check("dev_end_respected", True, 0, 0, "not applicable to the full build"))

    report = QAReport(dataset_hash=manifest.dataset_hash, checks=tuple(checks))
    if write:
        (build_dir / "qa.json").write_text(report.model_dump_json(indent=2))
    return report
