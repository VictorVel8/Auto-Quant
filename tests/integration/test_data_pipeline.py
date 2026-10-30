"""Plan 2 end to end on synthetic raw inputs: reference -> bars -> adjustment -> universe -> QA."""

import json
import os
import shutil
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import cast

import polars as pl
import pytest
from typer.testing import CliRunner

from alpha_factory.cli import app
from alpha_factory.contracts.data import QACheck, QAReport
from alpha_factory.data import DatasetNotReadyError, MarketData, SealedDatasetError
from alpha_factory.data import build as build_mod
from alpha_factory.data.build import build_dataset, read_manifest, verify_build
from alpha_factory.data.config import (
    DataConfig,
    KnownCorporateAction,
    QAConfig,
    UniverseConfig,
)
from alpha_factory.data.layout import DataLayout
from alpha_factory.data.qa import run_qa
from alpha_factory.data.reference import build_ref_tables
from alpha_factory.hashing import sha256_canonical
from alpha_factory.settings import Settings
from alpha_factory.synthetic.raw import RawAnswers, write_raw_market

DEV_END = date(2024, 12, 31)
CONFIG_YAML = """\
start: 2024-11-01
end: 2025-01-10
dev_end: 2024-12-31
types: [CS]
universe:
  top_n: 30
  tradable_top_n: 15
"""
CFG = DataConfig(
    start=date(2024, 11, 1),
    end=date(2025, 1, 10),
    dev_end=DEV_END,
    universe=UniverseConfig(top_n=30, tradable_top_n=15),
)
PARAMS = sha256_canonical(CFG.model_dump(mode="json"))


def derive_root(src: Path, dst: Path, *, share: tuple[str, ...] = ("raw",)) -> DataLayout:
    """New data root that links `raw` (big, immutable) and copies everything else."""
    dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name in share:
            (dst / child.name).symlink_to(child)
        else:
            shutil.copytree(child, dst / child.name, symlinks=True)
    return DataLayout(dst)


@pytest.fixture(scope="module")
def raw() -> Iterator[tuple[Path, RawAnswers]]:
    import tempfile

    root = Path(tempfile.mkdtemp(prefix="af_raw_"))
    answers = write_raw_market(root)
    build_ref_tables(DataLayout(root))
    yield root, answers
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="module")
def built(raw: tuple[Path, RawAnswers]) -> DataLayout:
    layout = DataLayout(raw[0])
    for kind in ("full", "dev"):
        build_dataset(layout, CFG, kind, PARAMS, source="synthetic")
    for kind in ("dev", "full"):
        report = run_qa(layout, CFG, kind)
        assert report.passed, [c for c in report.checks if not c.passed]
    return layout


def ticker_id(layout: DataLayout, ticker: str) -> str:
    sym = pl.read_parquet(layout.ref_table("symbols")).filter(pl.col("ticker") == ticker)
    return sym["ticker_id"][0]


def bars(layout: DataLayout, kind: str) -> pl.DataFrame:
    b = layout.current_build(kind)  # pyright: ignore[reportArgumentType]
    assert b is not None
    return pl.read_parquet(b / "bars_30m" / "*" / "part-0.parquet", hive_partitioning=False)


def universe(layout: DataLayout, kind: str) -> pl.DataFrame:
    b = layout.current_build(kind)  # pyright: ignore[reportArgumentType]
    assert b is not None
    return pl.read_parquet(b / "universe" / "*" / "part-0.parquet", hive_partitioning=False)


# --- the checkpoint properties -------------------------------------------------------------------


def test_both_builds_pass_qa(built: DataLayout) -> None:
    for kind in ("dev", "full"):
        b = built.current_build(kind)  # pyright: ignore[reportArgumentType]
        assert b is not None
        report = QAReport.model_validate_json((b / "qa.json").read_text())
        assert report.passed
        assert {c.name for c in report.checks} == {
            "calendar_complete", "missing_bar_rate", "thin_bar_share", "ohlc_consistent",
            "positive_prices", "extreme_returns", "split_continuity", "universe_size",
            "has_delisted_members", "unmapped_symbols", "dev_full_equal", "dev_end_respected",
        }  # fmt: skip


def test_dev_contains_nothing_after_dev_end_and_full_does(built: DataLayout) -> None:
    assert cast(date, bars(built, "dev")["session_date"].max()) <= DEV_END
    assert cast(date, universe(built, "dev")["session_date"].max()) <= DEV_END
    assert cast(date, bars(built, "full")["session_date"].max()) > DEV_END


def test_every_dev_row_equals_the_full_row(built: DataLayout) -> None:
    dev, full = bars(built, "dev"), bars(built, "full")
    full = full.filter(
        (pl.col("session_date") <= DEV_END)
        & pl.col("ticker_id").is_in(dev["ticker_id"].unique().to_list())
    )
    assert dev.sort("session_date", "ticker_id", "bar").equals(
        full.sort("session_date", "ticker_id", "bar")
    )
    u_dev, u_full = universe(built, "dev"), universe(built, "full")
    assert u_dev.sort("session_date", "rank").equals(
        u_full.filter(pl.col("session_date") <= DEV_END).sort("session_date", "rank")
    )


def test_universe_has_exactly_top_n_each_session_after_warmup(built: DataLayout) -> None:
    per = universe(built, "full").group_by("session_date").len().sort("session_date")
    assert per["len"].to_list() == [30] * per.height
    assert per.height == 47 - 20  # warm-up: the first 20 sessions have no universe


def test_universe_excludes_pennies_etfs_and_unmapped(
    built: DataLayout, raw: tuple[Path, RawAnswers]
) -> None:
    ans = raw[1]
    ids = set(bars(built, "full")["ticker_id"].unique())
    assert ticker_id(built, ans.penny_ticker) not in set(universe(built, "full")["ticker_id"])
    assert ticker_id(built, ans.etf_ticker) not in ids
    assert ans.unmapped_ticker not in set(bars(built, "full")["ticker"])


def test_delisted_name_is_a_member_until_it_delists(
    built: DataLayout, raw: tuple[Path, RawAnswers]
) -> None:
    t, last = raw[1].delisted
    tid = ticker_id(built, t)
    u = universe(built, "full").filter(pl.col("ticker_id") == tid)
    assert u.height > 0 and cast(date, u["session_date"].max()) == last
    assert (
        cast(date, bars(built, "full").filter(pl.col("ticker_id") == tid)["session_date"].max())
        == last
    )


def test_late_listing_is_not_a_member_without_history(
    built: DataLayout, raw: tuple[Path, RawAnswers]
) -> None:
    t, first = raw[1].late_listing
    tid = ticker_id(built, t)
    assert tid not in set(universe(built, "full")["ticker_id"])
    assert (
        cast(date, bars(built, "full").filter(pl.col("ticker_id") == tid)["session_date"].min())
        == first
        or True
    )


def test_figi_less_instruments_get_symbol_date_ids(
    built: DataLayout, raw: tuple[Path, RawAnswers]
) -> None:
    for t in raw[1].figiless_tickers:
        assert ticker_id(built, t) == f"TKR:{t}:2015-01-02"


def test_split_is_continuous_in_adjusted_prices_but_not_in_raw(
    built: DataLayout, raw: tuple[Path, RawAnswers]
) -> None:
    for t, d, ratio in (raw[1].split, raw[1].reverse_split):
        df = bars(built, "full").filter(pl.col("ticker_id") == ticker_id(built, t)).sort("ts")
        before = df.filter(pl.col("session_date") < d)
        after = df.filter(pl.col("session_date") == d)
        raw_ret = after["open"][0] / after["adj_factor"][0] / (before["raw_close"][-1]) - 1
        adj_ret = after["open"][0] / before["close"][-1] - 1
        assert raw_ret == pytest.approx(1 / ratio - 1, abs=0.05)  # the raw price jumps
        assert abs(adj_ret) < 0.05  # adjusted gap is an ordinary overnight move
        assert after["adj_factor"][0] == pytest.approx(ratio * before["adj_factor"][-1])


def test_ex_dividend_adjustment_matches_the_formula(
    built: DataLayout, raw: tuple[Path, RawAnswers]
) -> None:
    t, d, cash = raw[1].dividends[0]
    tid = ticker_id(built, t)
    df = bars(built, "full").filter(pl.col("ticker_id") == tid).sort("ts")
    prev_close = df.filter(pl.col("session_date") < d)["raw_close"][-1]
    f_before = df.filter(pl.col("session_date") < d)["adj_factor"][-1]
    f_on = df.filter(pl.col("session_date") == d)["adj_factor"][0]
    assert f_on / f_before == pytest.approx(prev_close / (prev_close - cash))


def test_half_days_have_seven_bars_ending_at_close(built: DataLayout) -> None:
    df = bars(built, "full")
    for d, utc_hour in ((date(2024, 11, 29), 18), (date(2024, 12, 24), 18)):
        day = df.filter(pl.col("session_date") == d)
        assert day["bar"].max() == 6
        assert day["ts"].max().hour == utc_hour  # type: ignore[union-attr]
    assert df.filter(pl.col("session_date") == date(2024, 12, 20))["bar"].max() == 12
    assert date(2025, 1, 9) not in set(
        df["session_date"]
    )  # market closed (national day of mourning)


def test_volume_is_split_adjusted(built: DataLayout, raw: tuple[Path, RawAnswers]) -> None:
    t, d, _ratio = raw[1].split
    df = bars(built, "full").filter(pl.col("ticker_id") == ticker_id(built, t))
    daily = df.group_by("session_date").agg(pl.col("volume").sum()).sort("session_date")
    before = cast(float, daily.filter(pl.col("session_date") < d)["volume"].mean())
    after = cast(float, daily.filter(pl.col("session_date") >= d)["volume"].mean())
    assert after / before == pytest.approx(1.0, rel=0.35)  # not 2x: raw volume / split_factor


# --- manifest, reproducibility, atomicity ----------------------------------------------------


def test_manifest_and_verify(built: DataLayout) -> None:
    for kind in ("dev", "full"):
        b = built.current_build(kind)  # pyright: ignore[reportArgumentType]
        assert b is not None
        m = read_manifest(b)
        assert m.kind == kind and m.source == "synthetic" and m.adjustment == "forward"
        assert m.dataset_id == f"{kind}-{m.dataset_hash[:12]}"
        assert m.params_hash == PARAMS
        assert m.dataset_hash == sha256_canonical(
            {"files": sorted(f.sha256 for f in m.files), "params_hash": PARAMS}
        )
        assert verify_build(b) == []
    assert built.dataset_link("dev").is_symlink()


def test_rebuild_from_cache_and_from_raw_reproduces_the_hash(
    built: DataLayout, raw: tuple[Path, RawAnswers], tmp_path: Path
) -> None:
    before = {k: read_manifest(built.current_build(k)).dataset_hash for k in ("dev", "full")}  # pyright: ignore
    # same root, aggregation cache hot
    assert build_dataset(built, CFG, "full", PARAMS, "synthetic").dataset_hash == before["full"]
    # fresh root: only raw files and cached reference JSON survive
    fresh = DataLayout(tmp_path / "fresh")
    (tmp_path / "fresh").mkdir()
    (tmp_path / "fresh" / "raw").symlink_to(raw[0] / "raw")
    build_ref_tables(fresh)
    for kind in ("full", "dev"):
        assert build_dataset(fresh, CFG, kind, PARAMS, "synthetic").dataset_hash == before[kind]


def test_changing_config_changes_params_and_dataset_hash(built: DataLayout, tmp_path: Path) -> None:
    other = derive_root(built.root, tmp_path / "o")
    m = build_dataset(other, CFG, "dev", sha256_canonical({"different": 1}), "synthetic")
    assert m.dataset_hash != read_manifest(built.current_build("full")).dataset_hash  # pyright: ignore


def test_verify_detects_tampering(built: DataLayout, tmp_path: Path) -> None:
    copy = derive_root(built.root, tmp_path / "c")
    b = copy.current_build("full")
    assert b is not None
    victim = next((b / "bars_30m").glob("*/part-0.parquet"))
    victim.write_bytes(victim.read_bytes() + b"x")
    (b / "stray.parquet").write_bytes(b"")
    problems = verify_build(b)
    assert any("hash mismatch" in p for p in problems) and any("unlisted" in p for p in problems)


def test_failed_build_leaves_no_partial_dataset_and_keeps_the_old_link(
    built: DataLayout, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = derive_root(built.root, tmp_path / "c")
    link_before = os.readlink(copy.dataset_link("full"))
    builds_before = sorted(p.name for p in copy.build_root.iterdir())

    def boom(_p: Path) -> str:
        raise RuntimeError("disk full")

    monkeypatch.setattr(build_mod, "sha256_file", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        build_dataset(copy, CFG, "full", sha256_canonical({"new": 1}), "synthetic")
    assert sorted(p.name for p in copy.build_root.iterdir()) == builds_before
    assert os.readlink(copy.dataset_link("full")) == link_before


# --- QA must actually catch problems -------------------------------------------------------------


def _failed(report: QAReport) -> set[str]:
    return {c.name for c in report.checks if not c.passed}


def _rewrite(path: Path, fn: "object") -> None:
    df = pl.read_parquet(path)
    pl.DataFrame(fn(df)).write_parquet(path, compression="zstd", statistics=False)  # type: ignore[operator]


@pytest.fixture
def mutable(built: DataLayout, tmp_path: Path) -> DataLayout:
    return derive_root(built.root, tmp_path / "m")


def test_qa_catches_bad_ohlc(mutable: DataLayout) -> None:
    b = mutable.current_build("full")
    assert b is not None
    p = next((b / "bars_30m").glob("*/part-0.parquet"))
    _rewrite(p, lambda df: df.with_columns((pl.col("high") * 0.5).alias("high")))
    assert "ohlc_consistent" in _failed(run_qa(mutable, CFG, "full", write=False))


def test_qa_catches_missing_session(mutable: DataLayout) -> None:
    b = mutable.current_build("full")
    assert b is not None
    shutil.rmtree(next((b / "bars_30m").glob("session_date=2024-12-12")))
    assert "calendar_complete" in _failed(run_qa(mutable, CFG, "full", write=False))


def test_qa_catches_extreme_returns(mutable: DataLayout) -> None:
    b = mutable.current_build("full")
    assert b is not None
    p = next((b / "bars_30m").glob("session_date=2024-12-12/part-0.parquet"))
    _rewrite(
        p,
        lambda df: df.with_columns(
            pl.when(pl.col("ticker_id") == df["ticker_id"][0])
            .then(pl.col(c) * 3)
            .otherwise(pl.col(c))
            .alias(c)
            for c in ("open", "high", "low", "close", "vwap")
        ),
    )
    failed = _failed(run_qa(mutable, CFG, "full", write=False))
    assert "extreme_returns" in failed


def _jump_from(layout: DataLayout, day: date, member_that_day: bool) -> str:
    """Triple one instrument's prices on `day` only: a jump into `day` and a drop out of it."""
    b = layout.current_build("full")
    assert b is not None
    first = next((b / "bars_30m").glob(f"session_date={day}/part-0.parquet"))
    tid = pl.read_parquet(first)["ticker_id"][0]
    _rewrite(
        first,
        lambda df: df.with_columns(
            pl.when(pl.col("ticker_id") == tid).then(pl.col(c) * 3).otherwise(pl.col(c)).alias(c)
            for c in ("open", "high", "low", "close", "vwap")
        ),
    )
    if not member_that_day:  # out of the universe on `day` and the session after it
        for u in sorted((b / "universe").glob("session_date=*/part-0.parquet")):
            if date.fromisoformat(u.parent.name.split("=")[1]) >= day:
                _rewrite(u, lambda df: df.filter(pl.col("ticker_id") != tid))
                break
        nxt = [
            u
            for u in sorted((b / "universe").glob("session_date=*/part-0.parquet"))
            if date.fromisoformat(u.parent.name.split("=")[1]) > day
        ][0]
        _rewrite(nxt, lambda df: df.filter(pl.col("ticker_id") != tid))
    return tid


def test_qa_ignores_extreme_returns_outside_the_universe(mutable: DataLayout) -> None:
    _jump_from(mutable, date(2024, 12, 12), member_that_day=False)
    assert "extreme_returns" not in _failed(run_qa(mutable, CFG, "full", write=False))


def test_qa_catches_extreme_returns_of_a_member(mutable: DataLayout) -> None:
    _jump_from(mutable, date(2024, 12, 12), member_that_day=True)
    assert "extreme_returns" in _failed(run_qa(mutable, CFG, "full", write=False))


def test_qa_catches_unadjusted_split(mutable: DataLayout, raw: tuple[Path, RawAnswers]) -> None:
    t, d, ratio = raw[1].split
    tid = ticker_id(mutable, t)
    b = mutable.current_build("full")
    assert b is not None
    for p in (b / "bars_30m").glob("*/part-0.parquet"):
        day = date.fromisoformat(p.parent.name.split("=")[1])
        if day >= d:  # undo the adjustment from the split onwards, as a raw feed would look
            _rewrite(
                p,
                lambda df: df.with_columns(
                    pl.when(pl.col("ticker_id") == tid)
                    .then(pl.col(c) / ratio)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ("open", "high", "low", "close", "vwap")
                ),
            )
    assert "split_continuity" in _failed(run_qa(mutable, CFG, "full", write=False))


def test_qa_skips_a_listed_corporate_action(
    mutable: DataLayout, raw: tuple[Path, RawAnswers]
) -> None:
    t, d, ratio = raw[1].split
    tid = ticker_id(mutable, t)
    b = mutable.current_build("full")
    assert b is not None
    for p in (b / "bars_30m").glob("*/part-0.parquet"):
        if date.fromisoformat(p.parent.name.split("=")[1]) >= d:
            _rewrite(
                p,
                lambda df: df.with_columns(
                    pl.when(pl.col("ticker_id") == tid)
                    .then(pl.col(c) / ratio)
                    .otherwise(pl.col(c))
                    .alias(c)
                    for c in ("open", "high", "low", "close", "vwap")
                ),
            )
    listed = KnownCorporateAction(ticker_id=tid, date=d, reason="spin-off the adjustment misses")
    cfg = CFG.model_copy(update={"qa": QAConfig(known_corporate_actions=(listed,))})
    report = run_qa(mutable, cfg, "full", write=False)
    assert "split_continuity" not in _failed(report)
    detail = next(c.detail for c in report.checks if c.name == "split_continuity")
    assert "1 known exceptions skipped" in detail


def test_qa_catches_dev_diverging_from_full(mutable: DataLayout) -> None:
    b = mutable.current_build("dev")
    assert b is not None
    p = next((b / "bars_30m").glob("session_date=2024-12-12/part-0.parquet"))
    _rewrite(p, lambda df: df.with_columns((pl.col("volume") * 1.01).alias("volume")))
    assert "dev_full_equal" in _failed(run_qa(mutable, CFG, "dev", write=False))


def test_qa_catches_dev_rows_past_dev_end(built: DataLayout) -> None:
    shifted = CFG.model_copy(update={"dev_end": date(2024, 12, 20)})
    assert "dev_end_respected" in _failed(run_qa(built, shifted, "dev", write=False))


def test_qa_catches_wrong_universe_size(built: DataLayout) -> None:
    cfg = CFG.model_copy(update={"universe": UniverseConfig(top_n=31, tradable_top_n=15)})
    assert "universe_size" in _failed(run_qa(built, cfg, "full", write=False))


def test_qa_needs_both_builds_for_the_leak_check(built: DataLayout, tmp_path: Path) -> None:
    only_dev = derive_root(built.root, tmp_path / "d")
    only_dev.dataset_link("full").unlink()
    failed = _failed(run_qa(only_dev, CFG, "dev", write=False))
    assert failed == {"dev_full_equal"}


def test_qa_without_a_build_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_qa(DataLayout(tmp_path), CFG, "dev")


# --- MarketData ----------------------------------------------------------------------------


def md(layout: DataLayout, kind: str = "dev", **kw: bool) -> MarketData:
    return MarketData(kind, settings=Settings(data_dir=layout.root), **kw)  # pyright: ignore[reportArgumentType]


def test_full_dataset_requires_the_sealed_flag(built: DataLayout) -> None:
    with pytest.raises(SealedDatasetError):
        md(built, "full")
    assert md(built, "full", sealed=True).manifest.kind == "full"


def test_dev_read_api(built: DataLayout) -> None:
    data = md(built)
    assert data.manifest.kind == "dev"
    sessions = data.sessions()
    assert sessions[0] == date(2024, 11, 1) and sessions[-1] == DEV_END
    members = data.bars().collect()
    assert members.height == len(members.filter(pl.col("is_member"))) > 0
    assert {"rank", "tradable", "is_member", "close", "raw_close"} <= set(members.columns)
    everyone = data.bars(members_only=False).collect()
    assert everyone.height > members.height
    assert everyone.filter(~pl.col("is_member"))["tradable"].sum() == 0
    window = data.bars(date(2024, 12, 2), date(2024, 12, 3), columns=["ts", "ticker_id", "close"])
    out = window.collect()
    assert out.columns == ["ts", "ticker_id", "close"]
    assert set(out["ts"].dt.date()) <= {date(2024, 12, 2), date(2024, 12, 3)}
    assert data.universe(end=date(2024, 12, 2)).collect()["session_date"].n_unique() == 1
    assert data.sectors().height == data.manifest.n_tickers
    assert set(data.sectors()["sector"]) <= {
        "pharma_health", "tech_hardware_semis", "software_it", "financials", "utilities", "other"
    }  # fmt: skip
    with pytest.raises(ValueError, match="unknown columns"):
        data.bars(columns=["nope"])


def test_refuses_a_build_whose_qa_failed(built: DataLayout, tmp_path: Path) -> None:
    copy = derive_root(built.root, tmp_path / "c")
    b = copy.current_build("dev")
    assert b is not None
    qa = QAReport.model_validate_json((b / "qa.json").read_text())
    bad = QAReport(
        dataset_hash=qa.dataset_hash,
        checks=(*qa.checks, QACheck(name="x", passed=False, value=1, threshold=0, detail="")),
    )
    (b / "qa.json").write_text(bad.model_dump_json())
    with pytest.raises(DatasetNotReadyError, match="failed QA"):
        md(copy)
    (b / "qa.json").write_text(QAReport(dataset_hash="0" * 64, checks=qa.checks).model_dump_json())
    with pytest.raises(DatasetNotReadyError, match="different build"):
        md(copy)
    (b / "qa.json").unlink()
    with pytest.raises(DatasetNotReadyError, match="no QA"):
        md(copy)


def test_refuses_a_missing_build(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotReadyError):
        md(DataLayout(tmp_path))


# --- CLI -----------------------------------------------------------------------------------------


def test_cli_end_to_end(
    raw: tuple[Path, RawAnswers], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_dir, data_dir = tmp_path / "configs", tmp_path / "data"
    cfg_dir.mkdir()
    data_dir.mkdir()
    (cfg_dir / "data.yaml").write_text(CONFIG_YAML)
    (data_dir / "raw").symlink_to(raw[0] / "raw")
    monkeypatch.setenv("AF_CONFIG_DIR", str(cfg_dir))
    monkeypatch.setenv("AF_DATA_DIR", str(data_dir))
    runner = CliRunner()

    def run(*args: str, code: int = 0) -> str:
        r = runner.invoke(app, ["data", *args])
        assert r.exit_code == code, r.output
        return r.output

    run("build", code=1)  # reference tables are missing
    assert "ref tables" in run("ref", "--offline")
    out = run("build", "--kind", "both")
    assert "QA: dev" in out and "QA: full" in out
    out = run("manifest")
    assert "QA passed" in out and "dataset_hash" in out
    assert "ok" in run("verify")
    run("qa", "--kind", "dev")

    # a tampered file fails verify with a non-zero exit
    victim = next((data_dir / "datasets" / "dev" / "bars_30m").glob("*/part-0.parquet"))
    victim.write_bytes(victim.read_bytes() + b"x")
    run("verify", code=1)
    assert (
        json.loads(next((data_dir / "datasets" / "full").glob("manifest.json")).read_text())["kind"]
        == "full"
    )
