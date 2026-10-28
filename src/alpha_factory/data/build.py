"""Build a dataset: bars, adjustment factors, universe, sectors and a manifest, atomically."""

import os
import shutil
import uuid
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from alpha_factory.contracts.data import DataManifest, FileEntry
from alpha_factory.data.adjust import apply_adjustment, compute_adj_factors
from alpha_factory.data.bars import SessionStats, ensure_session_cache
from alpha_factory.data.calendar import TradingCalendar
from alpha_factory.data.config import DataConfig
from alpha_factory.data.layout import DataLayout, Kind
from alpha_factory.data.sectors import sector_for_sic
from alpha_factory.data.universe import build_universe
from alpha_factory.gitinfo import git_commit
from alpha_factory.hashing import sha256_bytes, sha256_canonical, sha256_file
from alpha_factory.logging import get_logger

log = get_logger(__name__)
WORKERS = min(8, os.cpu_count() or 1)
DV_SCALE = 10_000


def _pmap[A, B](fn: Callable[[A], B], items: Iterable[A]) -> list[B]:
    with ThreadPoolExecutor(WORKERS) as ex:
        return list(ex.map(fn, items))


def _write(df: pl.DataFrame, path: Path) -> tuple[Path, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=False)
    return path, df.height


def read_ref(layout: DataLayout, name: str) -> pl.DataFrame:
    return pl.read_parquet(layout.ref_table(name))


def build_dataset(
    layout: DataLayout,
    cfg: DataConfig,
    kind: Kind,
    params_hash: str,
    source: Literal["massive", "synthetic"] = "massive",
) -> DataManifest:
    """Build one dataset kind and point datasets/<kind> at it.

    The same code builds dev and full; dev just stops at cfg.dev_end. Raw 30-min bars are
    aggregated once per session into a shared cache, so building both kinds aggregates once.
    """
    cal = TradingCalendar(cfg.start, cfg.end)
    last = cfg.end if kind == "full" else cfg.dev_end
    sessions = cal.sessions(cfg.start, last)
    instruments = read_ref(layout, "instruments")
    symbols = read_ref(layout, "symbols")
    splits, dividends, sic = (read_ref(layout, n) for n in ("splits", "dividends", "sic"))
    symbols_key = sha256_bytes(
        layout.ref_table("instruments").read_bytes() + layout.ref_table("symbols").read_bytes()
    )

    # 1. raw 30-min bars per session (cached; skipped when up to date)
    def aggregate(d: date) -> tuple[date, SessionStats | None]:
        return d, ensure_session_cache(
            layout, d, cal.n_bars(d), symbols, instruments, symbols_key, cfg
        )

    stats = dict(_pmap(aggregate, sessions))
    present = [d for d in sessions if stats[d] is not None]
    if not present:
        raise RuntimeError("no raw minute files found for the requested range")
    log.info("aggregated", kind=kind, sessions=len(present), missing=len(sessions) - len(present))

    # 2. daily table, universe, instruments that ever become members
    def daily_for(d: date) -> pl.DataFrame:
        # Per-session so float sums never depend on which other sessions are in the build.
        return (
            pl.read_parquet(layout.bars_cache(d))  # sorted by (ticker_id, bar)
            .group_by("ticker_id", "session_date", maintain_order=True)
            .agg(
                pl.col("close").last().alias("raw_close"),
                # Integer 1e-4 dollars: exact sums regardless of thread scheduling.
                (pl.col("close") * pl.col("volume") * DV_SCALE)
                .round()
                .cast(pl.Int64)
                .sum()
                .alias("dollar_volume"),
            )
            .with_columns(pl.col("dollar_volume") / DV_SCALE)
        )

    daily = pl.concat(_pmap(daily_for, present))
    universe = build_universe(daily, instruments, sessions, cfg.universe)
    ever = sorted(universe["ticker_id"].unique().to_list())

    # 3. causal adjustment factors for those instruments
    factors = compute_adj_factors(
        daily.filter(pl.col("ticker_id").is_in(ever)), splits, dividends, sessions[0]
    )
    factors_by_session = factors.partition_by("session_date", as_dict=True)
    universe_by_session = universe.partition_by("session_date", as_dict=True)

    tmp = layout.build_root / f".tmp-{uuid.uuid4().hex}"
    written: list[tuple[Path, int]] = []

    def write_session(d: date) -> list[tuple[Path, int]]:
        out: list[tuple[Path, int]] = []
        f = factors_by_session.get((d,))
        if f is not None:
            raw = pl.read_parquet(layout.bars_cache(d)).filter(pl.col("ticker_id").is_in(ever))
            bars = apply_adjustment(raw, f)
            out.append(_write(bars, tmp / "bars_30m" / f"session_date={d}" / "part-0.parquet"))
        u = universe_by_session.get((d,))
        if u is not None:
            out.append(_write(u, tmp / "universe" / f"session_date={d}" / "part-0.parquet"))
        return out

    try:
        for part in _pmap(write_session, present):
            written.extend(part)

        sectors = (
            instruments.filter(pl.col("ticker_id").is_in(ever))
            .join(sic, on="ticker_id", how="left")
            .select("ticker_id", "sic_code")
            .sort("ticker_id")
        )
        sectors = sectors.with_columns(
            pl.Series(
                "sector", [sector_for_sic(c) for c in sectors["sic_code"].to_list()], pl.String
            )
        )
        present_stats = [st for d in present if (st := stats[d]) is not None]
        session_stats = pl.DataFrame(
            {
                "session_date": [st.session_date for st in present_stats],
                "cs_dollar_volume": [st.cs_dollar_volume for st in present_stats],
                "unmapped_dollar_volume": [st.unmapped_dollar_volume for st in present_stats],
            }
        )
        written.append(_write(factors, tmp / "adj_factors.parquet"))
        written.append(_write(sectors, tmp / "sectors.parquet"))
        written.append(_write(session_stats, tmp / "session_stats.parquet"))

        entries = sorted(
            (
                FileEntry(
                    path=p.relative_to(tmp).as_posix(),
                    sha256=sha256_file(p),
                    rows=rows,
                    bytes=p.stat().st_size,
                )
                for p, rows in written
            ),
            key=lambda e: e.path,
        )
        dataset_hash = sha256_canonical(
            {"files": sorted(e.sha256 for e in entries), "params_hash": params_hash}
        )
        manifest = DataManifest(
            dataset_id=f"{kind}-{dataset_hash[:12]}",
            dataset_hash=dataset_hash,
            kind=kind,
            source=source,
            start=present[0],
            end=present[-1],
            n_sessions=len(present),
            n_tickers=len(ever),
            bar_minutes=30,
            adjustment="forward",
            params_hash=params_hash,
            code_commit=git_commit(),
            created_at=datetime.now(UTC),
            files=tuple(entries),
        )
        # The manifest is the last file written: a build without one is incomplete.
        (tmp / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        final = layout.build_dir(manifest.dataset_id)
        if final.exists():
            shutil.rmtree(tmp)  # identical content already built
        else:
            tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    point_link(layout, kind, manifest.dataset_id)
    return manifest


def point_link(layout: DataLayout, kind: Kind, dataset_id: str) -> None:
    """Atomically repoint datasets/<kind> at build/<dataset_id>."""
    layout.datasets_root.mkdir(parents=True, exist_ok=True)
    link, tmp = layout.dataset_link(kind), layout.datasets_root / f".{kind}.{uuid.uuid4().hex}"
    tmp.symlink_to(Path("..") / "build" / dataset_id)
    os.replace(tmp, link)


def read_manifest(build_dir: Path) -> DataManifest:
    return DataManifest.model_validate_json((build_dir / "manifest.json").read_text())


def verify_build(build_dir: Path) -> list[str]:
    """Re-hash every file against the manifest. Returns human-readable problems (empty = ok)."""
    manifest = read_manifest(build_dir)
    problems: list[str] = []
    for e in manifest.files:
        p = build_dir / e.path
        if not p.exists():
            problems.append(f"missing: {e.path}")
        elif sha256_file(p) != e.sha256:
            problems.append(f"hash mismatch: {e.path}")
    known = {e.path for e in manifest.files} | {"manifest.json", "qa.json"}
    for p in build_dir.rglob("*.parquet"):
        if p.relative_to(build_dir).as_posix() not in known:
            problems.append(f"unlisted: {p.relative_to(build_dir)}")
    return problems
