"""The synthetic market written in Plan 2's dataset layout, so MarketData can open it.

`write_datasets` produces build/<id>/{bars_30m,universe}/session_date=D/part-0.parquet,
adj_factors.parquet, sectors.parquet, manifest.json and a QA report, and points
datasets/dev and datasets/full at them. QA is a single "synthetic_fixture" check: the real
checks need reference tables the synthetic market does not have.

`synthetic_sandbox` builds a SandboxSpec whose zones fit the synthetic calendar.
"""

import shutil
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from alpha_factory.contracts.data import DataManifest, FileEntry, QACheck, QAReport
from alpha_factory.contracts.evaluation import GateSpec
from alpha_factory.contracts.sandbox import (
    CostSpec,
    DateRange,
    Fold,
    LabelSpec,
    PortfolioSpec,
    SandboxSpec,
    UniverseSpec,
)
from alpha_factory.data.build import point_link
from alpha_factory.data.layout import DataLayout
from alpha_factory.hashing import sha256_canonical, sha256_file
from alpha_factory.synthetic.market import SyntheticMarket

FOLD_SESSIONS = 70
HOLDOUT_SESSIONS = 60
WARMUP_SESSIONS = 21


def zones(market: SyntheticMarket) -> dict[str, DateRange]:
    """Fold, holdout and sealed ranges laid over the synthetic sessions."""
    s = market.sessions
    out: dict[str, DateRange] = {}
    i = WARMUP_SESSIONS
    for k in range(1, 5):
        out[f"F{k}"] = DateRange(start=s[i], end=s[i + FOLD_SESSIONS - 1])
        i += FOLD_SESSIONS
    out["holdout"] = DateRange(start=s[i], end=s[i + HOLDOUT_SESSIONS - 1])
    i += HOLDOUT_SESSIONS - 1  # last holdout session
    out["sealed"] = DateRange(start=s[i + 6], end=s[-1])  # 5 sessions of embargo between
    return out


def synthetic_sandbox(market: SyntheticMarket, dataset_hash: str) -> SandboxSpec:
    z = zones(market)
    return SandboxSpec(
        sandbox_version="synthetic-1",
        dataset_hash=dataset_hash,
        universe=UniverseSpec(top_n=150, adv_window_days=20, min_price=5.0, tradable_top_n=100),
        labels=(
            LabelSpec(name="h1", hours=1),
            LabelSpec(name="h2", hours=2),
            LabelSpec(name="h4", hours=4),
        ),
        folds=tuple(Fold(fold_id=k, train=None, val=z[k]) for k in ("F1", "F2", "F3", "F4")),
        gate_holdout=z["holdout"],
        sealed_test=z["sealed"],
        costs=CostSpec(sell_fee_bps=0.3),
        portfolio=PortfolioSpec(),
        gate=GateSpec(),
    )


def _write(written: list[tuple[Path, int]], df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd", statistics=False)
    written.append((path, df.height))


def write_datasets(
    market: SyntheticMarket, root: Path, *, dev_end: date | None = None
) -> dict[Literal["dev", "full"], DataManifest]:
    """Write dev (sessions up to dev_end, default the end of the holdout) and full builds."""
    dev_end = dev_end or zones(market)["holdout"].end
    frames = market.frames()
    layout = DataLayout(root)
    params_hash = sha256_canonical(market._params())  # pyright: ignore[reportPrivateUsage]
    out: dict[Literal["dev", "full"], DataManifest] = {}
    for kind in ("dev", "full"):
        tmp = layout.build_root / f".tmp-{kind}"
        shutil.rmtree(tmp, ignore_errors=True)
        written: list[tuple[Path, int]] = []

        bars = (
            frames.bars.filter(pl.col("session_date") <= dev_end) if kind == "dev" else frames.bars
        )
        universe = (
            frames.universe.filter(pl.col("session_date") <= dev_end)
            if kind == "dev"
            else frames.universe
        )
        for d, part in bars.partition_by("session_date", as_dict=True).items():
            _write(written, part, tmp / "bars_30m" / f"session_date={d[0]}" / "part-0.parquet")
        for d, part in universe.partition_by("session_date", as_dict=True).items():
            _write(written, part, tmp / "universe" / f"session_date={d[0]}" / "part-0.parquet")
        factors = (
            bars.select("ticker_id", "session_date", "adj_factor")
            .unique()
            .with_columns(
                pl.col("adj_factor").alias("split_factor"), pl.lit(1.0).alias("div_factor")
            )
            .select("ticker_id", "session_date", "split_factor", "div_factor", "adj_factor")
            .sort("ticker_id", "session_date")
        )
        _write(written, factors, tmp / "adj_factors.parquet")
        _write(
            written,
            frames.sectors.with_columns(pl.lit(None, pl.String).alias("sic_code")).select(
                "ticker_id", "sic_code", "sector"
            ),
            tmp / "sectors.parquet",
        )
        entries = tuple(
            sorted(
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
        )
        dataset_hash = sha256_canonical(
            {"files": sorted(e.sha256 for e in entries), "params_hash": params_hash}
        )
        sessions = [d for d in market.sessions if kind == "full" or d <= dev_end]
        manifest = DataManifest(
            dataset_id=f"{kind}-{dataset_hash[:12]}",
            dataset_hash=dataset_hash,
            kind=kind,
            source="synthetic",
            start=sessions[0],
            end=sessions[-1],
            n_sessions=len(sessions),
            n_tickers=market.n_tickers,
            bar_minutes=30,
            adjustment="forward",
            params_hash=params_hash,
            code_commit="synthetic",
            created_at=datetime(2000, 1, 1, tzinfo=UTC),
            files=entries,
        )
        (tmp / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        qa = QAReport(
            dataset_hash=dataset_hash,
            checks=(
                QACheck(
                    name="synthetic_fixture",
                    passed=True,
                    value=0,
                    threshold=0,
                    detail="synthetic data; the real QA checks need reference tables",
                ),
            ),
        )
        (tmp / "qa.json").write_text(qa.model_dump_json(indent=2))
        final = layout.build_dir(manifest.dataset_id)
        if final.exists():
            shutil.rmtree(tmp)
        else:
            tmp.rename(final)
        point_link(layout, kind, manifest.dataset_id)
        out[kind] = manifest
    return out
