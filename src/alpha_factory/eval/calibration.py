"""Null calibration and the two sanity checks that run with it.

Null trials are calibration, not search: they do not count toward N and never enter the pool.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import polars as pl

from alpha_factory.contracts.base import Horizon, Record
from alpha_factory.contracts.data import BAR_SCHEMA
from alpha_factory.data.marketdata import MarketData
from alpha_factory.dsl import compile_expr, random_expression
from alpha_factory.eval.context import EvalContext
from alpha_factory.eval.evaluate import search_daily_ic
from alpha_factory.sandbox.causality import causality_violations
from alpha_factory.sandbox.panel import Panel

QUANTILE = 0.99
SHUFFLE_TOLERANCE = 0.002
CANARY_MULTIPLE = 5.0


class NullCalibration(Record):
    """IC floors from random formulas, keyed by the data and sandbox they were measured on."""

    sandbox_hash: str
    dataset_hash: str
    n_expressions: int
    seed: int
    quantile: float
    ic_floor: dict[Horizon, float]
    shuffle_mean_ic: float | None = None


def horizon_ic(scores: pl.DataFrame, ctx: EvalContext, horizon: Horizon) -> float:
    """Mean daily IC of unsigned scores over the search folds (0 when there is none)."""
    d = search_daily_ic(scores, ctx, horizon, 1)
    return float(d["ic"].mean()) if d.height else 0.0  # type: ignore[arg-type]


def run_null(
    ctx: EvalContext,
    n: int = 1000,
    seed: int = 0,
    max_depth: int = 4,
    progress: Callable[[int], None] | None = None,
) -> NullCalibration:
    """Evaluate n random expressions at every horizon (sign +1) and take the 99th percentile of
    the absolute mean IC per horizon as the floor."""
    rng = np.random.default_rng(seed)
    horizons: list[Horizon] = [s.name for s in ctx.sandbox.labels]
    seen: set[str] = set()
    ics: dict[Horizon, list[float]] = {h: [] for h in horizons}
    while len(seen) < n:
        factor = compile_expr(random_expression(rng, max_depth))
        if factor.expr_hash in seen:
            continue
        seen.add(factor.expr_hash)
        scores = (
            factor.run(ctx.panel.df)
            .filter(pl.col("session_date").is_between(ctx.start, ctx.end))
            .select("tix", "ts", "session_date", "score")
        )
        for h in horizons:
            ics[h].append(abs(horizon_ic(scores, ctx, h)))
        if progress:
            progress(len(seen))
    floor: dict[Horizon, float] = {h: float(np.quantile(v, QUANTILE)) for h, v in ics.items()}
    return NullCalibration(
        sandbox_hash=ctx.sandbox.content_hash(),
        dataset_hash=ctx.dataset_hash,
        n_expressions=n,
        seed=seed,
        quantile=QUANTILE,
        ic_floor=floor,
    )


def shuffle_labels(labels: pl.DataFrame, columns: list[str], seed: int) -> pl.DataFrame:
    """Permute each label column across names at every timestamp."""
    rng = np.random.default_rng(seed)
    df = labels.sort("ts", "tix")
    ts = df["ts"].to_numpy()
    cuts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1], True])
    out = df
    for col in columns:
        y = df[col].to_numpy().copy()
        for lo, hi in pairwise(cuts):
            y[lo:hi] = y[lo:hi][rng.permutation(hi - lo)]
        out = out.with_columns(pl.Series(col, y).fill_nan(None))  # missing stays null
    return out


def label_shuffle_check(
    ctx: EvalContext, horizon: Horizon, n_expressions: int = 20, seed: int = 0
) -> float:
    """Mean IC of random expressions against labels permuted across names. Must be about 0."""
    shuffled = shuffle_labels(ctx.labels, [f"y_{horizon}"], seed)
    shuffled_ctx = EvalContext(
        sandbox=ctx.sandbox, dataset_hash=ctx.dataset_hash, panel=ctx.panel, labels=shuffled,
        baselines=ctx.baselines, regimes=ctx.regimes, start=ctx.start, end=ctx.end, zone=ctx.zone,
    )  # fmt: skip
    rng = np.random.default_rng(seed + 1)
    vals: list[float] = []
    for _ in range(n_expressions):
        factor = compile_expr(random_expression(rng, 3))
        scores = (
            factor.run(ctx.panel.df)
            .filter(pl.col("session_date").is_between(ctx.start, ctx.end))
            .select("tix", "ts", "session_date", "score")
        )
        vals.append(horizon_ic(scores, shuffled_ctx, horizon))
    return float(np.mean(vals))


# --- the leakage canary -----------------------------------------------------------------------
def canary_feature(panel: Panel) -> pl.DataFrame:
    """The AQuA bug: cumulative intraday volume divided by the FULL-day volume.

    Not expressible in the DSL (that is the point): the denominator uses volume after t.
    """
    day = pl.col("volume").sum().over(["tix", "sidx"])
    cum = pl.col("volume").fill_null(0).cum_sum().over(["tix", "sidx"])
    return panel.df.select(
        "tix", "tsi", "ts", "session_date", "bar", "sidx", "is_member",
        pl.when(pl.col("is_member")).then(cum / day).alias("score"),
    )  # fmt: skip


@dataclass(frozen=True)
class CanaryReport:
    ic: float  # mean daily IC of the leaky feature at the checked horizon
    floor: float  # the null floor it is compared with
    ratio: float
    scores_high: bool  # ic above 5x the floor
    flagged_lookahead: bool  # the causality test caught it


def run_canary(
    ctx: EvalContext,
    data: MarketData,
    floor: float,
    horizon: Horizon = "h1",
    causality_sessions: int = 12,
) -> CanaryReport:
    """Score the leaky feature and run the causality test on it. Both should fire."""
    scores = canary_feature(ctx.panel).filter(pl.col("session_date").is_between(ctx.start, ctx.end))
    ic = horizon_ic(scores.select("tix", "ts", "session_date", "score"), ctx, horizon)

    # causality on the last few scored sessions, rebuilt from raw bars so the noise is real
    sessions = [d for d in data.sessions() if d <= ctx.end][-causality_sessions:]
    bars = data.bars(
        sessions[0], sessions[-1], columns=list(BAR_SCHEMA), members_only=False
    ).collect()
    universe = data.universe(sessions[0], sessions[-1]).collect()
    counts = bars.group_by("session_date").agg((pl.col("bar").max() + 1).alias("n"))
    session_bars = {d: int(n) for d, n in zip(counts["session_date"], counts["n"], strict=True)}
    cutoff: datetime = bars.filter(pl.col("session_date") == sessions[len(sessions) // 2])[
        "ts"
    ].min()  # type: ignore[assignment]
    violations = causality_violations(
        lambda p: canary_feature(p).select("tix", "ts", "score"),
        bars, universe, data.sectors(), session_bars, cutoff,
    )  # fmt: skip
    ratio = abs(ic) / floor if floor > 0 else float("inf")
    return CanaryReport(ic, floor, ratio, ratio > CANARY_MULTIPLE, violations > 0)


def calibration_path(data_dir: Path, sandbox_hash: str) -> Path:
    """Where null_calibration.json lives: one file per sandbox hash."""
    return data_dir / "calibration" / f"null_calibration_{sandbox_hash[:16]}.json"


def save_calibration(cal: NullCalibration, data_dir: Path) -> Path:
    path = calibration_path(data_dir, cal.sandbox_hash)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(cal.model_dump_json(indent=2))
    return path


def load_calibration(
    data_dir: Path, sandbox_hash: str, dataset_hash: str
) -> NullCalibration | None:
    """The saved calibration for this sandbox and dataset, or None if absent or stale."""
    path = calibration_path(data_dir, sandbox_hash)
    if not path.exists():
        return None
    cal = NullCalibration.model_validate_json(path.read_text())
    return cal if cal.sandbox_hash == sandbox_hash and cal.dataset_hash == dataset_hash else None
