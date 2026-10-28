"""Causality check: a value at time t must not change when data after t changes.

`causality_violations` rebuilds the panel from bars whose rows after a cutoff have been replaced
by noise, recomputes a feature, and counts the rows at or before the cutoff whose value moved
(nulls included). The DSL must score zero for every expression; the leakage canary must not.
"""

from collections.abc import Callable
from datetime import date, datetime

import numpy as np
import polars as pl

from alpha_factory.sandbox.panel import Panel, build_panel

PRICE_COLUMNS = ("open", "high", "low", "close", "vwap")
FEATURE_KEYS = ("tix", "ts", "score")


def perturb_after(bars: pl.DataFrame, cutoff: datetime, rng: np.random.Generator) -> pl.DataFrame:
    """Bars with prices, volume and trades after `cutoff` multiplied by random noise.

    raw_close and adj_factor are left alone: they are not DSL fields and membership is separate.
    """
    later = bars["ts"] > cutoff
    n = bars.height
    price_noise = np.exp(rng.normal(0.0, 0.3, n))
    size_noise = np.exp(rng.normal(0.0, 0.5, n))
    noisy = [
        pl.when(later).then(pl.col(c) * pl.Series(price_noise)).otherwise(pl.col(c)).alias(c)
        for c in PRICE_COLUMNS
    ]
    noisy += [
        pl.when(later).then(pl.col("volume") * pl.Series(size_noise)).otherwise(pl.col("volume")),
        pl.when(later)
        .then((pl.col("trades") * pl.Series(size_noise)).round().cast(pl.Int64))
        .otherwise(pl.col("trades")),
    ]
    return bars.with_columns(noisy)


def causality_violations(
    feature: Callable[[Panel], pl.DataFrame],
    bars: pl.DataFrame,
    universe: pl.DataFrame,
    sectors: pl.DataFrame,
    session_bars: dict[date, int],
    cutoff: datetime,
    seed: int = 0,
) -> int:
    """Rows at or before `cutoff` whose feature value changed when later data changed.

    `feature(panel)` returns a frame with tix, ts and score columns.
    """
    tickers = sorted(bars["ticker_id"].unique().to_list())
    before = feature(build_panel(bars, universe, sectors, session_bars, tickers))
    noisy = perturb_after(bars, cutoff, np.random.default_rng(seed))
    after = feature(build_panel(noisy, universe, sectors, session_bars, tickers))
    joined = before.select(FEATURE_KEYS).join(
        after.select(FEATURE_KEYS), on=["tix", "ts"], suffix="_after", how="full", coalesce=True
    )
    early = joined.filter(pl.col("ts") <= cutoff)
    return early.filter(~pl.col("score").eq_missing(pl.col("score_after"))).height
