"""Two causal market regimes, each with two states, from the equal-weight universe return.

For session d only the 20 sessions BEFORE d are used: volatility is high when the standard
deviation of their daily returns exceeds the expanding median of earlier values, and the trend is
up when their cumulative return is positive. Sessions without enough history have no state.
"""

import numpy as np
import polars as pl

LOOKBACK = 20
MIN_HISTORY = 20  # earlier volatility values needed before an expanding median is meaningful
STATES = ("vol_high", "vol_low", "trend_up", "trend_down")


def universe_daily_returns(panel: pl.DataFrame) -> pl.DataFrame:
    """session_date, ret: the mean close-to-close return of the day's universe members."""
    return (
        panel.filter((pl.col("bar") == pl.col("last_bar")) & pl.col("is_member"))
        .with_columns((pl.col("close") / pl.col("prevclose") - 1).alias("r"))
        .group_by("session_date")
        .agg(pl.col("r").drop_nulls().mean().alias("ret"))
        .drop_nulls("ret")
        .sort("session_date")
    )


def regime_table(panel: pl.DataFrame) -> pl.DataFrame:
    """session_date, vol_high (bool|null), trend_up (bool|null)."""
    daily = universe_daily_returns(panel)
    r = daily["ret"].to_numpy()
    n = len(r)
    vol = np.full(n, np.nan)
    trend = np.full(n, np.nan)
    for d in range(LOOKBACK, n):
        window = r[d - LOOKBACK : d]
        vol[d] = window.std(ddof=1)
        trend[d] = np.prod(1 + window) - 1
    high: list[bool | None] = [None] * n
    up: list[bool | None] = [None] * n
    for d in range(n):
        if np.isnan(vol[d]):
            continue
        history = vol[:d][~np.isnan(vol[:d])]
        if len(history) >= MIN_HISTORY:
            high[d] = bool(vol[d] > np.median(history))
        up[d] = bool(trend[d] > 0)
    return pl.DataFrame(
        {"session_date": daily["session_date"], "vol_high": high, "trend_up": up},
        schema={"session_date": pl.Date, "vol_high": pl.Boolean, "trend_up": pl.Boolean},
    )


def ic_by_regime(
    daily: pl.DataFrame, regimes: pl.DataFrame, min_days: int = 10
) -> dict[str, float]:
    """Mean daily IC in each of the four states. A state with fewer than `min_days` days is 0."""
    j = daily.join(regimes, on="session_date", how="left")
    out: dict[str, float] = {}
    for state, col, want in (
        ("vol_high", "vol_high", True),
        ("vol_low", "vol_high", False),
        ("trend_up", "trend_up", True),
        ("trend_down", "trend_up", False),
    ):
        x = j.filter(pl.col(col) == want)["ic"].drop_nulls()
        out[state] = float(x.mean()) if len(x) >= min_days else 0.0  # type: ignore[arg-type]
    return out
