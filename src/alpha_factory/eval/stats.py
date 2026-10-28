"""Daily IC series and the statistics computed from it.

Intraday ICs within one session are strongly correlated, so every statistic works on the daily
IC: the mean of the session's bar ICs. Treating bars as independent would inflate t-stats.
"""

from dataclasses import dataclass
from datetime import date
from math import isnan, sqrt

import numpy as np
import polars as pl

from alpha_factory.contracts.evaluation import FoldStat
from alpha_factory.contracts.sandbox import DateRange, Fold

NW_LAG = 5


def intraday_ics(
    scores: pl.DataFrame, labels: pl.DataFrame, label: str, min_names: int
) -> pl.DataFrame:
    """Cross-sectional Spearman IC at every timestamp: ts, session_date, n, ic.

    scores: tix, ts, session_date, score. labels: tix, ts and the label column. Names with a
    missing score or label at t are dropped; timestamps with fewer than `min_names` are skipped.
    """
    joined = (
        scores.select("tix", "ts", "session_date", "score")
        .join(labels.select("tix", "ts", label), on=["tix", "ts"])
        .drop_nulls(["score", label])
    )
    ranked = joined.with_columns(
        pl.col("score").rank("average").over("ts").alias("rs"),
        pl.col(label).rank("average").over("ts").alias("ry"),
    )
    return (
        ranked.group_by("ts", "session_date")
        .agg(pl.len().alias("n"), pl.corr("rs", "ry").alias("ic"))
        .filter((pl.col("n") >= min_names) & pl.col("ic").is_finite())
        .sort("ts")
    )


def daily_ic(ics: pl.DataFrame, sign: int = 1) -> pl.DataFrame:
    """session_date, ic: the mean bar IC of each session, multiplied by the sign used."""
    return (
        ics.group_by("session_date")
        .agg(pl.col("ic").mean() * sign)
        .rename({"ic": "ic"})
        .sort("session_date")
    )


def newey_west_tstat(x: np.ndarray, lag: int = NW_LAG) -> float:
    """mean(x) / sqrt(Omega / D) with Omega = g0 + 2 * sum_l (1 - l/(lag+1)) * g_l."""
    d = len(x)
    if d < lag + 2:
        return 0.0
    c = x - x.mean()
    omega = float(np.dot(c, c)) / d
    for l in range(1, lag + 1):  # noqa: E741 - lag index
        omega += 2 * (1 - l / (lag + 1)) * float(np.dot(c[l:], c[:-l])) / d
    if omega <= 1e-20:  # no variance, or rounding noise from a constant series
        return 0.0
    return float(x.mean() / sqrt(omega / d))


def icir(x: np.ndarray) -> float:
    """Mean over standard deviation of the daily IC (not annualised)."""
    if len(x) < 2:
        return 0.0
    sd = float(x.std(ddof=1))
    return float(x.mean() / sd) if sd > 0 else 0.0


@dataclass(frozen=True)
class SeriesStats:
    mean: float
    tstat: float
    icir: float
    n_days: int


def series_stats(x: np.ndarray) -> SeriesStats:
    if len(x) == 0:
        return SeriesStats(0.0, 0.0, 0.0, 0)
    return SeriesStats(float(x.mean()), newey_west_tstat(x), icir(x), len(x))


def in_range(d: date, r: DateRange) -> bool:
    return r.start <= d <= r.end


def fold_stats(daily: pl.DataFrame, folds: tuple[Fold, ...]) -> tuple[FoldStat, ...]:
    """Mean, Newey-West t and day count of the daily IC inside each fold's validation range."""
    out: list[FoldStat] = []
    for f in folds:
        x = daily.filter(pl.col("session_date").is_between(f.val.start, f.val.end))["ic"].to_numpy()
        s = series_stats(x)
        out.append(FoldStat(fold_id=f.fold_id, ic_mean=s.mean, ic_tstat=s.tstat, n_days=s.n_days))
    return tuple(out)


def by_year(daily: pl.DataFrame) -> dict[int, float]:
    g = daily.group_by(pl.col("session_date").dt.year().alias("year")).agg(pl.col("ic").mean())
    return {int(y): float(v) for y, v in zip(g["year"], g["ic"], strict=True)}


def finite(x: float | None, default: float = 0.0) -> float:
    return default if x is None or isnan(x) else float(x)


_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)  # fmt: skip
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)  # fmt: skip
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)  # fmt: skip
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


def norm_ppf(p: np.ndarray) -> np.ndarray:
    """Inverse standard normal CDF (Acklam's rational approximation, relative error ~1e-9)."""
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    low, high = 0.02425, 1 - 0.02425
    mid = (p >= low) & (p <= high)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (
        (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
        * q
        / (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)
    )
    for tail, sign in ((p < low, 1.0), (p > high, -1.0)):
        pt = p[tail] if sign > 0 else 1 - p[tail]
        qt = np.sqrt(-2 * np.log(pt))
        val = (((((_C[0] * qt + _C[1]) * qt + _C[2]) * qt + _C[3]) * qt + _C[4]) * qt + _C[5]) / (
            (((_D[0] * qt + _D[1]) * qt + _D[2]) * qt + _D[3]) * qt + 1
        )
        out[tail] = sign * val
    return out
