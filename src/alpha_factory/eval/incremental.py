"""Incremental IC: what a factor adds beyond the six baselines.

At each timestamp the factor and the baselines are turned into normal scores (rank, then the
inverse normal), the factor is regressed on the baselines across names by OLS, and the residual's
Spearman IC against the demeaned label is taken. Averaged by day like any other IC.
"""

from datetime import date, datetime
from itertools import pairwise

import numpy as np
import polars as pl

from alpha_factory.eval.baselines import BASELINES
from alpha_factory.eval.stats import norm_ppf


def incremental_ics(
    scores: pl.DataFrame, baselines: pl.DataFrame, labels: pl.DataFrame, label: str, min_names: int
) -> pl.DataFrame:
    """ts, session_date, n, ic of the baseline-residualised factor.

    scores must already carry the sign (tix, ts, session_date, score). labels is the demeaned
    label column of the horizon.
    """
    names = list(BASELINES)
    j = (
        scores.select("tix", "ts", "session_date", "score")
        .join(baselines, on=["tix", "ts"])
        .join(labels.select("tix", "ts", label), on=["tix", "ts"])
        .drop_nulls()
        .sort("ts", "tix")
    )
    cols = ["score", *names]
    j = j.with_columns(
        [
            ((pl.col(c).rank("average").over("ts") - 0.5) / pl.len().over("ts")).alias(f"p_{c}")
            for c in cols
        ]
    )
    z = {c: norm_ppf(j[f"p_{c}"].to_numpy()) for c in cols}
    y = j[label].to_numpy()
    ts = j["ts"].to_numpy()
    cuts = np.flatnonzero(np.r_[True, ts[1:] != ts[:-1], True])
    rows_ts: list[datetime] = []
    rows_day: list[date] = []
    rows_n: list[int] = []
    rows_ic: list[float] = []
    days = j["session_date"].to_list()
    ts_list = j["ts"].to_list()
    x_all = np.column_stack([np.ones(len(j)), *(z[c] for c in names)])
    f_all = z["score"]
    for lo, hi in pairwise(cuts):
        n = hi - lo
        if n < min_names:
            continue
        x, f = x_all[lo:hi], f_all[lo:hi]
        beta = np.linalg.lstsq(x, f, rcond=None)[0]
        resid: np.ndarray = f - x @ beta
        ry = _rank(y[lo:hi])
        rr = _rank(resid)
        sd = rr.std() * ry.std()
        if sd == 0:
            continue
        rows_ts.append(ts_list[lo])
        rows_day.append(days[lo])
        rows_n.append(int(n))
        if resid.std() < 1e-9 * max(f.std(), 1e-300):
            # The factor lies in the span of the baselines: its residual is rounding noise, whose
            # ranks are random. It adds nothing, so the incremental IC is exactly zero.
            rows_ic.append(0.0)
        else:
            rows_ic.append(float(((rr - rr.mean()) * (ry - ry.mean())).mean() / sd))
    return pl.DataFrame(
        {"ts": rows_ts, "session_date": rows_day, "n": rows_n, "ic": rows_ic},
        schema={
            "ts": pl.Datetime("us", "UTC"),
            "session_date": pl.Date,
            "n": pl.Int64,
            "ic": pl.Float64,
        },
    )


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    return pl.Series(x).rank("average").to_numpy()
