"""Combining factor scores into one signal: rank average and walk-forward ridge.

Lives in eval/, not research/, because the evaluation worker and final_eval both need it and the
sealed side must never import the research loop.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from math import ceil

import numpy as np
import polars as pl

from alpha_factory.contracts.sandbox import DateRange

KEYS = ("tix", "ts", "session_date", "tsi")


def rank_features(frames: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """One centred cross-sectional rank column per factor, in (-0.5, 0.5), null where missing.

    Every frame is a factor's run over the same panel, so they share their key rows.
    """
    out: pl.DataFrame | None = None
    for fid, df in frames.items():
        if out is None:
            out = df.select(*KEYS, pl.col("score").alias(fid))
        else:
            part = df.select("tix", "ts", pl.col("score").alias(fid)).unique(["tix", "ts"])
            out = out.join(part, on=["tix", "ts"], how="left")
    if out is None:
        raise ValueError("no factors to combine")
    ranks = [
        ((pl.col(f).rank("average").over("ts") / pl.col(f).count().over("ts")) - 0.5).alias(f)
        for f in frames
    ]
    return out.with_columns(ranks)


def rank_average(frames: Mapping[str, pl.DataFrame], signs: Mapping[str, int]) -> pl.DataFrame:
    """Mean over factors of sign x centred rank. Needs at least half the factors present."""
    feats = rank_features(frames)
    names = list(frames)
    signed = [(pl.col(f) * signs[f]).alias(f) for f in names]
    need = ceil(len(names) / 2)
    return (
        feats.with_columns(signed)
        .with_columns(
            pl.mean_horizontal(names).alias("score"),
            pl.sum_horizontal([pl.col(f).is_not_null().cast(pl.Int32) for f in names]).alias("_n"),
        )
        .with_columns(pl.when(pl.col("_n") >= need).then(pl.col("score")).alias("score"))
        .select(*KEYS, "score")
    )


@dataclass(frozen=True)
class RidgeModel:
    factors: tuple[str, ...]
    weights: np.ndarray
    alpha: float

    def predict(self, feats: pl.DataFrame) -> pl.DataFrame:
        x = feats.select(list(self.factors)).fill_null(0.0).to_numpy()
        keep = feats.select(
            pl.all_horizontal([pl.col(f).is_not_null() for f in self.factors]).alias("ok")
        )["ok"].to_numpy()
        score = np.where(keep, x @ self.weights, np.nan)
        return feats.select(*KEYS).with_columns(pl.Series("score", score).fill_nan(None))


def ridge_fit(
    feats: pl.DataFrame, labels: pl.DataFrame, factors: Sequence[str], label: str, alpha: float
) -> RidgeModel:
    """Closed-form ridge of the demeaned label on factor ranks: w = (X'X + alpha I)^-1 X'y."""
    j = feats.join(labels.select("tix", "ts", label), on=["tix", "ts"]).drop_nulls(
        [*factors, label]
    )
    if j.height == 0:
        return RidgeModel(tuple(factors), np.zeros(len(factors)), alpha)
    x = j.select(list(factors)).to_numpy()
    y = j[label].to_numpy()
    w = np.linalg.solve(x.T @ x + alpha * np.eye(len(factors)), x.T @ y)
    return RidgeModel(tuple(factors), w, alpha)


def walk_forward_ridge(
    feats: pl.DataFrame,
    labels: pl.DataFrame,
    factors: Sequence[str],
    label: str,
    folds: Sequence[DateRange],
    sessions: Sequence[date],
    alpha: float,
    purge_sessions: int = 1,
    embargo_sessions: int = 5,
) -> pl.DataFrame:
    """Out-of-sample ridge scores for folds 2..k.

    Fold k is scored by a model trained on sessions before it, minus an embargo and a purge so that
    no training label overlaps the validation period. The first fold has no training data and so
    no score.
    """
    pieces: list[pl.DataFrame] = []
    index = {d: i for i, d in enumerate(sessions)}
    for k in range(1, len(folds)):
        first_val = min(d for d in sessions if d >= folds[k].start)
        cutoff = index[first_val] - embargo_sessions - purge_sessions
        if cutoff <= 0:
            continue
        train_end = sessions[cutoff - 1]
        train = feats.filter(pl.col("session_date") <= train_end)
        model = ridge_fit(train, labels, factors, label, alpha)
        val = feats.filter(pl.col("session_date").is_between(folds[k].start, folds[k].end))
        pieces.append(model.predict(val))
    if not pieces:
        return feats.select(*KEYS).with_columns(pl.lit(None, pl.Float64).alias("score")).head(0)
    return pl.concat(pieces)
