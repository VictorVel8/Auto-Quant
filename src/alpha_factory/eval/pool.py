"""The pool of accepted factors, and correlation with it."""

from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from alpha_factory.dsl import CompiledFactor
from alpha_factory.eval.stats import intraday_ics


@dataclass(frozen=True)
class PoolEntry:
    factor_id: str
    factor: CompiledFactor
    sign: Literal[1, -1] = 1


@dataclass
class Pool:
    """Accepted factors. Their scores are computed on demand and cached per panel."""

    entries: tuple[PoolEntry, ...] = ()
    _cache: dict[tuple[str, int], pl.DataFrame] = field(
        default_factory=dict[tuple[str, int], pl.DataFrame], repr=False
    )

    def ids(self) -> set[str]:
        return {e.factor_id for e in self.entries}

    def scores(self, entry: PoolEntry, panel: pl.DataFrame) -> pl.DataFrame:
        key = (entry.factor_id, id(panel))
        if key not in self._cache:
            out = entry.factor.run(panel).select("tix", "ts", "session_date", "tsi", "score")
            self._cache[key] = out.with_columns(pl.col("score") * entry.sign)
        return self._cache[key]


def mean_rank_corr(a: pl.DataFrame, b: pl.DataFrame, min_names: int) -> float | None:
    """Mean over timestamps of the cross-sectional Spearman correlation of two score frames."""
    ics = intraday_ics(
        a.select("tix", "ts", "session_date", "score"),
        b.select("tix", "ts", pl.col("score").alias("y")),
        "y",
        min_names,
    )
    return float(ics["ic"].mean()) if ics.height else None  # type: ignore[arg-type]


def max_corr_with_pool(
    scores: pl.DataFrame, pool: Pool, panel: pl.DataFrame, min_names: int, every: int = 4
) -> tuple[float, str | None]:
    """Largest absolute mean Spearman correlation with any pool factor, on every 4th timestamp."""
    sub = scores.filter(pl.col("tsi") % every == 0)
    best, best_id = 0.0, None
    for entry in pool.entries:
        other = pool.scores(entry, panel).filter(pl.col("tsi") % every == 0)
        c = mean_rank_corr(sub, other, min_names)
        if c is not None and abs(c) > best:
            best, best_id = abs(c), entry.factor_id
    return best, best_id
