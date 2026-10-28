"""Six fixed baseline factors. A factor whose raw IC passes but whose incremental IC does not is
mostly one of these in disguise. Versioned with the sandbox: change one and the sandbox version
must change too."""

import polars as pl

from alpha_factory.dsl import compile_expr

BASELINES_VERSION = "1"
BASELINES: dict[str, str] = {
    "reversal_1bar": "-$ret",
    "reversal_1day": "-Pct($close, 13)",
    "momentum_10day": "Pct($close, 130)",
    "volume_shock": "$volume / Mean($volume, 130)",
    "overnight_gap": "$dayopen / $prevclose - 1",
    "intraday_move": "$close / $dayopen - 1",
}


def baseline_scores(panel: pl.DataFrame) -> pl.DataFrame:
    """tix, ts and one column per baseline (null outside the universe)."""
    out: pl.DataFrame | None = None
    for name, expr in BASELINES.items():
        col = compile_expr(expr).run(panel).select("tix", "ts", pl.col("score").alias(name))
        out = col if out is None else out.join(col, on=["tix", "ts"])
    assert out is not None
    return out
