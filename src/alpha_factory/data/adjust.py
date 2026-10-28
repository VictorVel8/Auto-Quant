"""Causal forward adjustment: each session's factor uses only corporate actions already happened.

split_factor(d) = product of split ratios with execution date <= d (ratio = shares after / before)
div_factor(d)   = product over ex-dates e <= d of C(e-1) / (C(e-1) - cash)
adj_factor      = split_factor * div_factor

Actions on or before the first session of the data are ignored, so levels read as "price in share
units of the dataset's first day". Nothing here looks past the session being adjusted.
"""

from datetime import date

import polars as pl

from alpha_factory.contracts.data import BAR_SCHEMA

FACTOR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ticker_id": pl.String,
    "session_date": pl.Date,
    "split_factor": pl.Float64,
    "div_factor": pl.Float64,
    "adj_factor": pl.Float64,
}


def compute_adj_factors(
    daily: pl.DataFrame, splits: pl.DataFrame, dividends: pl.DataFrame, first_session: date
) -> pl.DataFrame:
    """Factors for every (ticker_id, session_date) row of `daily` (which carries raw_close)."""
    daily = daily.select("ticker_id", "session_date", "raw_close").sort("session_date", "ticker_id")

    split_events = (
        splits.filter(pl.col("execution_date") > first_session)
        .select(
            "ticker_id",
            pl.col("execution_date").alias("date"),
            pl.col("ratio").alias("split_mult"),
            pl.lit(1.0).alias("div_mult"),
        )
        .with_columns(pl.col("date").cast(pl.Date))
    )

    prev_close = daily.select(
        "ticker_id",
        pl.col("session_date").alias("ex_date"),
        pl.col("raw_close").alias("prev_close"),
    )
    div_events = (
        dividends.filter(pl.col("ex_date") > first_session)
        .sort("ex_date")
        .join_asof(
            prev_close.sort("ex_date"),
            on="ex_date",
            by="ticker_id",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .select(
            "ticker_id",
            pl.col("ex_date").alias("date"),
            pl.lit(1.0).alias("split_mult"),
            pl.when(
                pl.col("prev_close").is_not_null()
                & (pl.col("cash_amount") > 0)
                & (pl.col("prev_close") > pl.col("cash_amount"))
            )
            .then(pl.col("prev_close") / (pl.col("prev_close") - pl.col("cash_amount")))
            .otherwise(1.0)
            .alias("div_mult"),
        )
    )

    events = (
        pl.concat([split_events, div_events])
        .group_by("ticker_id", "date")
        .agg(pl.col("split_mult").product(), pl.col("div_mult").product())
        .sort("ticker_id", "date")
        .with_columns(
            pl.col("split_mult").cum_prod().over("ticker_id").alias("split_factor"),
            pl.col("div_mult").cum_prod().over("ticker_id").alias("div_factor"),
        )
        .select("ticker_id", "date", "split_factor", "div_factor")
        .sort("date")
    )

    out = (
        daily.select("ticker_id", "session_date")
        .join_asof(
            events,
            left_on="session_date",
            right_on="date",
            by="ticker_id",
            strategy="backward",
            check_sortedness=False,
        )
        .with_columns(pl.col("split_factor").fill_null(1.0), pl.col("div_factor").fill_null(1.0))
        .with_columns((pl.col("split_factor") * pl.col("div_factor")).alias("adj_factor"))
        .select([pl.col(c).cast(t) for c, t in FACTOR_SCHEMA.items()])
        .sort("ticker_id", "session_date")
    )
    return out


def apply_adjustment(raw: pl.DataFrame, factors: pl.DataFrame) -> pl.DataFrame:
    """Raw bars + factors -> bars in the Plan 1 bar schema.

    Prices (open, high, low, close, vwap) scale by adj_factor, volume divides by split_factor,
    raw_close stays unadjusted.
    """
    adj = pl.col("adj_factor")
    df = raw.join(factors, on=["ticker_id", "session_date"], how="left")
    if df["adj_factor"].null_count():
        raise ValueError("bars without an adjustment factor")
    return (
        df.select(
            "session_date",
            "ts",
            "bar",
            "ticker_id",
            "ticker",
            (pl.col("open") * adj).alias("open"),
            (pl.col("high") * adj).alias("high"),
            (pl.col("low") * adj).alias("low"),
            (pl.col("close") * adj).alias("close"),
            pl.col("close").alias("raw_close"),
            adj.alias("adj_factor"),
            (pl.col("volume") / pl.col("split_factor")).alias("volume"),
            "trades",
            (pl.col("vwap") * adj).alias("vwap"),
            "n_minutes",
        )
        .select([pl.col(c).cast(t) for c, t in BAR_SCHEMA.items()])
        .sort("ticker_id", "bar")
    )
