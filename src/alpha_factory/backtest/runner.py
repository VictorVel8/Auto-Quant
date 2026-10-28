"""backtest(): fetch prices, tradability and sectors from MarketData and run the simulation."""

import polars as pl

from alpha_factory.backtest.engine import simulate
from alpha_factory.backtest.result import BacktestResult
from alpha_factory.contracts.sandbox import SandboxSpec
from alpha_factory.data.marketdata import MarketData


def backtest(
    scores: pl.LazyFrame | pl.DataFrame,
    sandbox: SandboxSpec,
    data: MarketData,
    *,
    record_orders: bool = False,
) -> BacktestResult:
    """What would this score have earned on the sandbox's account after dollar costs?

    scores: ticker_id, ts, score (any score frame: one factor, the combined signal, ...).
    """
    frame = scores.collect() if isinstance(scores, pl.LazyFrame) else scores
    frame = frame.drop_nulls("score")
    local = pl.col("ts").dt.convert_time_zone("America/New_York").dt.date()
    lo, hi = frame.select(local.min().alias("lo"), local.max().alias("hi")).row(0)
    bars = data.bars(
        lo,
        hi,
        columns=["session_date", "bar", "ticker_id", "open", "raw_close", "adj_factor"],
        members_only=False,
    ).collect()
    members = data.universe(lo, hi).select("session_date", "ticker_id", "tradable").collect()
    return simulate(
        frame.select("ticker_id", "ts", "score"),
        bars,
        members,
        data.sectors().select("ticker_id", "sector"),
        sandbox,
        record_orders=record_orders,
    )
