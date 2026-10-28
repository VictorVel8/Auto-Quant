"""Hand-built bars and panels for unit tests."""

from collections.abc import Mapping, Sequence
from datetime import date, datetime

import polars as pl

from alpha_factory.contracts.data import BAR_SCHEMA
from alpha_factory.contracts.evaluation import GateSpec
from alpha_factory.contracts.sandbox import (
    CostSpec,
    DateRange,
    Fold,
    LabelSpec,
    PortfolioSpec,
    SandboxSpec,
    UniverseSpec,
)
from alpha_factory.sandbox.panel import Panel, bar_end, build_panel

D1, D2, D3 = date(2024, 3, 4), date(2024, 3, 5), date(2024, 3, 6)


def bar_rows(
    ticker: str,
    session: date,
    closes: Sequence[float | None],
    *,
    adj: float = 1.0,
    volume: float = 1000.0,
) -> list[dict[str, object]]:
    """Bars for one ticker-session: open of bar b is the close of b-1 (first open = close[0])."""
    rows: list[dict[str, object]] = []
    prev = closes[0]
    for b, c in enumerate(closes):
        if c is not None and prev is not None:
            rows.append(
                {
                    "session_date": session, "ts": bar_end(session, b), "bar": b,
                    "ticker_id": ticker, "ticker": ticker.upper(),
                    "open": prev, "high": max(prev, c) + 0.01, "low": min(prev, c) - 0.01,
                    "close": c, "raw_close": c / adj, "adj_factor": adj, "volume": volume,
                    "trades": 10, "vwap": (prev + c) / 2, "n_minutes": 30,
                }
            )  # fmt: skip
        if c is not None:
            prev = c
    return rows


def bars_frame(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).select([pl.col(c).cast(t) for c, t in BAR_SCHEMA.items()])


def universe_frame(tickers: list[str], sessions: list[date]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "session_date": [d for d in sessions for _ in tickers],
            "ticker_id": [t for _ in sessions for t in tickers],
            "rank": [i + 1 for _ in sessions for i, _t in enumerate(tickers)],
            "adv_usd_20d": [1e6] * (len(tickers) * len(sessions)),
            "raw_prev_close": [50.0] * (len(tickers) * len(sessions)),
            "tradable": [True] * (len(tickers) * len(sessions)),
        },
        schema_overrides={"rank": pl.Int32},
    )


def make_panel(
    prices: Mapping[str, Mapping[date, Sequence[float | None]]],
    members: list[str] | None = None,
    sector_of: Mapping[str, str] | None = None,
) -> Panel:
    """prices: ticker -> session -> list of bar closes (None = missing bar)."""
    rows = [r for t, sess in prices.items() for d, cl in sess.items() for r in bar_rows(t, d, cl)]
    sessions = sorted({d for sess in prices.values() for d in sess})
    session_bars = {d: max(len(sess[d]) for sess in prices.values() if d in sess) for d in sessions}
    tickers = sorted(prices)
    sectors = pl.DataFrame(
        {"ticker_id": tickers, "sector": [(sector_of or {}).get(t, "s0") for t in tickers]}
    )
    return build_panel(
        bars_frame(rows), universe_frame(members or tickers, sessions), sectors, session_bars
    )


def small_sandbox() -> SandboxSpec:
    return SandboxSpec(
        sandbox_version="test",
        dataset_hash="a" * 64,
        universe=UniverseSpec(top_n=10, tradable_top_n=10),
        labels=(
            LabelSpec(name="h1", hours=1),
            LabelSpec(name="h2", hours=2),
            LabelSpec(name="h4", hours=4),
        ),
        folds=(
            Fold(
                fold_id="F1",
                train=None,
                val=DateRange(start=date(2024, 3, 4), end=date(2024, 3, 8)),
            ),
        ),
        gate_holdout=DateRange(start=date(2024, 3, 11), end=date(2024, 3, 15)),
        sealed_test=DateRange(start=date(2024, 3, 25), end=date(2024, 3, 29)),
        costs=CostSpec(sell_fee_bps=0.3),
        portfolio=PortfolioSpec(),
        gate=GateSpec(),
    )


__all__ = ["D1", "D2", "D3", "bar_rows", "bars_frame", "datetime", "make_panel", "small_sandbox"]
