"""Backtester golden test: every number below was worked out by hand (see the comments)."""

from datetime import date, time

import numpy as np
import polars as pl
import pytest
from tests.helpers.panels import D1, D2, small_sandbox

from alpha_factory.backtest import simulate
from alpha_factory.backtest.costs import order_cost
from alpha_factory.contracts.sandbox import CostSpec, PortfolioSpec
from alpha_factory.sandbox.panel import bar_end

COSTS = CostSpec(
    commission_per_share=0.0035, min_commission_per_order=0.35, max_commission_pct=0.01,
    half_spread_bps=2.0, sell_fee_bps=0.5,
)  # fmt: skip


def sandbox(**portfolio: object):  # type: ignore[no-untyped-def]
    base = dict(
        capital_usd=10_000, gross_leverage=1.0, positions_per_side=1,
        rebalance_times=(time(10, 0),), hold_buffer=0, sector_neutral=False,
    )  # fmt: skip
    return small_sandbox().model_copy(
        update={"costs": COSTS, "portfolio": PortfolioSpec(**{**base, **portfolio})}
    )


def bar_row(d: date, b: int, t: str, o: float, c: float) -> dict[str, object]:
    return {
        "session_date": d,
        "bar": b,
        "ticker_id": t,
        "open": o,
        "raw_close": c,
        "adj_factor": 1.0,
    }


def day_bars(
    d: date, prices: dict[str, tuple[float, float, float, float]]
) -> list[dict[str, object]]:
    """prices: ticker -> (close bar 0, open bar 1, close at the last bar 12, open bar 0)."""
    rows = []
    for t, (c0, o1, c12, o0) in prices.items():
        rows += [bar_row(d, 0, t, o0, c0), bar_row(d, 1, t, o1, o1), bar_row(d, 12, t, c12, c12)]
    return rows


def frames(
    per_day: dict[date, tuple[dict[str, float], dict[str, tuple[float, float, float, float]]]],
):  # type: ignore[no-untyped-def]
    bars, scores, members = [], [], []
    for d, (sc, px) in per_day.items():
        bars += day_bars(d, px)
        scores += [{"ticker_id": t, "ts": bar_end(d, 0), "score": s} for t, s in sc.items()]
        members += [{"session_date": d, "ticker_id": t, "tradable": True} for t in sc]
    tickers = sorted({r["ticker_id"] for r in bars})  # type: ignore[type-var]
    sectors = pl.DataFrame({"ticker_id": tickers, "sector": ["s"] * len(tickers)})
    return pl.DataFrame(scores), pl.DataFrame(bars), pl.DataFrame(members), sectors


GOLDEN = {
    D1: ({"A": 3.0, "B": 2.0, "C": 1.0},
         {"A": (100.0, 100.5, 102.0, 100.0), "B": (25.0,) * 4, "C": (50.0, 49.8, 48.0, 50.0)}),
    D2: ({"A": 1.0, "B": 2.0, "C": 3.0},
         {"A": (102.0, 101.5, 101.0, 102.0), "B": (25.0,) * 4, "C": (48.0, 48.2, 49.0, 48.0)}),
}  # fmt: skip


def test_golden_two_days_three_tickers_to_the_cent() -> None:
    result = simulate(*frames(GOLDEN), sandbox(), record_orders=True)
    d1, d2 = result.daily.sort("date").rows(named=True)

    # Day 1: long A, short C; N = 1 so each leg is 10,000 * 1.0 / 2 = 5,000.
    #   A: floor(5000 / 100.00) = 50 shares, in at open 100.50, out at close 102.00 -> +75.00
    #   C: floor(5000 / 50.00) = 100 shares, short at 49.80, covered at 48.00 -> +180.00
    #   A in : 0.35 + 5025.00 * 2bp = 1.005                              -> 1.3550
    #   C in : 0.35 + 4980.00 * 2bp + 4980.00 * 0.5bp (sell) = 0.996 + 0.249 -> 1.5950
    #   A out: 0.35 + 5100.00 * 2bp + 5100.00 * 0.5bp (sell) = 1.020 + 0.255 -> 1.6250
    #   C out: 0.35 + 4800.00 * 2bp = 0.960                              -> 1.3100
    assert d1["gross_pnl"] == pytest.approx(255.00, abs=1e-9)
    assert d1["costs"] == pytest.approx(1.355 + 1.595 + 1.625 + 1.31, abs=1e-9)  # 5.885
    assert d1["net_pnl"] == pytest.approx(249.115, abs=1e-9)
    assert d1["orders"] == 4
    assert d1["turnover"] == pytest.approx((5025 + 4980 + 5100 + 4800) / 10_000)

    # Day 2: long C, short A.
    #   C: floor(5000 / 48.00) = 104 shares, in 48.20, out 49.00 -> +83.20
    #   A: floor(5000 / 102.00) = 49 shares, short 101.50, covered 101.00 -> +24.50
    #   C in : commission 104 * 0.0035 = 0.364 (above the minimum) + 5012.80 * 2bp = 1.00256
    #          -> 1.36656
    #   A in : 0.35 + 4973.50 * 2bp + 4973.50 * 0.5bp = 0.9947 + 0.248675 -> 1.593375
    #   C out: 0.364 + 5096.00 * 2bp + 5096.00 * 0.5bp = 1.0192 + 0.2548 -> 1.638
    #   A out: 0.35 + 4949.00 * 2bp = 0.9898                              -> 1.3398
    assert d2["gross_pnl"] == pytest.approx(107.70, abs=1e-9)
    assert d2["costs"] == pytest.approx(1.36656 + 1.593375 + 1.638 + 1.3398, abs=1e-9)  # 5.937735
    assert d2["net_pnl"] == pytest.approx(101.762265, abs=1e-9)

    s = result.summary
    net = np.array([249.115, 101.762265]) / 10_000
    assert s.n_days == 2 and s.hit_rate == 1.0
    assert s.net_sharpe == pytest.approx(net.mean() / net.std(ddof=1) * np.sqrt(252))
    assert s.mean_daily_net_bps == pytest.approx(net.mean() * 1e4)
    assert s.cost_share_of_gross == pytest.approx((5.885 + 5.937735) / (255.00 + 107.70))
    assert s.max_drawdown == 0.0 and s.avg_positions == 2.0
    assert set(result.net_sharpe_by_year) == {2024}

    orders = result.orders
    assert orders is not None and orders.height == 8
    first = orders.filter(pl.col("date") == D1).sort("bar", "ticker_id")
    assert first.select("ticker_id", "side", "shares", "price", "reason").rows() == [
        ("A", "buy", 50, 100.5, "open_long"),
        ("C", "sell", 100, 49.8, "open_short"),
        ("A", "sell", 50, 102.0, "exit"),
        ("C", "buy", 100, 48.0, "exit"),
    ]


def test_order_cost_formula_pieces() -> None:
    # minimum commission binds on small orders, the 1% cap binds on tiny notionals
    assert order_cost(COSTS, 10, 50.0, is_sell=False) == pytest.approx(0.35 + 500 * 2e-4)
    assert order_cost(COSTS, 1, 20.0, is_sell=False) == pytest.approx(0.20 + 20 * 2e-4)  # 1% of 20
    assert order_cost(COSTS, 1000, 10.0, is_sell=True) == pytest.approx(
        3.5 + 1e4 * 2e-4 + 1e4 * 0.5e-4
    )


def test_hold_buffer_keeps_names_and_saves_orders() -> None:
    rb = (time(10, 0), time(12, 0))

    # at 12:00 the long A slips to second place: with buffer 1 it stays, without it is replaced
    rows, scores, members = [], [], []
    for t in "ABC":
        for b in (0, 1, 4, 5, 12):
            rows.append(bar_row(D1, b, t, 50.0, 50.0))
    for b, sc in ((0, {"A": 3.0, "B": 2.0, "C": 1.0}), (4, {"A": 2.0, "B": 3.0, "C": 1.0})):
        scores += [{"ticker_id": t, "ts": bar_end(D1, b), "score": s} for t, s in sc.items()]
    members = [{"session_date": D1, "ticker_id": t, "tradable": True} for t in "ABC"]
    sectors = pl.DataFrame({"ticker_id": list("ABC"), "sector": ["s"] * 3})
    args = (pl.DataFrame(scores), pl.DataFrame(rows), pl.DataFrame(members), sectors)
    no_buffer = simulate(*args, sandbox(rebalance_times=rb, hold_buffer=0))
    buffered = simulate(*args, sandbox(rebalance_times=rb, hold_buffer=1))
    # without the buffer, A is sold and B bought at 12:00, which adds two orders
    assert no_buffer.daily["orders"][0] > buffered.daily["orders"][0]
    assert buffered.daily["orders"][0] == 4  # open A and C, close both at the exit


def test_half_day_skips_rebalances_without_a_next_bar_and_a_missing_exit_uses_the_last_price() -> (
    None
):
    sb = sandbox(rebalance_times=(time(10, 0), time(14, 0)))
    rows = []
    for t, c12 in (("A", None), ("B", 30.0)):
        rows += [bar_row(D1, 0, t, 50.0, 50.0), bar_row(D1, 1, t, 50.0, 50.0)]
        if c12 is not None:
            rows.append(bar_row(D1, 12, t, c12, c12))
        else:
            rows.append(bar_row(D1, 6, t, 52.0, 52.0))  # A's last traded price of the session
    rows += [
        bar_row(D1, 12, "C", 40.0, 40.0),
        bar_row(D1, 0, "C", 40.0, 40.0),
        bar_row(D1, 1, "C", 40.0, 40.0),
    ]
    scores = [
        {"ticker_id": t, "ts": bar_end(D1, 0), "score": s}
        for t, s in (("A", 3.0), ("B", 2.0), ("C", 1.0))
    ]
    members = [{"session_date": D1, "ticker_id": t, "tradable": True} for t in "ABC"]
    sectors = pl.DataFrame({"ticker_id": list("ABC"), "sector": ["s"] * 3})
    res = simulate(
        pl.DataFrame(scores),
        pl.DataFrame(rows),
        pl.DataFrame(members),
        sectors,
        sb,
        record_orders=True,
    )
    assert res.orders is not None
    exit_a = res.orders.filter((pl.col("ticker_id") == "A") & (pl.col("reason") == "exit"))
    assert exit_a["price"][0] == 52.0  # no bar at the close: the last traded price that session


def test_non_tradable_names_are_never_traded() -> None:
    scores, bars, members, sectors = frames(GOLDEN)
    members = members.with_columns(
        pl.when(pl.col("ticker_id") == "A").then(False).otherwise(True).alias("tradable")
    )
    res = simulate(scores, bars, members, sectors, sandbox(), record_orders=True)
    assert res.orders is not None and "A" not in set(res.orders["ticker_id"])
