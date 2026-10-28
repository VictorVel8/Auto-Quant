# pyright: basic
"""Whole-share intraday simulation at the rebalance times, flat at the close, costs in dollars.

At each rebalance bar the book is long the top N and short the bottom N tradable names by score
(sector-neutral if configured), with a hold buffer that keeps an existing name while its rank is
within N + buffer. Trades fill at the NEXT bar's raw open. Everything closes at the session's last
bar close (or a name's last traded price that session). The same code serves a single factor, the
combined signal and the final evaluation: the input is any (ticker_id, ts, score) frame.
"""

from datetime import date, time
from math import floor, sqrt
from typing import cast

import numpy as np
import polars as pl

from alpha_factory.backtest.costs import order_cost
from alpha_factory.backtest.result import BacktestResult, BacktestSummary
from alpha_factory.contracts.sandbox import CostSpec, SandboxSpec
from alpha_factory.sandbox.panel import make_grid

DAYS_PER_YEAR = 252


def rebalance_bars(times: tuple[time, ...], bar_minutes: int) -> list[int]:
    """Bar index whose END is each rebalance time (09:30 open): 10:00 is bar 0 for 30-min bars."""
    return [(t.hour * 60 + t.minute - 9 * 60 - 30) // bar_minutes - 1 for t in times]


def _sharpe(r: np.ndarray) -> float:
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * sqrt(DAYS_PER_YEAR))


def _max_drawdown(cum: np.ndarray) -> float:
    if len(cum) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.r_[0.0, cum])
    return float((peak[1:] - cum).max())


class _Day:
    """Orders and costs of one session."""

    def __init__(
        self,
        d: date,
        costs: CostSpec,
        record: bool,
        rows: list[tuple[date, int, str, str, int, float, float, float, str]],
    ) -> None:
        self.d, self.costs, self.record, self.rows = d, costs, record, rows
        self.costs_paid = self.traded = 0.0
        self.n_orders = 0

    def book(self, b: int, t: str, shares: int, price: float, is_sell: bool, why: str) -> float:
        c = order_cost(self.costs, shares, price, is_sell=is_sell)
        self.costs_paid += c
        self.traded += shares * price
        self.n_orders += 1
        if self.record:
            side = "sell" if is_sell else "buy"
            self.rows.append((self.d, b, t, side, shares, price, shares * price, c, why))
        return c


def simulate(
    scores: pl.DataFrame,
    bars: pl.DataFrame,
    members: pl.DataFrame,
    sectors: pl.DataFrame,
    sandbox: SandboxSpec,
    *,
    record_orders: bool = False,
) -> BacktestResult:
    """scores: ticker_id, ts, score. bars: session_date, bar, ticker_id, open, raw_close,
    adj_factor. members: session_date, ticker_id, tradable. sectors: ticker_id, sector."""
    pf, costs = sandbox.portfolio, sandbox.costs
    n_side = pf.positions_per_side
    notional_each = pf.capital_usd * pf.gross_leverage / (2 * n_side)
    reb = rebalance_bars(pf.rebalance_times, sandbox.bar_minutes)

    sess_bars = {
        d: int(n) + 1 for d, n in bars.group_by("session_date").agg(pl.col("bar").max()).iter_rows()
    }
    grid = make_grid(sess_bars).select("session_date", "bar", "ts")
    sec_of = dict(sectors.select("ticker_id", "sector").iter_rows())
    sc = (
        scores.join(grid, on="ts")
        .filter(pl.col("bar").is_in(reb))
        .select("session_date", "bar", "ticker_id", "score")
    )
    tradable = members.filter(pl.col("tradable")).select("session_date", "ticker_id")
    raw = bars.with_columns((pl.col("open") / pl.col("adj_factor")).alias("raw_open")).select(
        "session_date", "bar", "ticker_id", "raw_open", "raw_close"
    )

    daily_rows: list[tuple[date, float, float, int, float, float, float]] = []
    order_rows: list[tuple[date, int, str, str, int, float, float, float, str]] = []
    sc_by = sc.partition_by("session_date", as_dict=True)
    raw_by = raw.partition_by("session_date", as_dict=True)
    tr_by = {
        k[0]: set(v["ticker_id"])
        for k, v in tradable.partition_by("session_date", as_dict=True).items()
    }

    for d in sorted(sess_bars):
        px = raw_by.get((d,))
        sdf = sc_by.get((d,))
        if px is None or sdf is None:
            continue
        last_bar = sess_bars[d] - 1
        open_px: dict[tuple[str, int], float] = {}
        close_px: dict[tuple[str, int], float] = {}
        last_close: dict[str, float] = {}
        for t, b, o, c in (
            px.sort("bar").select("ticker_id", "bar", "raw_open", "raw_close").iter_rows()
        ):
            if o is not None:
                open_px[(t, b)] = o
            if c is not None:
                close_px[(t, b)] = c
                last_close[t] = c
        by_bar = {b: g for (b,), g in sdf.partition_by("bar", as_dict=True).items()}
        ok_today = tr_by.get(d, set())

        pos: dict[str, tuple[int, float]] = {}  # ticker -> (signed shares, entry price)
        gross = 0.0
        day = _Day(d, costs, record_orders, order_rows)
        max_expo = 0.0
        pos_counts: list[int] = []
        book = day.book

        for b in reb:
            nxt = b + 1
            if b not in by_bar or nxt > last_bar:
                continue
            g = by_bar[b]
            cand = {
                t: s
                for t, s in zip(g["ticker_id"], g["score"], strict=True)
                if t in ok_today and s is not None and np.isfinite(s) and (t, b) in close_px
            }
            if pf.sector_neutral and cand:
                by_sec: dict[str, list[float]] = {}
                for t, s in cand.items():
                    by_sec.setdefault(sec_of.get(t, "other"), []).append(s)
                means = {k: float(np.mean(v)) for k, v in by_sec.items()}
                cand = {t: s - means[sec_of.get(t, "other")] for t, s in cand.items()}
            order = sorted(cand, key=lambda t: (-cand[t], t))  # best first, ties by id
            rank = {t: i + 1 for i, t in enumerate(order)}
            n_c = len(order)
            keep = {
                t
                for t, (q, _) in pos.items()
                if t in rank
                and (
                    rank[t] <= n_side + pf.hold_buffer
                    if q > 0
                    else n_c - rank[t] + 1 <= n_side + pf.hold_buffer
                )
            }
            closing = [t for t in pos if t not in keep and (t, nxt) in open_px]
            for t in closing:
                q, entry = pos.pop(t)
                px_fill = open_px[(t, nxt)]
                gross += q * (px_fill - entry)
                book(nxt, t, abs(q), px_fill, q > 0, "rebalance_close")
            n_long = sum(1 for q, _ in pos.values() if q > 0)
            n_short = sum(1 for q, _ in pos.values() if q < 0)
            blocked = set(pos) | set(closing)
            for t in order:  # best first -> new longs
                if n_long >= n_side:
                    break
                if t in blocked or (t, nxt) not in open_px:
                    continue
                shares = floor(notional_each / close_px[(t, b)])
                if shares < 1:
                    continue
                pos[t] = (shares, open_px[(t, nxt)])
                blocked.add(t)
                n_long += 1
                book(nxt, t, shares, open_px[(t, nxt)], False, "open_long")
            for t in reversed(order):  # worst first -> new shorts
                if n_short >= n_side:
                    break
                if t in blocked or (t, nxt) not in open_px:
                    continue
                shares = floor(notional_each / close_px[(t, b)])
                if shares < 1:
                    continue
                pos[t] = (-shares, open_px[(t, nxt)])
                blocked.add(t)
                n_short += 1
                book(nxt, t, shares, open_px[(t, nxt)], True, "open_short")
            max_expo = max(max_expo, sum(abs(q) * e for q, e in pos.values()) / pf.capital_usd)
            pos_counts.append(len(pos))

        for t, (q, entry) in list(pos.items()):  # flat by the close
            exit_px = close_px.get((t, last_bar), last_close.get(t, entry))
            gross += q * (exit_px - entry)
            book(last_bar, t, abs(q), exit_px, q > 0, "exit")
        daily_rows.append(
            (
                d,
                gross,
                day.costs_paid,
                day.n_orders,
                max_expo,
                day.traded / pf.capital_usd,
                float(np.mean(pos_counts)) if pos_counts else 0.0,
            )
        )

    daily = pl.DataFrame(
        daily_rows,
        schema=[
            "date",
            "gross_pnl",
            "costs",
            "orders",
            "gross_exposure",
            "turnover",
            "avg_positions",
        ],
        orient="row",
    ).with_columns((pl.col("gross_pnl") - pl.col("costs")).alias("net_pnl"))
    daily = daily.select(
        "date",
        "gross_pnl",
        "costs",
        "net_pnl",
        "orders",
        "gross_exposure",
        "turnover",
        "avg_positions",
    )
    orders = (
        pl.DataFrame(
            order_rows,
            schema=[
                "date",
                "bar",
                "ticker_id",
                "side",
                "shares",
                "price",
                "notional",
                "cost",
                "reason",
            ],
            orient="row",
        )
        if record_orders
        else None
    )
    return BacktestResult(
        daily, _summary(daily, pf.capital_usd), *_breakdowns(daily, sandbox), orders
    )


def _summary(daily: pl.DataFrame, capital: float) -> BacktestSummary:
    if daily.is_empty():
        return BacktestSummary(0.0, 0.0, 0.0, None, 0.0, 0.0, 0.0, 0)
    net = daily["net_pnl"].to_numpy() / capital
    gross_r = daily["gross_pnl"].to_numpy() / capital
    gross_total = float(daily["gross_pnl"].sum())
    return BacktestSummary(
        net_sharpe=_sharpe(net),
        gross_sharpe=_sharpe(gross_r),
        mean_daily_net_bps=float(net.mean() * 1e4),
        cost_share_of_gross=float(daily["costs"].sum()) / gross_total if gross_total > 0 else None,
        avg_positions=float(cast(float, daily["avg_positions"].mean())),
        max_drawdown=_max_drawdown(np.cumsum(net)),
        hit_rate=float(cast(float, (daily["net_pnl"] > 0).mean())),
        n_days=daily.height,
    )


def _breakdowns(
    daily: pl.DataFrame, sandbox: SandboxSpec
) -> tuple[dict[int, float], dict[str, float]]:
    cap = sandbox.portfolio.capital_usd
    years: dict[int, float] = {}
    if not daily.is_empty():
        by_year = daily.group_by(pl.col("date").dt.year().alias("year")).agg(pl.col("net_pnl"))
        for y, pnl in zip(by_year["year"], by_year["net_pnl"], strict=True):
            years[int(y)] = _sharpe(pnl.to_numpy() / cap)
    folds = {
        f.fold_id: _sharpe(
            daily.filter(pl.col("date").is_between(f.val.start, f.val.end))["net_pnl"].to_numpy()
            / cap
        )
        for f in sandbox.folds
    }
    return years, folds
