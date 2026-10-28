"""What a backtest returns."""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class BacktestSummary:
    net_sharpe: float  # annualised: mean daily net return / std * sqrt(252)
    gross_sharpe: float
    mean_daily_net_bps: float
    cost_share_of_gross: float | None  # costs / gross P&L; None when gross P&L is not positive
    avg_positions: float
    max_drawdown: float  # peak-to-trough of cumulative net P&L, as a fraction of capital
    hit_rate: float  # share of days with positive net P&L
    n_days: int


@dataclass(frozen=True)
class BacktestResult:
    daily: pl.DataFrame  # date, gross_pnl, costs, net_pnl, orders, gross_exposure, turnover
    summary: BacktestSummary
    net_sharpe_by_year: dict[int, float]
    net_sharpe_by_fold: dict[str, float]
    orders: pl.DataFrame | None  # every simulated order, when requested
