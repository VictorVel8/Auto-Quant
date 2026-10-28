"""Backtester: whole-share intraday simulation with dollar costs."""

from alpha_factory.backtest.engine import simulate
from alpha_factory.backtest.result import BacktestResult, BacktestSummary
from alpha_factory.backtest.runner import backtest

__all__ = ["BacktestResult", "BacktestSummary", "backtest", "simulate"]
