"""Typed view of configs/data.yaml, the only config Plan 2 reads."""

from datetime import date
from typing import Literal

from pydantic import model_validator

from alpha_factory.contracts.base import Record


class UniverseConfig(Record):
    top_n: int = 1000
    tradable_top_n: int = 500
    adv_window_days: int = 20
    min_price: float = 5.0
    min_sessions_present: int = 18


class BarsConfig(Record):
    minutes: Literal[30] = 30
    thin_bar_minutes: int = 5


class DownloadConfig(Record):
    concurrency: int = 8
    retries: int = 5


class KnownCorporateAction(Record):
    """An event the adjustment cannot see (a spin-off next to a split), so QA skips its gap."""

    ticker_id: str
    date: date
    reason: str


class QAConfig(Record):
    max_missing_bar_rate: float = 0.01
    max_thin_bar_share: float = 0.05
    max_extreme_return_share: float = 0.0001
    extreme_return_abs: float = 0.30
    max_split_gap_abs: float = 0.25
    max_unmapped_dollar_share: float = 0.005
    known_corporate_actions: tuple[KnownCorporateAction, ...] = ()


class DataConfig(Record):
    start: date
    end: date
    dev_end: date  # last session before the sealed window plus embargo
    types: tuple[str, ...] = ("CS",)
    universe: UniverseConfig = UniverseConfig()
    bars: BarsConfig = BarsConfig()
    download: DownloadConfig = DownloadConfig()
    qa: QAConfig = QAConfig()

    @model_validator(mode="after")
    def _ordered(self) -> "DataConfig":
        if not self.start < self.dev_end < self.end:
            raise ValueError("need start < dev_end < end")
        if self.universe.tradable_top_n > self.universe.top_n:
            raise ValueError("tradable_top_n cannot exceed top_n")
        return self
