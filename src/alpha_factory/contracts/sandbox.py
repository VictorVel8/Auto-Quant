"""The sandbox spec (hashed; every result points back to it) and the sealed final evaluation."""

from datetime import date, datetime, time, timedelta
from itertools import pairwise
from typing import Literal

from pydantic import model_validator

from alpha_factory.contracts.base import Horizon, Record, Sha256
from alpha_factory.contracts.evaluation import GateSpec
from alpha_factory.hashing import sha256_canonical


def _weekdays_between(a: date, b: date) -> int:
    """Weekdays strictly between two dates (holidays are not known to a contract)."""
    return sum(1 for i in range(1, (b - a).days) if (a + timedelta(days=i)).weekday() < 5)


class DateRange(Record):
    start: date
    end: date

    @model_validator(mode="after")
    def _ordered(self) -> "DateRange":
        if self.start >= self.end:
            raise ValueError("start must be before end")
        return self


class UniverseSpec(Record):
    top_n: int = 1000
    adv_window_days: int = 20
    min_price: float = 5.0
    tradable_top_n: int | None = 500


class LabelSpec(Record):
    name: Horizon
    hours: int
    entry: Literal["next_bar_open"] = "next_bar_open"
    exit: Literal["k_hours_or_close"] = "k_hours_or_close"
    demean: bool = True


class Fold(Record):
    fold_id: str
    train: DateRange | None  # None for pure factor IC; set for the combiner
    val: DateRange


class CostSpec(Record):
    commission_per_share: float = 0.0035
    min_commission_per_order: float = 0.35
    max_commission_pct: float = 0.01
    half_spread_bps: float = 2.0
    sell_fee_bps: float


class PortfolioSpec(Record):
    capital_usd: float = 50_000
    gross_leverage: float = 1.0
    positions_per_side: int = 10
    rebalance_times: tuple[time, ...] = (time(10, 0), time(12, 0), time(14, 0))
    hold_buffer: int = 5  # keep a name until its rank falls past N + buffer
    sector_neutral: bool = True
    exit_at_close: bool = True


class SandboxSpec(Record):
    sandbox_version: str
    dataset_hash: Sha256
    bar_minutes: Literal[30] = 30
    universe: UniverseSpec
    labels: tuple[LabelSpec, ...]
    folds: tuple[Fold, ...]
    purge_bars: int = 8
    embargo_days: int = 5
    gate_holdout: DateRange
    sealed_test: DateRange
    costs: CostSpec
    portfolio: PortfolioSpec
    gate: GateSpec

    @model_validator(mode="after")
    def _no_overlap_with_test(self) -> "SandboxSpec":
        for f in self.folds:
            last = f.val.end if f.train is None else max(f.val.end, f.train.end)
            if last >= self.gate_holdout.start:
                raise ValueError(f"{f.fold_id} overlaps the gate holdout")
        ordered = sorted(self.folds, key=lambda f: f.val.start)
        for a, b in pairwise(ordered):
            if a.val.end >= b.val.start:
                raise ValueError(f"{a.fold_id} and {b.fold_id} overlap")
        if self.gate_holdout.end >= self.sealed_test.start:
            raise ValueError("gate holdout must end before the sealed test")
        gap = _weekdays_between(self.gate_holdout.end, self.sealed_test.start)
        if gap < self.embargo_days:
            raise ValueError(
                f"embargo: {gap} weekdays between holdout and sealed test, need {self.embargo_days}"
            )
        longest = max(label.hours for label in self.labels) * 60 // self.bar_minutes
        if self.purge_bars < longest:
            raise ValueError("purge_bars must cover the longest label")
        return self

    def content_hash(self) -> str:
        return sha256_canonical(self.model_dump(mode="json"))


class FinalEvalRequest(Record):
    lib_version: str
    combiner_hash: Sha256
    sandbox_hash: Sha256

    def request_hash(self) -> str:
        return sha256_canonical(self.model_dump(mode="json"))


class FinalEvalReport(Record):
    request_hash: Sha256
    ic_mean: float
    ic_tstat: float
    net_sharpe: float
    net_sharpe_by_year: dict[int, float]
    turnover: float
    cost_share_of_gross: float
    created_at: datetime


class CombinerSpec(Record):
    kind: Literal["rank_average", "ridge"]
    target_horizon: Horizon = "h2"
    ridge_alpha: float = 10.0


class LibFactor(Record):
    factor_id: str
    canonical_expr: str
    sign: Literal[1, -1]
    horizon: Horizon


class FrozenLibrary(Record):
    """A frozen, hashed snapshot of the pool and its combiner. Only libraries reach final_eval."""

    lib_version: str  # "lib_v3"
    sandbox_hash: Sha256
    factors: tuple[LibFactor, ...]
    combiner: CombinerSpec
    frozen_at: datetime

    def content_hash(self) -> str:
        return sha256_canonical(self.model_dump(mode="json", exclude={"frozen_at"}))
