"""Evaluation results, the LLM-facing feedback projection, and the gate."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict

from alpha_factory.contracts.base import Horizon, Record, Sha256
from alpha_factory.contracts.proposals import RefutationCheck


class ReasonCode(StrEnum):
    # Precedence order: the first failing check is `reason`, later ones go to `flags`.
    INVALID_SCHEMA = "invalid_schema"
    INVALID_PARSE = "invalid_parse"
    LOOKAHEAD = "lookahead"
    TOO_COMPLEX = "too_complex"
    DUPLICATE = "duplicate"
    LOW_IC = "low_ic"
    WRONG_SIGN = "wrong_sign"
    UNSTABLE = "unstable"
    SUBSUMED = "subsumed"
    CORRELATED = "correlated"
    REGIME_ONLY = "regime_only"
    COST_KILLED = "cost_killed"
    HOLDOUT_FAIL = "holdout_fail"  # gate holdout; the LLM sees no number
    ACCEPTED = "accepted"


class FoldStat(Record):
    fold_id: str
    ic_mean: float
    ic_tstat: float
    n_days: int


class RefutationOutcome(Record):
    check: RefutationCheck
    triggered: bool
    detail: str


class EvalResult(Record):
    trial_id: str
    proposal_id: str | None
    expr_hash: Sha256
    dataset_hash: Sha256
    sandbox_hash: Sha256
    code_commit: str
    horizon: Horizon
    sign_used: Literal[1, -1]
    folds: tuple[FoldStat, ...]
    ic_mean: float
    ic_tstat: float
    ic_worst_fold: float
    icir: float
    ic_by_year: dict[int, float]
    ic_by_horizon: dict[Horizon, float]
    turnover: float
    max_corr_pool: float
    max_corr_pool_id: str | None
    incremental_ic: float
    ic_by_regime: dict[str, float]  # e.g. {"vol_high": 0.02, "vol_low": 0.01}
    condition_ic_in: float | None
    condition_ic_out: float | None
    net_sharpe_val: float | None
    reason: ReasonCode
    flags: tuple[ReasonCode, ...]
    refutations: tuple[RefutationOutcome, ...]
    hypothesis_refuted: bool
    runtime_s: float


class EvalFeedback(BaseModel):
    """The rounded projection the LLM sees. Built by code only."""

    model_config = ConfigDict(frozen=True)
    trial_id: str
    family: str
    horizon: Horizon
    expression: str
    reason: ReasonCode
    ic: float  # rounded to 0.005
    ic_worst_fold: float  # rounded to 0.005
    years_positive: str  # "3/4"
    corr_to_pool: float  # rounded to 0.05
    note: str


class GateSpec(Record):
    min_ic_mean: float | None = None  # None: use the null-calibrated 99th percentile
    min_ic_tstat: float = 3.0  # floor on the daily-IC t-stat
    deflate_by_trials: bool = True  # raises the t-stat bar as trials accumulate
    min_ic_worst_fold: float = 0.0
    min_years_positive_frac: float = 0.75
    max_corr_pool: float = 0.70
    min_incremental_ic: float = 0.003
    min_net_sharpe_val: float = 0.0
    holdout_min_ic_ratio: float = 0.5  # holdout IC at least this share of search IC, same sign
