"""Messages between the orchestrator and the evaluation worker.

They live in compute/ (not the contracts package) because nothing else uses them. The worker is
stateless: it never touches the research store, and everything it needs travels in the request.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict

from alpha_factory.contracts import EvalResult, Horizon, LibFactor, RefutationCheck, Sha256


class _Msg(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvalSpec(_Msg):
    trial_key: str  # proposal_id + horizon + sign; the idempotency key
    canonical_expr: str
    condition: str | None
    horizon: Horizon
    sign: Literal[1, -1]
    parent_exprs: tuple[str, ...] = ()  # for correlated_with_parent
    refutations: tuple[RefutationCheck, ...] = ()


class EvalBatchRequest(_Msg):
    sandbox_hash: Sha256
    dataset_hash: Sha256
    code_commit: str
    n_evaluated: int  # N for the deflation formula, before this batch
    batch_size: int | None = None  # whole batch size when specs are a chunk of it (sets N)
    pool: tuple[LibFactor, ...] = ()  # for pool correlation
    specs: tuple[EvalSpec, ...]


class EvalOutcome(_Msg):
    trial_key: str
    result: EvalResult | None
    n_at_eval: int  # the N used for this result's t bar
    holdout_ok: bool | None = None  # pass/fail only; None when the holdout was not consulted
    error: str | None = None


class EvalBatchResponse(_Msg):
    outcomes: tuple[EvalOutcome, ...]


class PoolLooRequest(_Msg):
    sandbox_hash: Sha256
    dataset_hash: Sha256
    pool: tuple[LibFactor, ...]
    target_horizon: Horizon = "h2"


class PoolLooResponse(_Msg):
    combined_ic_without: dict[str, float]  # factor_id -> combined IC if that factor were removed
