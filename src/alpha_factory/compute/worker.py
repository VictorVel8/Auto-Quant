"""The evaluation worker: evaluate_batch and pool_loo. Runs in-process or on Modal.

It refuses a request whose sandbox or dataset hash does not match the data it holds, so a stale
deployment fails loudly instead of producing numbers under the wrong hash.
"""

from collections.abc import Mapping
from typing import Literal

import polars as pl

from alpha_factory.compute.specs import (
    EvalBatchRequest,
    EvalBatchResponse,
    EvalOutcome,
    EvalSpec,
    PoolLooRequest,
    PoolLooResponse,
)
from alpha_factory.contracts import LibFactor, ReasonCode, SandboxSpec
from alpha_factory.data.marketdata import MarketData
from alpha_factory.dsl import compile_expr
from alpha_factory.eval.combine import rank_average
from alpha_factory.eval.context import EvalContext, build_context
from alpha_factory.eval.evaluate import TrialInfo, evaluate, search_daily_ic
from alpha_factory.eval.holdout import holdout_ok
from alpha_factory.eval.pool import Pool, PoolEntry


class StaleDeploymentError(RuntimeError):
    """The request was written for different data or a different sandbox than this worker has."""


def build_pool(factors: tuple[LibFactor, ...]) -> Pool:
    return Pool(
        tuple(PoolEntry(f.factor_id, compile_expr(f.canonical_expr), f.sign) for f in factors)
    )


class Evaluator:
    def __init__(
        self, data: MarketData, sandbox: SandboxSpec, null_floor: Mapping[str, float] | None = None
    ) -> None:
        self.data, self.sandbox, self.null_floor = data, sandbox, null_floor
        self._search: EvalContext | None = None
        self._holdout: EvalContext | None = None

    @property
    def search(self) -> EvalContext:
        if self._search is None:
            self._search = build_context(self.data, self.sandbox, zone="search")
        return self._search

    @property
    def holdout(self) -> EvalContext:
        if self._holdout is None:
            self._holdout = build_context(self.data, self.sandbox, zone="holdout")
        return self._holdout

    def check(self, sandbox_hash: str, dataset_hash: str) -> None:
        if sandbox_hash != self.sandbox.content_hash():
            raise StaleDeploymentError("sandbox hash does not match this worker's sandbox")
        if dataset_hash != self.data.manifest.dataset_hash:
            raise StaleDeploymentError("dataset hash does not match this worker's data")

    def evaluate_batch(self, request: EvalBatchRequest) -> EvalBatchResponse:
        self.check(request.sandbox_hash, request.dataset_hash)
        pool = build_pool(request.pool)
        by_expr = {e.factor.expression: e.factor_id for e in pool.entries}
        n_trials = request.n_evaluated + (request.batch_size or len(request.specs))
        out: list[EvalOutcome] = []
        for spec in request.specs:
            out.append(self._one(spec, pool, by_expr, n_trials, request.code_commit))
        return EvalBatchResponse(outcomes=tuple(out))

    def _one(
        self, spec: EvalSpec, pool: Pool, by_expr: dict[str, str], n_trials: int, commit: str
    ) -> EvalOutcome:  # fmt: skip
        try:
            factor = compile_expr(spec.canonical_expr, condition=spec.condition)
            info = TrialInfo(
                n_trials=n_trials,
                null_floor=self.null_floor,
                refutations=spec.refutations,
                parent_ids=tuple(by_expr[e] for e in spec.parent_exprs if e in by_expr),
                code_commit=commit,
            )
            result = evaluate(
                factor, spec.horizon, sign=spec.sign, pool=pool, sandbox=self.sandbox,
                context=self.search, trial=info,
            )  # fmt: skip
            ok: bool | None = None
            # The holdout also confirms a replacement candidate (CORRELATED and nothing else).
            if result.reason == ReasonCode.ACCEPTED or (
                result.reason == ReasonCode.CORRELATED and not result.flags
            ):
                ok = holdout_ok(result, factor, self.sandbox, context=self.holdout)
                if not ok and result.reason == ReasonCode.ACCEPTED:
                    result = result.model_copy(update={"reason": ReasonCode.HOLDOUT_FAIL})
            return EvalOutcome(
                trial_key=spec.trial_key, result=result, n_at_eval=n_trials, holdout_ok=ok
            )
        except Exception as e:  # one bad spec must not sink the batch
            return EvalOutcome(
                trial_key=spec.trial_key, result=None, n_at_eval=n_trials, error=repr(e)
            )

    def pool_loo(self, request: PoolLooRequest) -> PoolLooResponse:
        """Combined search-fold IC (rank average) with each pool factor left out in turn."""
        self.check(request.sandbox_hash, request.dataset_hash)
        ctx, pool = self.search, build_pool(request.pool)
        frames = {e.factor_id: pool.scores(e, ctx.panel.df) for e in pool.entries}
        signs = {e.factor_id: e.sign for e in pool.entries}
        out: dict[str, float] = {}
        for fid in frames:
            rest = {k: v for k, v in frames.items() if k != fid}
            if not rest:
                out[fid] = 0.0
                continue
            combined = rank_average(
                rest, {k: 1 for k in rest} | {k: signs[k] for k in rest}
            ).filter(pl.col("session_date").is_between(ctx.start, ctx.end))
            daily = search_daily_ic(combined, ctx, request.target_horizon, 1)
            out[fid] = float(daily["ic"].mean()) if daily.height else 0.0  # type: ignore[arg-type]
        return PoolLooResponse(combined_ic_without=out)


class EvalBackend:
    """Where evaluations run. The local backend runs them in-process."""

    def evaluate(self, request: EvalBatchRequest) -> EvalBatchResponse:
        raise NotImplementedError

    def pool_loo(self, request: PoolLooRequest) -> PoolLooResponse:
        raise NotImplementedError


class LocalBackend(EvalBackend):
    def __init__(self, evaluator: Evaluator) -> None:
        self.evaluator = evaluator

    def evaluate(self, request: EvalBatchRequest) -> EvalBatchResponse:
        return self.evaluator.evaluate_batch(request)

    def pool_loo(self, request: PoolLooRequest) -> PoolLooResponse:
        return self.evaluator.pool_loo(request)


Sign = Literal[1, -1]
