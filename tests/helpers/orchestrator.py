"""Fakes for orchestrator tests: a scripted proposer and an in-memory evaluation backend."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tests.helpers.store import GOAL, manifest, result, sandbox

from alpha_factory.agents.proposer import ProposalOutcome
from alpha_factory.compute.specs import (
    EvalBatchRequest,
    EvalBatchResponse,
    EvalOutcome,
    PoolLooRequest,
    PoolLooResponse,
)
from alpha_factory.compute.worker import EvalBackend
from alpha_factory.contracts import BatchDirective, RawProposalBatch, ReasonCode, ResearchGoal
from alpha_factory.orchestrator import Orchestrator
from alpha_factory.research.config import LoopConfig
from alpha_factory.research.store import ResearchStore


def item(
    expr: str,
    family: str = "short_term_reversal",
    sign: str = "positive",
    horizon: str = "h1",
    **kw: object,
) -> dict[str, object]:
    return {
        "type": "new", "parents": [], "family": family, "horizon": horizon,
        "hypothesis": "Stocks that fell over the last hour rebound over the next hour.",
        "mechanism": "Liquidity providers demand a premium to absorb order-flow imbalance.",
        "expected_sign": sign, "expression": expr, "condition": None,
        "refutations": ["cost_killed"], "novelty": "A test proposal.", **kw,
    }  # fmt: skip


class ScriptedProposer:
    """Returns a fixed list of items each batch (a list per call, consumed in order)."""

    def __init__(self, *batches: list[dict[str, object]]) -> None:
        self.batches = list(batches)
        self.calls = 0

    def propose(
        self, directive: BatchDirective, goal: ResearchGoal, store: ResearchStore, rng_seed: int
    ) -> ProposalOutcome:
        self.calls += 1
        items = self.batches.pop(0) if self.batches else []
        return ProposalOutcome(raw=RawProposalBatch(rationale="scripted", proposals=items))


class FakeBackend(EvalBackend):
    """Reason by canonical expression; everything else is LOW_IC."""

    def __init__(
        self, reasons: dict[Any, ReasonCode] | None = None, tstat: dict[str, float] | None = None
    ) -> None:
        self.reasons = reasons or {}
        self.tstat = tstat or {}
        self.requests: list[EvalBatchRequest] = []
        self.loo: Callable[[PoolLooRequest], dict[str, float]] = lambda r: {
            f.factor_id: 0.0 for f in r.pool
        }
        self.fail_next = 0
        self.max_corr_id: dict[str, str] = {}
        self.holdout: dict[str, bool] = {}

    def evaluate(self, request: EvalBatchRequest) -> EvalBatchResponse:
        self.requests.append(request)
        n = request.n_evaluated + (request.batch_size or len(request.specs))
        out = []
        for s in request.specs:
            if self.fail_next:
                self.fail_next -= 1
                out.append(
                    EvalOutcome(trial_key=s.trial_key, result=None, n_at_eval=n, error="boom")
                )
                continue
            reason = self.reasons.get(
                (s.canonical_expr, s.sign), self.reasons.get(s.canonical_expr, ReasonCode.LOW_IC)
            )
            res = result(sandbox(), "e" * 64, reason, trial_id=f"trl_{abs(hash((s.trial_key, len(self.requests)))):x}", sign=s.sign, horizon=s.horizon, tstat=self.tstat.get(s.canonical_expr, 5.0 if reason in (ReasonCode.ACCEPTED, ReasonCode.CORRELATED) else 1.0))  # fmt: skip
            res = res.model_copy(
                update={"max_corr_pool_id": self.max_corr_id.get(s.canonical_expr)}
            )
            ok = (
                self.holdout.get(s.canonical_expr, True)
                if reason in (ReasonCode.ACCEPTED, ReasonCode.CORRELATED)
                else None
            )
            out.append(EvalOutcome(trial_key=s.trial_key, result=res, n_at_eval=n, holdout_ok=ok))
        return EvalBatchResponse(outcomes=tuple(out))

    def pool_loo(self, request: PoolLooRequest) -> PoolLooResponse:
        return PoolLooResponse(combined_ic_without=self.loo(request))


def make(
    tmp: Path,
    *batches: list[dict[str, object]],
    backend: FakeBackend | None = None,
    cfg: LoopConfig | None = None,
    goal: ResearchGoal | None = None,
):  # type: ignore[no-untyped-def]
    """(orchestrator, store, backend, proposer, events) with one active goal."""
    store = ResearchStore(tmp / "store.sqlite")
    store.approve_goal(store.add_goal(goal or GOAL)) if False else None
    gid = store.add_goal(goal or GOAL)
    store.approve_goal(gid)
    backend = backend or FakeBackend()
    proposer = ScriptedProposer(*batches)
    events: list[object] = []
    orch = Orchestrator(
        store=store, cfg=cfg or LoopConfig(), backend=backend, sandbox=sandbox(), manifest=manifest(),
        proposers={"template": proposer, "llm": proposer}, notify=events.append,
    )  # fmt: skip
    return orch, store, backend, proposer, events
