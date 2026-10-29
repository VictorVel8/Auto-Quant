"""Builders for research-store tests."""

from datetime import UTC, datetime
from pathlib import Path

from tests.helpers.panels import small_sandbox

from alpha_factory.contracts import (
    BatchDirective,
    DataManifest,
    EvalResult,
    FactorProposal,
    FoldStat,
    ProposalRecord,
    ReasonCode,
    RefutationCheck,
    ResearchGoal,
    SandboxSpec,
)
from alpha_factory.hashing import sha256_canonical
from alpha_factory.research.store import ResearchStore

H = "a" * 64
NOW = datetime(2026, 1, 1, tzinfo=UTC)

GOAL = ResearchGoal(
    title="Reversal",
    question="Does short-term reversal survive costs at h1?",
    families=["short_term_reversal"],
    horizons=["h1"],
    budget_trials=100,
    stop_when="Two accepted factors.",
)
DIRECTIVE = BatchDirective(
    n_new=4, n_mutate=0, n_refine=0, n_combine=0,
    explore_families=("short_term_reversal",), avoid_families=(), min_outside_families=1,
)  # fmt: skip


def manifest(dataset_hash: str = H) -> DataManifest:
    return DataManifest(
        dataset_id="d", dataset_hash=dataset_hash, kind="dev", source="synthetic",
        start=NOW.date(), end=NOW.date(), n_sessions=1, n_tickers=1, bar_minutes=30,
        adjustment="forward", params_hash=H, code_commit="c", created_at=NOW, files=(),
    )  # fmt: skip


def sandbox() -> SandboxSpec:
    return small_sandbox()


def proposal(
    batch_id: str,
    goal_id: str,
    expr: str = "Neg(Pct($close, 2))",
    family: str = "short_term_reversal",
    n: int = 0,
) -> ProposalRecord:
    return ProposalRecord(
        proposal_id=f"prp_{batch_id}_{n}", batch_id=batch_id, goal_id=goal_id, llm_call_id=None,
        proposal=FactorProposal(
            type="new", parents=[], family=family, horizon="h1",
            hypothesis="Stocks that fell over the last hour rebound over the next hour.",
            mechanism="Liquidity providers demand a premium to absorb order-flow imbalance.",
            expected_sign="positive", expression=expr, condition=None,
            refutations=[RefutationCheck.COST_KILLED], novelty="Baseline reversal.",
        ),
        canonical_expr=expr, expr_hash=sha256_canonical({"e": expr}), ast_depth=3,
        windows_used=(2,), created_at=NOW,
    )  # fmt: skip


def result(
    sb: SandboxSpec,
    expr_hash: str,
    reason: ReasonCode = ReasonCode.LOW_IC,
    *,
    trial_id: str = "trl_x",
    sign: int = 1,
    horizon: str = "h1",
    tstat: float = 1.0,
    ic: float = 0.01,
) -> EvalResult:
    return EvalResult(
        trial_id=trial_id, proposal_id=None, expr_hash=expr_hash, dataset_hash=H,
        sandbox_hash=sb.content_hash(), code_commit="c", horizon=horizon,  # type: ignore[arg-type]
        sign_used=1 if sign > 0 else -1,
        folds=(FoldStat(fold_id="F1", ic_mean=ic, ic_tstat=tstat, n_days=10),),
        ic_mean=ic, ic_tstat=tstat, ic_worst_fold=ic, icir=0.1, ic_by_year={2024: ic},
        ic_by_horizon={"h1": ic}, turnover=0.5, max_corr_pool=0.0, max_corr_pool_id=None,
        incremental_ic=0.01, ic_by_regime={}, condition_ic_in=None, condition_ic_out=None,
        net_sharpe_val=1.0, reason=reason, flags=(), refutations=(), hypothesis_refuted=False,
        runtime_s=0.1,
    )  # fmt: skip


def open_store(tmp: Path) -> tuple[ResearchStore, str, str]:
    """A store with one active goal and one batch: (store, goal_id, batch_id)."""
    store = ResearchStore(tmp / "store.sqlite")
    goal_id = store.add_goal(GOAL)
    store.approve_goal(goal_id)
    batch_id = store.create_batch(goal_id, "template", DIRECTIVE)
    return store, goal_id, batch_id


def add_trial(
    store: ResearchStore, batch_id: str, expr: str, reason: ReasonCode, family: str = "short_term_reversal",
    *, sign: int = 1, n: int = 0, source: str = "template", **overrides: object,
) -> str:  # fmt: skip
    """Store a proposal and one evaluated trial for it, with chosen result fields."""
    goal_id = store.get_batch(batch_id)["goal_id"]
    prop = proposal(batch_id, goal_id, expr, family, n=n)
    store.add_proposals([prop])
    res = result(
        sandbox(), prop.expr_hash, reason, trial_id=f"trl_{prop.expr_hash[-12:]}", sign=sign
    )
    if overrides:
        res = res.model_copy(update=overrides)
    return store.record_trial(
        source=source, sandbox=sandbox(), manifest=manifest(), reason=reason, proposal=prop,  # type: ignore[arg-type]
        result=res, n_at_eval=20,
    )  # fmt: skip
