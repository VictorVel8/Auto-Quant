"""The real evaluation worker, and the whole loop with real evaluations, on the synthetic market."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from alpha_factory.agents.template import TemplateProposer
from alpha_factory.compute.specs import EvalBatchRequest, EvalSpec, PoolLooRequest
from alpha_factory.compute.worker import Evaluator, LocalBackend, StaleDeploymentError
from alpha_factory.contracts import (
    GateSpec,
    LibFactor,
    ReasonCode,
    RefutationCheck,
    ResearchGoal,
    SandboxSpec,
)
from alpha_factory.data import MarketData
from alpha_factory.eval.holdout import holdout_ok
from alpha_factory.orchestrator import Orchestrator
from alpha_factory.orchestrator.ops import reevaluate
from alpha_factory.research.config import LoopConfig
from alpha_factory.research.store import ResearchStore
from alpha_factory.settings import Settings
from alpha_factory.synthetic.dataset import synthetic_sandbox, write_datasets
from alpha_factory.synthetic.market import SyntheticMarket

PLANTED = "Neg(Pct($close, 2))"


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[MarketData, SandboxSpec]]:
    market = SyntheticMarket()
    root = tmp_path_factory.mktemp("loop")
    write_datasets(market, root)
    md = MarketData("dev", settings=Settings(data_dir=root))
    sb = synthetic_sandbox(market, md.manifest.dataset_hash)
    # the planted reversal is mostly the 1-bar baseline and loses to costs, so relax those gates
    lenient = sb.model_copy(
        update={"gate": GateSpec(min_incremental_ic=-1.0, min_net_sharpe_val=-100.0)}
    )
    yield md, lenient


def request(
    md: MarketData, sb: SandboxSpec, *specs: EvalSpec, pool: tuple[LibFactor, ...] = (), n: int = 0
) -> EvalBatchRequest:
    return EvalBatchRequest(sandbox_hash=sb.content_hash(), dataset_hash=md.manifest.dataset_hash, code_commit="test", n_evaluated=n, pool=pool, specs=specs)  # fmt: skip


def spec(expr: str, sign: int = 1, **kw: Any) -> EvalSpec:
    return EvalSpec(
        trial_key=f"k|{expr}|{sign}",
        canonical_expr=expr,
        condition=None,
        horizon="h1",
        sign=1 if sign > 0 else -1,
        **kw,
    )  # type: ignore[arg-type]


def test_the_worker_accepts_the_planted_factor_and_reports_only_a_holdout_bool(
    world: tuple[MarketData, SandboxSpec],
) -> None:
    md, sb = world
    ev = Evaluator(md, sb)
    out = ev.evaluate_batch(
        request(md, sb, spec(PLANTED), spec("Div($volume, Mean($volume, 26))"), n=7)
    ).outcomes
    planted, volume = out
    assert planted.result is not None and planted.result.reason == ReasonCode.ACCEPTED
    assert planted.holdout_ok is True and planted.n_at_eval == 9  # n_evaluated + the batch's size
    assert (
        volume.result is not None
        and volume.result.reason == ReasonCode.LOW_IC
        and volume.holdout_ok is None
    )
    assert planted.result.ic_mean == pytest.approx(0.03, abs=0.012)


def test_a_chunk_of_a_larger_batch_is_judged_under_the_whole_batchs_n(
    world: tuple[MarketData, SandboxSpec],
) -> None:
    md, sb = world
    req = request(md, sb, spec(PLANTED), n=40).model_copy(update={"batch_size": 12})
    (o,) = Evaluator(md, sb).evaluate_batch(req).outcomes
    assert o.n_at_eval == 52  # n_evaluated + the batch size, not + the one spec in this chunk


def test_a_wrong_sign_proposal_is_reported_as_such(world: tuple[MarketData, SandboxSpec]) -> None:
    md, sb = world
    (o,) = Evaluator(md, sb).evaluate_batch(request(md, sb, spec("Pct($close, 2)"))).outcomes
    assert (
        o.result is not None
        and o.result.reason == ReasonCode.WRONG_SIGN
        and o.result.hypothesis_refuted
    )


def test_the_holdout_failure_changes_the_reason_and_never_returns_a_number(
    world: tuple[MarketData, SandboxSpec],
) -> None:
    md, sb = world
    strict = sb.model_copy(
        update={"gate": sb.gate.model_copy(update={"holdout_min_ic_ratio": 5.0})}
    )
    (o,) = Evaluator(md, strict).evaluate_batch(request(md, strict, spec(PLANTED))).outcomes
    assert (
        o.result is not None
        and o.result.reason == ReasonCode.HOLDOUT_FAIL
        and o.holdout_ok is False
    )
    assert "holdout" not in o.result.model_dump_json().replace("holdout_fail", "")


def test_a_stale_request_is_refused_loudly(world: tuple[MarketData, SandboxSpec]) -> None:
    md, sb = world
    ev = Evaluator(md, sb)
    bad = request(md, sb, spec(PLANTED)).model_copy(update={"sandbox_hash": "f" * 64})
    with pytest.raises(StaleDeploymentError, match="sandbox"):
        ev.evaluate_batch(bad)
    bad = request(md, sb, spec(PLANTED)).model_copy(update={"dataset_hash": "f" * 64})
    with pytest.raises(StaleDeploymentError, match="dataset"):
        ev.evaluate_batch(bad)


def test_one_bad_spec_does_not_sink_the_batch(world: tuple[MarketData, SandboxSpec]) -> None:
    md, sb = world
    good, bad = (
        Evaluator(md, sb)
        .evaluate_batch(request(md, sb, spec(PLANTED), spec("Nope($close)")))
        .outcomes
    )
    assert good.result is not None and bad.result is None and "Nope" in (bad.error or "")


def test_pool_correlation_and_parent_refutation_use_the_request_pool(
    world: tuple[MarketData, SandboxSpec],
) -> None:
    md, sb = world
    pool = (
        LibFactor(
            factor_id="fac_twin", canonical_expr="Neg(Delta($close, 2))", sign=1, horizon="h1"
        ),
    )
    s = spec(
        PLANTED,
        parent_exprs=("Neg(Delta($close, 2))",),
        refutations=(RefutationCheck.CORRELATED_WITH_PARENT,),
    )
    (o,) = Evaluator(md, sb).evaluate_batch(request(md, sb, s, pool=pool)).outcomes
    assert (
        o.result is not None
        and o.result.max_corr_pool_id == "fac_twin"
        and o.result.max_corr_pool > 0.8
    )
    assert o.result.refutations[0].triggered


def test_leave_one_out_scores_each_factor_by_what_the_rest_can_do(
    world: tuple[MarketData, SandboxSpec],
) -> None:
    md, sb = world
    pool = tuple(LibFactor(factor_id=f"fac_{i}", canonical_expr=e, sign=1, horizon="h1") for i, e in enumerate([PLANTED, "Neg(Pct($close, 3))", "Div($volume, Mean($volume, 26))"]))  # fmt: skip
    out = Evaluator(md, sb).pool_loo(PoolLooRequest(sandbox_hash=sb.content_hash(), dataset_hash=md.manifest.dataset_hash, pool=pool, target_horizon="h1")).combined_ic_without  # fmt: skip
    assert set(out) == {"fac_0", "fac_1", "fac_2"}
    # removing the useless volume factor helps the combination; removing either reversal hurts it
    assert out["fac_2"] > out["fac_0"] and out["fac_2"] > out["fac_1"]


def test_holdout_ok_can_be_called_directly(world: tuple[MarketData, SandboxSpec]) -> None:
    md, sb = world
    ev = Evaluator(md, sb)
    (o,) = ev.evaluate_batch(request(md, sb, spec(PLANTED))).outcomes
    assert o.result is not None
    from alpha_factory.dsl import compile_expr

    assert holdout_ok(o.result, compile_expr(PLANTED), sb, context=ev.holdout)


def test_the_loop_runs_template_batches_with_real_evaluations_and_finds_the_reversal(
    world: tuple[MarketData, SandboxSpec], tmp_path: Path
) -> None:
    md, sb = world
    store = ResearchStore(tmp_path / "store.sqlite")
    goal = ResearchGoal(title="Reversal", question="Which forms of reversal predict h1 returns?", families=["short_term_reversal"], horizons=["h1"], budget_trials=60, stop_when="Never.")  # fmt: skip
    store.approve_goal(store.add_goal(goal))
    cfg = LoopConfig(batch_size=6, bandit_window=50)
    orch = Orchestrator(
        store=store, cfg=cfg, backend=LocalBackend(Evaluator(md, sb)), sandbox=sb, manifest=md.manifest,
        proposers={"template": TemplateProposer(), "llm": TemplateProposer()},
    )  # fmt: skip
    for _ in range(3):
        assert orch.run_once()
    reasons = store.reason_counts()
    assert store.total_trials() >= 18 and store.n_evaluated(sb.content_hash()) >= 12
    # the reversal templates produce real, significant ICs even though they are 1-bar-baseline-like
    assert (
        reasons.get("accepted", 0) + reasons.get("wrong_sign", 0) + reasons.get("correlated", 0)
    ) >= 1
    # every stored evaluation re-computes to identical numbers from its stored inputs
    backend = LocalBackend(Evaluator(md, sb))
    checked = 0
    for t in store.goal_trials(store.active_goal_id() or ""):  # type: ignore[arg-type]
        if t.evaluated and t.proposal_id:
            r = reevaluate(store, backend, sb, md.manifest.dataset_hash, t.trial_id)
            assert r.identical, r.differences
            checked += 1
    assert checked >= 12
