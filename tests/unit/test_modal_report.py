import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from tests.helpers.orchestrator import FakeBackend
from tests.helpers.store import DIRECTIVE, add_trial, open_store

from alpha_factory.compute.modal_dev import CHUNK, ModalBackend, chunks, prepare_data_dir
from alpha_factory.compute.specs import (
    EvalBatchRequest,
    EvalSpec,
    PoolLooRequest,
)
from alpha_factory.contracts import LLMCallRecord, ReasonCode
from alpha_factory.research.report import ab_comparison, render_html, wilson

REPO = Path(__file__).parents[2]


def test_the_data_directory_is_built_from_the_volume_layout(tmp_path: Path) -> None:
    mount = tmp_path / "vol"
    (mount / "build" / "dev-abc123").mkdir(parents=True)
    (mount / "build" / "dev-abc123" / "manifest.json").write_text("{}")
    (mount / "calibration").mkdir()
    (mount / "current_dev.txt").write_text("dev-abc123\n")
    work = tmp_path / "work"
    for _ in range(2):  # idempotent
        prepare_data_dir(mount, "dev", work)
    assert (work / "datasets" / "dev" / "manifest.json").read_text() == "{}"
    assert (work / "calibration").resolve() == (mount / "calibration").resolve()
    (mount / "current_dev.txt").write_text("dev-other\n")
    (mount / "build" / "dev-other").mkdir()
    prepare_data_dir(mount, "dev", work)
    assert (work / "datasets" / "dev").resolve() == (mount / "build" / "dev-other").resolve()


def test_chunks() -> None:
    assert (
        chunks(list(range(7)), 3) == [[0, 1, 2], [3, 4, 5], [6]]
        and chunks([], 3) == []
        and CHUNK == 3
    )


class FakeFunction:
    """Stands in for a modal.Function: .map over JSON strings and .remote."""

    def __init__(self, backend: FakeBackend) -> None:
        self.backend, self.inputs = backend, []

    def map(self, inputs, order_outputs=True):  # type: ignore[no-untyped-def]
        self.inputs = list(inputs)
        for i in self.inputs:
            yield self.backend.evaluate(EvalBatchRequest.model_validate_json(i)).model_dump_json()

    def remote(self, payload: str) -> str:
        return self.backend.pool_loo(PoolLooRequest.model_validate_json(payload)).model_dump_json()


def spec(i: int) -> EvalSpec:
    return EvalSpec(
        trial_key=f"k{i}",
        canonical_expr=f"Neg(Pct($close, {i + 2}))",
        condition=None,
        horizon="h1",
        sign=1,
    )


def test_the_modal_backend_chunks_a_batch_and_keeps_order_and_the_batch_wide_n() -> None:
    fake = FakeBackend({"Neg(Pct($close, 5))": ReasonCode.ACCEPTED})
    fn = FakeFunction(fake)
    backend = ModalBackend(evaluate_fn=fn, loo_fn=fn)
    req = EvalBatchRequest(
        sandbox_hash="a" * 64,
        dataset_hash="b" * 64,
        code_commit="c",
        n_evaluated=40,
        specs=tuple(spec(i) for i in range(7)),
    )
    resp = backend.evaluate(req)
    assert [o.trial_key for o in resp.outcomes] == [f"k{i}" for i in range(7)]  # order preserved
    sent = [EvalBatchRequest.model_validate_json(i) for i in fn.inputs]
    assert [len(s.specs) for s in sent] == [3, 3, 1] and {s.batch_size for s in sent} == {7}
    # every chunk is judged under the same N: n_evaluated + the whole batch
    assert {o.n_at_eval for o in resp.outcomes} == {47}
    assert (
        resp.outcomes[3].result is not None
        and resp.outcomes[3].result.reason == ReasonCode.ACCEPTED
    )


def test_the_modal_backend_forwards_leave_one_out() -> None:
    fake = FakeBackend()
    backend = ModalBackend(evaluate_fn=FakeFunction(fake), loo_fn=FakeFunction(fake))
    req = PoolLooRequest(sandbox_hash="a" * 64, dataset_hash="b" * 64, pool=())
    assert backend.pool_loo(req).combined_ic_without == {}


@pytest.mark.parametrize(
    "script,app,functions",
    [
        ("modal_dev.py", "alpha-factory-eval", {"evaluate_batch", "pool_loo"}),
        ("modal_sealed.py", "alpha-factory-final", {"final_eval_remote"}),
    ],
)
def test_the_deploy_scripts_define_the_expected_apps(
    script: str, app: str, functions: set[str]
) -> None:
    spec_ = importlib.util.spec_from_file_location(f"deploy_{script}", REPO / "deploy" / script)
    assert spec_ is not None and spec_.loader is not None
    mod = importlib.util.module_from_spec(spec_)
    spec_.loader.exec_module(mod)
    assert mod.app.name == app and set(mod.app.registered_functions) == functions


def test_the_dev_and_sealed_apps_use_different_volumes_and_names() -> None:
    from alpha_factory.compute import modal_dev as m
    from alpha_factory.compute import modal_sealed as sealed

    assert m.DEV_APP != sealed.FULL_APP and m.DEV_VOLUME != sealed.FULL_VOLUME
    sealed_text = (REPO / "deploy" / "modal_sealed.py").read_text()
    dev_text = (REPO / "deploy" / "modal_dev.py").read_text()
    assert "FULL_VOLUME" in sealed_text and "DEV_VOLUME" not in sealed_text
    assert (
        "DEV_VOLUME" in dev_text
        and "FULL_VOLUME" not in dev_text
        and "run_final_eval" not in dev_text
    )


def test_the_systemd_unit_runs_the_loop_and_holds_no_secrets() -> None:
    unit = (REPO / "deploy" / "alpha-factory-loop.service").read_text()
    assert "af loop run" in unit and "Restart=on-failure" in unit and "KillSignal=SIGTERM" in unit
    for secret in ("API_KEY", "TOKEN", "WEBHOOK", "SECRET"):
        assert secret not in unit.replace("DEV Modal token", "")


# --- reports -------------------------------------------------------------------------------------
def test_wilson_interval() -> None:
    lo, hi = wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert wilson(0, 0) == (0.0, 0.0)
    assert wilson(0, 20)[0] == 0.0 and 0.1 < wilson(0, 20)[1] < 0.2
    assert wilson(20, 20)[1] == 1.0 and wilson(20, 20)[0] > 0.8


def llm_call(cost: float) -> LLMCallRecord:
    return LLMCallRecord(call_id="llm_1", role="proposer", provider="p", model="m", prompt_hash="a" * 64, contract_hash=None, tokens_in=1, tokens_out=1, latency_ms=1, cost_usd=cost, ok=True, error=None, created_at=datetime.now(UTC))  # fmt: skip


def test_ab_comparison_uses_an_equal_budget_and_reports_spend_per_acceptance(
    tmp_path: Path,
) -> None:
    store, goal_id, tbatch = open_store(tmp_path)
    lbatch = store.create_batch(goal_id, "llm", DIRECTIVE)
    for i in range(10):  # the template proposer evaluated 10 trials, 2 accepted
        add_trial(
            store,
            tbatch,
            f"Neg(Pct($open, {i + 2}))",
            ReasonCode.ACCEPTED if i in (1, 5) else ReasonCode.LOW_IC,
            n=i,
        )
    for i in range(6):  # the LLM only 6, 3 accepted
        add_trial(
            store,
            lbatch,
            f"Neg(Pct($high, {i + 2}))",
            ReasonCode.ACCEPTED if i % 2 == 0 else ReasonCode.LOW_IC,
            n=100 + i,
        )
    store.add_llm_call(llm_call(3.0))
    seen: list[int] = []
    rows = {r.proposer: r for r in ab_comparison(store, lambda f: (seen.append(len(f)), 0.02)[1])}
    t, l = rows["template"], rows["llm"]
    assert t.evaluated == l.evaluated == 6  # the smaller budget, applied to both
    assert (t.accepted, l.accepted) == (2, 3)  # the template's first 6 trials hold two acceptances
    assert l.rate == 0.5 and l.ci[0] < 0.5 < l.ci[1]
    assert l.spend_per_acceptance == pytest.approx(1.0) and t.spend_per_acceptance == 0.0
    assert sorted(seen) == [2, 3] and l.best_combined_ic == 0.02


def test_the_html_report_has_every_section_and_counts_sealed_uses(tmp_path: Path) -> None:
    store, _, bid = open_store(tmp_path)
    add_trial(store, bid, "Neg(Pct($close, 2))", ReasonCode.ACCEPTED)
    store.add_to_pool(store.batch_trials(bid)[0].trial_id)
    store.create_belief(
        statement="Reversal works <b>here</b>.",
        family="short_term_reversal",
        horizon="h1",
        next_test=None,
    )
    html = render_html(store)
    for part in (
        "Goal progress",
        "Reasons",
        "Acceptance by family",
        "Pool",
        "Beliefs",
        "LLM spend",
        "Sealed window (used 0 time(s))",
        "<svg",
    ):
        assert part in html
    assert "&lt;b&gt;here&lt;/b&gt;" in html and "<b>here</b>" not in html  # content is escaped
    assert json.dumps(html)  # a plain string
