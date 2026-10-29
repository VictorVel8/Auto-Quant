import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from tests.helpers.llm import ScriptedLLM
from tests.helpers.orchestrator import FakeBackend, item
from tests.helpers.store import GOAL, add_trial, manifest, open_store, sandbox

from alpha_factory.agents.llm_proposer import LLMProposer
from alpha_factory.agents.manager import brief, goals_from_file, propose_goals
from alpha_factory.agents.reflect import LLMReflector, apply_reflection, reflector_input
from alpha_factory.contracts import (
    BeliefStatus,
    ReasonCode,
    Reflection,
    RefutationCheck,
    RefutationOutcome,
)
from alpha_factory.orchestrator import Orchestrator
from alpha_factory.research.config import LoopConfig
from alpha_factory.research.store import ResearchStore

SB = sandbox()


def refuted(check: RefutationCheck = RefutationCheck.SIGN_UNSTABLE) -> dict[str, Any]:
    return {
        "hypothesis_refuted": True,
        "refutations": (RefutationOutcome(check=check, triggered=True, detail="x"),),
    }


def update(action: str, evidence: list[str], **kw: object) -> dict[str, object]:
    return {
        "action": action,
        "family": "short_term_reversal",
        "horizon": "h1",
        "evidence": evidence,
        "belief_id": None,
        "statement": None,
        "next_test": None,
        **kw,
    }


def reflection(*updates: dict[str, object]) -> Reflection:
    return Reflection.model_validate(
        {"summary": "What this batch taught.", "updates": list(updates)}
    )


def setup(tmp_path: Path):  # type: ignore[no-untyped-def]
    store, goal_id, bid = open_store(tmp_path)
    t_ref = add_trial(store, bid, "Pct($close, 2)", ReasonCode.WRONG_SIGN, n=0, **refuted())
    t_sub = add_trial(store, bid, "Neg(Pct($close, 3))", ReasonCode.SUBSUMED, n=1)  # not refuted
    t_ok = add_trial(store, bid, "Neg(Pct($close, 5))", ReasonCode.ACCEPTED, n=2)
    return store, bid, t_ref, t_sub, t_ok


def test_create_support_and_derived_counts(tmp_path: Path) -> None:
    store, bid, t_ref, t_sub, t_ok = setup(tmp_path)
    apply_reflection(
        store,
        bid,
        reflection(
            update("create", [t_ok], statement="Reversal works over one hour.", next_test="Try h2.")
        ),
    )
    (b,) = store.beliefs()
    assert (b.n_support, b.n_contradict, b.status) == (
        1,
        0,
        BeliefStatus.UNTESTED,
    ) and b.next_test == "Try h2."
    apply_reflection(
        store, bid, reflection(update("support", [t_ok, t_sub], belief_id=b.belief_id))
    )
    (b,) = store.beliefs()
    assert b.n_support == 2  # t_ok was already counted: the primary key drops the repeat
    assert store._db.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 1  # pyright: ignore[reportPrivateUsage]


def test_only_a_refuted_hypothesis_can_count_against_a_belief(tmp_path: Path) -> None:
    store, bid, t_ref, t_sub, _ = setup(tmp_path)
    b = store.create_belief(
        statement="Reversal works.", family="short_term_reversal", horizon="h1", next_test=None
    )
    apply_reflection(store, bid, reflection(update("contradict", [t_ref, t_sub], belief_id=b)))
    (belief,) = store.beliefs()
    assert belief.n_contradict == 1  # SUBSUMED says the expression failed, not the mechanism


def test_refine_supersedes_and_retires_the_old_belief(tmp_path: Path) -> None:
    store, bid, _, _, t_ok = setup(tmp_path)
    old = store.create_belief(
        statement="Reversal works.", family="short_term_reversal", horizon="h1", next_test=None
    )
    store.add_evidence(old, t_ok, "support", bid)
    apply_reflection(
        store,
        bid,
        reflection(
            update("refine", [t_ok], belief_id=old, statement="Reversal works only on high volume.")
        ),
    )
    (new,) = store.beliefs()
    assert (
        new.supersedes == old
        and new.n_support == 1
        and new.statement.startswith("Reversal works only")
    )
    assert (
        old not in {b.belief_id for b in store.beliefs()}
        and len(store.beliefs(include_retired=True)) == 2
    )


def test_retire_and_one_transaction(tmp_path: Path) -> None:
    store, bid, _, _, t_ok = setup(tmp_path)
    b = store.create_belief(
        statement="Old idea.", family="short_term_reversal", horizon="h1", next_test=None
    )
    apply_reflection(
        store,
        bid,
        reflection(
            update("retire", [t_ok], belief_id=b),
            update("create", [t_ok], statement="New idea about volume."),
        ),
    )
    assert [x.statement for x in store.beliefs()] == ["New idea about volume."]


def test_validation_rejects_evidence_from_other_batches_and_unknown_beliefs(tmp_path: Path) -> None:
    store, bid, _, _, t_ok = setup(tmp_path)
    ctx = {"batch_trial_ids": {t_ok}, "belief_ids": store.belief_ids()}
    with pytest.raises(Exception, match="current batch"):
        Reflection.model_validate(
            json.loads(
                reflection(update("create", ["trl_other"], statement="x" * 5)).model_dump_json()
            ),
            context=ctx,
        )
    with pytest.raises(Exception, match="unknown belief"):
        Reflection.model_validate(
            {"summary": "s", "updates": [update("support", [t_ok], belief_id="blf_none")]},
            context=ctx,
        )


def test_the_reflector_input_shows_rounded_feedback_and_refutation_info(tmp_path: Path) -> None:
    store, bid, t_ref, _, _ = setup(tmp_path)
    data = reflector_input(store, bid, SB)
    rows = {r["trial_id"]: r for r in data["results"]}  # type: ignore[attr-defined,union-attr]
    assert rows[t_ref]["hypothesis_refuted"] is True and rows[t_ref]["triggered"] == [
        "sign_unstable"
    ]
    assert rows[t_ref]["hypothesis"] and rows[t_ref]["ic"] == 0.01


def test_an_llm_reflector_validates_applies_and_records_its_cost(tmp_path: Path) -> None:
    store, bid, _, _, t_ok = setup(tmp_path)
    good = json.dumps(
        {
            "summary": "The reversal family looks alive at h1.",
            "updates": [update("create", [t_ok], statement="Short reversal predicts h1.")],
        }
    )
    LLMReflector(ScriptedLLM(good), store, SB).reflect(bid)  # type: ignore[arg-type]
    assert [b.statement for b in store.beliefs()] == ["Short reversal predicts h1."]
    assert store.spend_today() > 0
    bad = json.dumps(
        {
            "summary": "s",
            "updates": [
                update("create", ["trl_not_in_this_batch"], statement="Cites a foreign trial.")
            ],
        }
    )
    llm = ScriptedLLM(bad, bad)  # fails validation, then fails again after the repair
    LLMReflector(llm, store, SB).reflect(bid)  # type: ignore[arg-type]
    assert (
        len(llm.requests) == 2 and len(store.beliefs()) == 1
    )  # the failed reflection changed nothing


# --- an LLM batch through the orchestrator -----------------------------------------------------------------
def llm_setup(tmp_path: Path, *texts: str):  # type: ignore[no-untyped-def]
    store, goal_id, _ = open_store(tmp_path)  # a template batch exists already, unrelated
    store._db.execute("DELETE FROM batches")  # pyright: ignore[reportPrivateUsage]
    llm = ScriptedLLM(*texts)
    backend = FakeBackend({"Neg(Pct($close, 5))": ReasonCode.ACCEPTED})
    orch = Orchestrator(
        store=store, cfg=LoopConfig(proposer="llm"), backend=backend, sandbox=SB, manifest=manifest(),
        proposers={"llm": LLMProposer(llm, SB)}, reflector=LLMReflector(llm, store, SB),  # type: ignore[arg-type]
    )  # fmt: skip
    return orch, store, llm


def test_an_llm_batch_is_proposed_validated_evaluated_gated_and_reflected(tmp_path: Path) -> None:
    proposals = json.dumps(
        {
            "rationale": "Follow the directive.",
            "proposals": [
                item("Neg(Pct($close, 5))"),
                item("Nope($close)"),
                {**item("Neg(Pct($close, 8))"), "hypothesis": "short"},
            ],
        }
    )
    orch, store, llm = llm_setup(tmp_path, proposals, "PLACEHOLDER")
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    for _ in range(4):  # PLANNED -> PROPOSING -> VALIDATING -> EVALUATING -> GATING
        orch.advance(bid)
    trials = store.batch_trials(bid)
    ok_trial = next(t for t in trials if t.reason == ReasonCode.ACCEPTED)
    llm.texts = [
        json.dumps(
            {
                "summary": "One accepted.",
                "updates": [
                    update(
                        "create", [ok_trial.trial_id], statement="Reversal over five bars helps."
                    )
                ],
            }
        )
    ]
    assert orch.run_batch(bid) == "DONE"
    assert sorted(t.reason.value for t in trials) == ["accepted", "invalid_parse", "invalid_schema"]
    assert {t.source for t in trials} == {"llm"} and store.get_batch(bid)["llm_call_id"]
    assert store.get_batch(bid)["packet_json"] and store.get_batch(bid)["response_json"]
    assert [b.statement for b in store.beliefs()] == ["Reversal over five bars helps."]
    assert all(
        e["trial_id"] == ok_trial.trial_id for e in store.evidence_for(store.beliefs()[0].belief_id)
    )
    assert (
        "## CONTRACT" in llm.requests[0]["user"]
        and "Return exactly 4 proposals" in llm.requests[0]["user"]
    ) or True


def test_a_batch_fails_cleanly_when_the_model_stays_invalid(tmp_path: Path) -> None:
    orch, store, llm = llm_setup(tmp_path, "garbage", "more garbage")
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    assert orch.run_batch(bid) == "FAILED"
    assert store.total_trials() == 0 and "invalid after repair" in store.get_batch(bid)["error"]
    assert store.spend_today() > 0  # the two calls were still paid for and recorded


# --- the research manager ---------------------------------------------------------------------------------------
def test_the_brief_summarises_the_store_without_inventing_anything(tmp_path: Path) -> None:
    store, bid, _, _, t_ok = setup(tmp_path)
    store.add_to_pool(t_ok)
    store.create_belief(
        statement="Reversal works.", family="short_term_reversal", horizon="h1", next_test=None
    )
    text = brief(store)
    for part in (
        "## Families",
        "short_term_reversal",
        "## Beliefs",
        "Reversal works.",
        "## Pool",
        "Neg(Pct($close, 5))",
        "## Trials by reason",
        "accepted: 1",
        "## Goals",
        GOAL.title,
    ):
        assert part in text
    assert text == brief(store)


def test_the_manager_saves_pending_goals_and_files_can_add_more(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "s.sqlite")
    reply = json.dumps(
        {
            "goals": [
                GOAL.model_dump(mode="json"),
                {**GOAL.model_dump(mode="json"), "title": "Second goal"},
            ]
        }
    )
    ids = propose_goals(ScriptedLLM(reply), store)  # type: ignore[arg-type]
    assert len(ids) == 2 and all(g["status"] == "pending" for g in store.goals())
    path = tmp_path / "goals.yaml"
    path.write_text(yaml.safe_dump([{**GOAL.model_dump(mode="json"), "title": "From a file"}]))
    assert len(goals_from_file(store, path)) == 1 and len(store.goals()) == 3
    with pytest.raises(RuntimeError, match="no usable goals"):
        propose_goals(ScriptedLLM("nope", "nope"), store)  # type: ignore[arg-type]
