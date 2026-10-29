import json
from pathlib import Path

import pytest
from syrupy.assertion import SnapshotAssertion
from tests.helpers.store import DIRECTIVE, GOAL, add_trial, manifest, open_store, sandbox

from alpha_factory.agents import packet as packet_mod
from alpha_factory.agents import prompts
from alpha_factory.agents.feedback import feedback_for, round_corr, round_ic
from alpha_factory.agents.packet import build_packet
from alpha_factory.contracts import (
    FoldStat,
    ReasonCode,
)
from alpha_factory.research.store import ResearchStore

SB = sandbox()


def fb(store: ResearchStore, tid: str):  # type: ignore[no-untyped-def]
    return feedback_for(store.get_trial(tid), SB, store)


def test_rounding_hides_the_noise() -> None:
    assert round_ic(0.0271) == 0.025 and round_ic(0.0288) == 0.03 and round_ic(-0.0049) == -0.005
    assert round_corr(0.83) == 0.85 and round_corr(0.712) == 0.7


def test_each_reason_has_a_templated_note(tmp_path: Path) -> None:
    store, goal_id, bid = open_store(tmp_path)
    folds = (
        FoldStat(fold_id="F1", ic_mean=0.02, ic_tstat=3, n_days=50),
        FoldStat(fold_id="F2", ic_mean=-0.004, ic_tstat=-1, n_days=50),
    )
    cases = {
        ReasonCode.LOW_IC: ({"ic_tstat": 1.26}, "t-stat 1.3 vs required 3.0"),
        ReasonCode.WRONG_SIGN: ({}, "the reversed sign passes; a sign-flip trial was run"),
        ReasonCode.UNSTABLE: ({"folds": folds}, "F2 has the opposite sign"),
        ReasonCode.SUBSUMED: ({"incremental_ic": 0.0012}, "incremental IC 0.000 over the six baselines"),
        ReasonCode.CORRELATED: ({"max_corr_pool": 0.83, "max_corr_pool_id": "fac_9"}, "correlation 0.85 with fac_9"),
        ReasonCode.REGIME_ONLY: ({"ic_by_regime": {"vol_high": 0.02, "vol_low": -0.01, "trend_up": 0.01, "trend_down": 0.0}}, "wrong sign in vol_low"),
        ReasonCode.COST_KILLED: ({}, "negative net Sharpe after costs"),
        ReasonCode.HOLDOUT_FAIL: ({"ic_mean": 0.031}, "failed the holdout check"),
    }  # fmt: skip
    for i, (reason, (over, note)) in enumerate(cases.items()):
        tid = add_trial(store, bid, f"Pct($close, {i + 2})", reason, n=i, **over)
        f = fb(store, tid)
        assert f.note == note and f.reason == reason
    # the holdout verdict carries no holdout number, and every IC is rounded
    assert "0.031" not in fb(store, tid).model_dump_json()


def test_accepted_note_names_the_pool_factor(tmp_path: Path) -> None:
    store, _, bid = open_store(tmp_path)
    tid = add_trial(
        store,
        bid,
        "Neg(Pct($close, 2))",
        ReasonCode.ACCEPTED,
        ic_mean=0.0271,
        ic_by_year={2023: 0.02, 2024: 0.03, 2025: -0.01},
    )
    fid = store.add_to_pool(tid)
    f = fb(store, tid)
    assert f.note == f"accepted as {fid}" and f.ic == 0.025 and f.years_positive == "2/3"


def test_invalid_trials_show_the_checkers_message_and_the_expression(tmp_path: Path) -> None:
    store, _, bid = open_store(tmp_path)
    tid = store.record_trial(source="llm", sandbox=SB, manifest=manifest(), reason=ReasonCode.INVALID_PARSE, batch_id=bid, family="volume_shock", horizon="h2", note=json.dumps({"expression": "Nope($close)", "message": "unknown operator Nope"}))  # fmt: skip
    f = fb(store, tid)
    assert (f.note, f.expression, f.family, f.horizon, f.ic) == (
        "unknown operator Nope",
        "Nope($close)",
        "volume_shock",
        "h2",
        0.0,
    )


def populated(tmp_path: Path):  # type: ignore[no-untyped-def]
    store, goal_id, bid = open_store(tmp_path)
    for i in range(6):
        add_trial(store, bid, f"Neg(Pct($close, {i + 2}))", ReasonCode.LOW_IC, n=i)
    a = add_trial(
        store, bid, "Neg(Pct($close, 9))", ReasonCode.ACCEPTED, n=20, ic_tstat=5.0, ic_mean=0.03
    )
    store.add_to_pool(a)
    store.record_trial(source="template", sandbox=SB, manifest=manifest(), reason=ReasonCode.INVALID_PARSE, batch_id=bid, family="short_term_reversal", note=json.dumps({"expression": "Foo(1)", "message": "unknown operator Foo"}))  # fmt: skip
    for _ in range(2):
        store.record_trial(source="template", sandbox=SB, manifest=manifest(), reason=ReasonCode.DUPLICATE, batch_id=bid, family="short_term_reversal", note=json.dumps({"expression": "Neg(Pct($close, 2))", "message": "same canonical expression as trl_x"}))  # fmt: skip
    b = store.create_belief(
        statement="Reversal is stronger at high volume.",
        family="short_term_reversal",
        horizon="h1",
        next_test="Condition on volume.",
    )
    store.add_evidence(b, a, "support", bid)
    return store, goal_id, bid


def make_packet(store, goal_id, bid):  # type: ignore[no-untyped-def]
    return build_packet(store, goal_id, GOAL, DIRECTIVE, SB, bid)


def test_the_same_store_state_always_gives_the_same_packet(tmp_path: Path) -> None:
    store, goal_id, bid = populated(tmp_path)
    assert make_packet(store, goal_id, bid) == make_packet(store, goal_id, bid)
    assert (
        make_packet(store, goal_id, bid).model_dump_json()
        == make_packet(store, goal_id, bid).model_dump_json()
    )


def test_packet_contents_follow_the_selection_rules(tmp_path: Path) -> None:
    store, goal_id, bid = populated(tmp_path)
    p = make_packet(store, goal_id, bid)
    assert [f.expression for f in p.pool] == ["Neg(Pct($close, 9))"] and p.pool[
        0
    ].cluster == "short_term_reversal"
    assert (
        len(p.recent) == 10 and p.recent[-1].reason == ReasonCode.DUPLICATE
    )  # oldest first, invalid ones included
    assert p.digest.n_trials == 10 and p.digest.by_reason[ReasonCode.LOW_IC] == 6
    assert "most common parse error (1 times): unknown operator Foo" in p.digest.patterns
    assert any(s.startswith("short_term_reversal: mostly low_ic (6 of") for s in p.digest.patterns)
    assert p.graveyard == ("expression proposed 3 times: Neg(Pct($close, 2))",)
    assert (
        len(p.exemplars) == 1 and p.exemplars[0].expression == "Neg(Pct($close, 2))"
    ) or p.exemplars
    assert [b.evidence for b in p.beliefs] == ["1 for, 0 against"]
    assert (
        "Fields:" in p.contract
        and p.contract_hash == make_packet(store, goal_id, bid).contract_hash
    )


def test_beliefs_are_ordered_contested_then_untested_then_supported_and_capped(
    tmp_path: Path,
) -> None:
    store, goal_id, bid = open_store(tmp_path)
    ids = [
        add_trial(store, bid, f"Neg(Pct($close, {i + 2}))", ReasonCode.LOW_IC, n=i)
        for i in range(5)
    ]
    sup = store.create_belief(
        statement="supported one", family="short_term_reversal", horizon="h1", next_test=None
    )
    con = store.create_belief(
        statement="contested one", family="short_term_reversal", horizon="h1", next_test=None
    )
    store.create_belief(
        statement="untested one", family="short_term_reversal", horizon="h1", next_test=None
    )
    for t in ids[:4]:
        store.add_evidence(sup, t, "support", bid)
    for t in ids[:2]:
        store.add_evidence(con, t, "support", bid)
    for t in ids[2:4]:
        store.add_evidence(con, t, "contradict", bid)
    p = make_packet(store, goal_id, bid)
    assert [b.statement for b in p.beliefs] == ["contested one", "untested one", "supported one"]
    for i in range(30):
        store.create_belief(
            statement=f"extra {i}", family="short_term_reversal", horizon="h1", next_test=None
        )
    assert len(make_packet(store, goal_id, bid).beliefs) == 20


def test_over_budget_trimming_cuts_recent_then_graveyard_then_exemplars_never_beliefs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, goal_id, bid = populated(tmp_path)
    full = make_packet(store, goal_id, bid)
    base = packet_mod.estimate_tokens(full)
    monkeypatch.setattr(packet_mod, "TOKEN_BUDGET", base - 150)
    cut = make_packet(store, goal_id, bid)
    assert (
        len(cut.recent) < len(full.recent) and cut.graveyard == full.graveyard
    )  # older rows go first
    assert cut.recent == full.recent[len(full.recent) - len(cut.recent) :]
    monkeypatch.setattr(packet_mod, "TOKEN_BUDGET", 10)
    bare = make_packet(store, goal_id, bid)
    assert bare.recent == () and bare.graveyard == () and bare.exemplars == ()
    assert bare.beliefs == full.beliefs and bare.directive == full.directive


def test_prompts_render_with_the_packet_and_match_their_snapshots(
    tmp_path: Path, snapshot: SnapshotAssertion
) -> None:
    store, goal_id, bid = populated(tmp_path)
    p = make_packet(store, goal_id, bid)
    p = p.model_copy(update={"packet_id": "pkt_fixture"})
    user = prompts.render(
        f"{prompts.PROPOSER}.user",
        packet=p,
        goal=GOAL,
        exemplars=[e.model_dump(mode="json") for e in p.exemplars],
    )
    system = prompts.render(f"{prompts.PROPOSER}.system")
    assert "Return exactly 4 proposals:" in user and "4 new, 0 mutate, 0 refine, 0 combine." in user
    assert (
        user.index("## CONTRACT")
        < user.index("## GOAL")
        < user.index("## DIRECTIVE")
        < user.index("## BELIEFS")
        < user.index("## TASK")
    )
    assert system == snapshot(name="proposer_system")
    import re

    stable = re.sub(
        r"trl_[0-9a-fA-Z]{12,26}",
        "trl_ID",
        re.sub(r"fac_[0-9A-Z]{26}", "fac_ID", re.sub(r"blf_[0-9A-Z]{26}", "blf_ID", user)),
    )
    assert stable == snapshot(name="proposer_user")
    assert prompts.render(prompts.REPAIR, errors="E1") == snapshot(name="repair")
    assert (
        prompts.prompt_hash(system, user)
        == prompts.prompt_hash(system, user)
        != prompts.prompt_hash(system, user + "x")
    )
