from pathlib import Path

import pytest
from tests.helpers.orchestrator import FakeBackend, item, make
from tests.helpers.store import sandbox

from alpha_factory.contracts import ReasonCode, ResearchGoal
from alpha_factory.orchestrator import (
    BudgetExceededError,
    LoopAlreadyRunningError,
    LoopLock,
    Orchestrator,
)
from alpha_factory.research.config import LoopConfig
from alpha_factory.research.notify import Acceptance, GoalEnded, LoopPaused
from alpha_factory.research.store import ResearchStore

SB = sandbox()
A, B, C = "Neg(Pct($close, 2))", "Neg(Pct($close, 3))", "Neg(Pct($close, 5))"


class Crash(BaseException):
    """Stands in for a kill -9: not an Exception, so nothing in the loop can swallow it."""


def test_a_batch_walks_through_every_state_and_records_every_trial(tmp_path: Path) -> None:
    orch, store, backend, _, _ = make(tmp_path, [item(A), item(B), item(C)])
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    seen = [store.get_batch(bid)["state"]]
    while seen[-1] not in ("DONE", "FAILED"):
        seen.append(orch.advance(bid))
    assert seen == [
        "PLANNED",
        "PROPOSING",
        "VALIDATING",
        "EVALUATING",
        "GATING",
        "REFLECTING",
        "DONE",
    ]
    assert [t.reason for t in store.batch_trials(bid)] == [ReasonCode.LOW_IC] * 3
    assert store.n_evaluated(SB.content_hash()) == 3 and len(backend.requests) == 1
    assert backend.requests[0].n_evaluated == 0 and len(backend.requests[0].specs) == 3


def test_invalid_proposals_become_counted_trials_with_their_reason(tmp_path: Path) -> None:
    bad_schema = {**item(A), "hypothesis": "short"}
    parse_error = item("Nope($close)")
    too_complex = item("Mean(Mean(Mean(Mean(Mean(Mean(Mean($close, 2), 2), 2), 2), 2), 2), 2)")
    good = item(B)
    dup_in_batch = item("Neg(Pct($close, 3))")  # same canonical expression as `good`
    orch, store, backend, _, _ = make(
        tmp_path, [bad_schema, parse_error, too_complex, good, dup_in_batch]
    )
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    orch.run_batch(bid)
    reasons = sorted(t.reason.value for t in store.batch_trials(bid))
    assert reasons == sorted(
        ["invalid_schema", "invalid_parse", "too_complex", "duplicate", "low_ic"]
    )
    assert store.total_trials() == 5 and store.n_evaluated(SB.content_hash()) == 1
    assert len(backend.requests[0].specs) == 1  # only the valid, novel proposal is evaluated
    parse = next(t for t in store.batch_trials(bid) if t.reason == ReasonCode.INVALID_PARSE)
    assert "Nope" in (parse.note or "") and parse.family == "short_term_reversal"


def test_a_duplicate_of_an_earlier_batch_is_caught(tmp_path: Path) -> None:
    orch, store, _, _, _ = make(tmp_path, [item(A)], [item(A)])
    for _ in range(2):
        orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    reasons = [t.reason for t in store.goal_trials(store.active_goal_id())]  # type: ignore[arg-type]
    assert sorted(r.value for r in reasons) == ["duplicate", "low_ic"]
    assert store.n_evaluated(SB.content_hash()) == 1


def test_wrong_sign_runs_a_counted_sign_flip_trial(tmp_path: Path) -> None:
    backend = FakeBackend({(A, 1): ReasonCode.WRONG_SIGN, (A, -1): ReasonCode.ACCEPTED})
    orch, store, _, _, events = make(tmp_path, [item(A)], backend=backend)
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    orch.run_batch(bid)
    trials = store.batch_trials(bid)
    assert [(t.source, t.sign_used, t.reason.value) for t in trials] == [
        ("template", 1, "wrong_sign"), ("sign_flip", -1, "accepted"),
    ]  # fmt: skip
    assert (
        store.n_evaluated(SB.content_hash()) == 2
    )  # testing the reversed sign is a real extra test
    assert len(store.pool()) == 1 and store.pool()[0].sign_used == -1
    assert len(backend.requests) == 2 and backend.requests[1].specs[0].sign == -1
    assert any(isinstance(e, Acceptance) for e in events)


def test_accepted_factors_enter_the_pool_and_holdout_failures_do_not(tmp_path: Path) -> None:
    backend = FakeBackend(
        {A: ReasonCode.ACCEPTED, B: ReasonCode.HOLDOUT_FAIL, C: ReasonCode.ACCEPTED}
    )
    backend.holdout = {A: True}
    orch, store, _, _, events = make(
        tmp_path, [item(A), item(B), item(C, family="volume_shock")], backend=backend
    )
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    assert {p.canonical_expr for p in store.pool()} == {A, C}
    acc = [e for e in events if isinstance(e, Acceptance)]
    assert len(acc) == 2 and acc[0].t_required == pytest.approx(3.0, abs=0.5)


def test_the_pool_is_passed_to_later_evaluations(tmp_path: Path) -> None:
    backend = FakeBackend({A: ReasonCode.ACCEPTED})
    orch, store, _, _, _ = make(
        tmp_path,
        [item(A)],
        [item(B, **{"type": "mutate", "parents": ["PLACEHOLDER"]})],
        backend=backend,
    )
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    (factor,) = store.pool()
    orch.proposers["template"].batches[0][0]["parents"] = [factor.factor_id]  # type: ignore[attr-defined]
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    second = backend.requests[-1]
    assert [f.canonical_expr for f in second.pool] == [A]
    assert second.specs[0].parent_exprs == (A,) and second.n_evaluated == 1


def test_an_unknown_parent_is_an_invalid_schema(tmp_path: Path) -> None:
    orch, store, _, _, _ = make(
        tmp_path, [item(A, **{"type": "mutate", "parents": ["fac_made_up"]})]
    )
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    orch.run_batch(bid)
    assert [t.reason for t in store.batch_trials(bid)] == [ReasonCode.INVALID_SCHEMA]


# --- replacement and the cap -----------------------------------------------------------------------
def test_a_correlated_only_trial_replaces_a_pool_factor_at_one_point_two_times_its_t(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        {A: ReasonCode.ACCEPTED, B: ReasonCode.CORRELATED, C: ReasonCode.CORRELATED},
        tstat={A: 4.0, B: 4.7, C: 4.9},
    )
    orch, store, _, _, _ = make(tmp_path, [item(A)], [item(B), item(C)], backend=backend)
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    (old,) = store.pool()
    backend.max_corr_id = {B: old.factor_id, C: old.factor_id}
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    # B has t 4.7 < 1.2 * 4.0 = 4.8, so it does not replace; C has 4.9 >= 4.8, so it does
    assert [p.canonical_expr for p in store.pool()] == [C]
    removed = store._db.execute(
        "SELECT removed_reason FROM pool WHERE factor_id=?", (old.factor_id,)
    ).fetchone()  # pyright: ignore[reportPrivateUsage]
    assert removed[0].startswith("replaced by fac_")


def test_correlated_with_other_failures_or_a_failed_holdout_never_replaces(tmp_path: Path) -> None:
    backend = FakeBackend(
        {A: ReasonCode.ACCEPTED, B: ReasonCode.CORRELATED}, tstat={A: 3.0, B: 9.0}
    )
    orch, store, _, _, _ = make(tmp_path, [item(A)], [item(B)], backend=backend)
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    backend.max_corr_id = {B: store.pool()[0].factor_id}
    backend.holdout = {B: False}
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    assert [p.canonical_expr for p in store.pool()] == [A]


def test_over_the_cap_the_factor_whose_removal_hurts_least_leaves(tmp_path: Path) -> None:
    exprs = [f"Neg(Pct($close, {n}))" for n in (2, 3, 5, 8)]
    backend = FakeBackend({e: ReasonCode.ACCEPTED for e in exprs})
    cfg = LoopConfig(pool_cap=3)
    orch, store, _, _, _ = make(tmp_path, [item(e) for e in exprs], backend=backend, cfg=cfg)
    seen: list[int] = []

    def loo(req):  # type: ignore[no-untyped-def]
        seen.append(len(req.pool))
        by = {f.canonical_expr: f.factor_id for f in req.pool}
        # dropping the 5-bar factor leaves the best combined IC, so it is the least useful
        return {fid: (0.05 if e == exprs[2] else 0.01) for e, fid in by.items()}

    backend.loo = loo
    orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
    assert seen == [4] and {p.canonical_expr for p in store.pool()} == {
        exprs[0],
        exprs[1],
        exprs[3],
    }


# --- crash recovery -------------------------------------------------------------------------------------
POINTS = [
    f"{w}:{s}"
    for s in ("PLANNED", "PROPOSING", "VALIDATING", "EVALUATING", "GATING", "REFLECTING")
    for w in ("before", "after")
] + ["evaluated", "recorded"]


@pytest.mark.parametrize("point", POINTS)
def test_killing_the_process_at_any_point_loses_and_duplicates_nothing(
    tmp_path: Path, point: str
) -> None:
    items = [item(A), item(B), item(C), item("Nope($close)")]
    backend = FakeBackend({A: ReasonCode.ACCEPTED, B: ReasonCode.WRONG_SIGN})

    def baseline() -> tuple[int, int, int, int]:
        orch, store, _, _, _ = make(
            tmp_path / "base",
            list(items),
            backend=FakeBackend({A: ReasonCode.ACCEPTED, B: ReasonCode.WRONG_SIGN}),
        )
        orch.run_batch(orch.plan_batch(store.active_goal_id()))  # type: ignore[arg-type]
        return (
            store.total_trials(),
            store.n_evaluated(SB.content_hash()),
            len(store.pool()),
            int(store.goals()[0]["trials_used"]),
        )

    expected = baseline()
    orch, store, _, proposer, _ = make(tmp_path / "crash", list(items), backend=backend)
    fired = {"n": 0}

    def fault(p: str) -> None:
        if p == point and fired["n"] == 0:
            fired["n"] += 1
            raise Crash(p)

    orch.fault = fault
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    with pytest.raises(Crash):
        orch.run_batch(bid)
    # a fresh process opens the same store file and resumes
    store2 = ResearchStore(tmp_path / "crash" / "store.sqlite")
    resumed = Orchestrator(
        store=store2,
        cfg=LoopConfig(),
        backend=backend,
        sandbox=SB,
        manifest=orch.manifest,
        proposers=orch.proposers,
    )
    assert resumed.run_once()
    assert store2.get_batch(bid)["state"] == "DONE"
    got = (
        store2.total_trials(),
        store2.n_evaluated(SB.content_hash()),
        len(store2.pool()),
        int(store2.goals()[0]["trials_used"]),
    )
    assert got == expected
    # the batch was proposed once (a saved response is never requested again); after DONE the
    # resumed loop simply starts the next, empty, scripted batch
    assert proposer.calls == (2 if point == "after:REFLECTING" else 1)


def test_a_stored_response_is_not_requested_again(tmp_path: Path) -> None:
    orch, store, _, proposer, _ = make(tmp_path, [item(A)], [item(B)])
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    orch.advance(bid)
    orch.advance(bid)  # PROPOSING done: response stored
    assert proposer.calls == 1
    store._db.execute("UPDATE batches SET state='PROPOSING' WHERE batch_id=?", (bid,))  # pyright: ignore[reportPrivateUsage]
    orch.advance(bid)
    assert proposer.calls == 1 and store.get_batch(bid)["state"] == "VALIDATING"


def test_evaluation_failures_keep_the_batch_resumable(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.fail_next = 1
    orch, store, _, _, _ = make(tmp_path, [item(A), item(B)], backend=backend)
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    with pytest.raises(Exception, match="boom"):
        orch.run_batch(bid)
    assert (
        store.get_batch(bid)["state"] == "EVALUATING" and store.n_evaluated(SB.content_hash()) == 0
    )
    orch.run_batch(bid)  # the worker recovered
    assert store.get_batch(bid)["state"] == "DONE" and store.n_evaluated(SB.content_hash()) == 2


# --- the loop, goals, budgets and pausing -----------------------------------------------------------------
def test_an_unusable_llm_response_fails_the_batch_without_trials(tmp_path: Path) -> None:
    from alpha_factory.agents.proposer import ProposalOutcome

    class Broken:
        def propose(self, *a: object, **k: object) -> ProposalOutcome:
            return ProposalOutcome(raw=None, failure="schema-invalid after repair")

    orch, store, _, _, _ = make(tmp_path)
    orch.proposers["template"] = Broken()  # type: ignore[assignment]
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    assert orch.run_batch(bid) == "FAILED"
    assert store.total_trials() == 0 and "schema-invalid" in store.get_batch(bid)["error"]


def test_the_goal_ends_when_its_trial_budget_is_used_and_the_next_goal_starts(
    tmp_path: Path,
) -> None:
    small = ResearchGoal(title="Tiny", question="A question long enough to pass.", families=["short_term_reversal"], horizons=["h1"], budget_trials=20, stop_when="Never.")  # fmt: skip
    items = [[item(f"Neg(Pct($close, {n}))") for n in (2, 3)], [item(f"Neg(Pct($open, {n}))") for n in range(2, 12)], [item(f"Neg(Pct($low, {n}))") for n in range(2, 12)]]  # fmt: skip
    orch, store, _, _, events = make(tmp_path, *items, goal=small)
    second = store.add_goal(small)
    store.approve_goal(second)  # waits in the queue
    g1 = store.active_goal_id()
    for _ in range(3):
        orch.run_once()
    assert (
        g1 is not None and next(g for g in store.goals() if g["goal_id"] == g1)["status"] == "done"
    )
    assert store.active_goal_id() == second
    ended = [e for e in events if isinstance(e, GoalEnded)]
    assert ended and ended[0].why == "trial budget used"


def test_a_goal_ends_after_batches_without_an_acceptance(tmp_path: Path) -> None:
    cfg = LoopConfig(goal={"stop_after_batches_without_accept": 3})  # type: ignore[arg-type]
    orch, store, _, _, events = make(
        tmp_path, *[[item(f"Neg(Pct($close, {n}))")] for n in (2, 3, 5, 8)], cfg=cfg
    )
    g1 = store.active_goal_id()
    for _ in range(3):
        assert orch.run_once()
    assert next(g for g in store.goals() if g["goal_id"] == g1)["status"] == "done"
    assert (
        "3 batches without an acceptance" in next(e for e in events if isinstance(e, GoalEnded)).why
    )
    assert not orch.run_once()  # nothing to do until you approve another goal


def test_nothing_runs_while_paused_and_the_daily_batch_cap_holds(tmp_path: Path) -> None:
    orch, store, _, proposer, _ = make(
        tmp_path,
        *[[item(f"Neg(Pct($close, {n}))")] for n in (2, 3, 5)],
        cfg=LoopConfig(max_batches_per_day=2),
    )
    store.set_flag("paused", "1")
    assert not orch.run_once() and proposer.calls == 0
    store.set_flag("paused", "0")
    assert orch.run_once() and orch.run_once() and not orch.run_once()  # the cap of 2
    assert proposer.calls == 2


def test_pausing_mid_batch_stops_between_states_and_resumes_later(tmp_path: Path) -> None:
    orch, store, _, _, _ = make(tmp_path, [item(A)])
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    orch.advance(bid)
    store.set_flag("paused", "1")
    assert orch.run_batch(bid) == "PROPOSING"
    store.set_flag("paused", "0")
    assert orch.run_batch(bid) == "DONE"


def test_the_llm_budget_is_checked_before_every_proposal(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from alpha_factory.contracts import LLMCallRecord

    cfg = LoopConfig(proposer="llm", daily_llm_budget_usd=1.0)
    orch, store, _, proposer, _ = make(tmp_path, [item(A)], cfg=cfg)
    store.add_llm_call(LLMCallRecord(call_id="llm_x", role="proposer", provider="p", model="m", prompt_hash="a" * 64, contract_hash=None, tokens_in=1, tokens_out=1, latency_ms=1, cost_usd=1.5, ok=True, error=None, created_at=datetime.now(UTC)))  # fmt: skip
    bid = orch.plan_batch(store.active_goal_id())  # type: ignore[arg-type]
    orch.advance(bid)
    with pytest.raises(BudgetExceededError):
        orch.advance(bid)
    assert proposer.calls == 0


def test_repeated_failures_pause_the_loop_and_alert(tmp_path: Path) -> None:
    backend = FakeBackend()
    backend.fail_next = 100
    orch, store, _, _, events = make(
        tmp_path, [item(A)], backend=backend, cfg=LoopConfig(max_consecutive_failures=3)
    )
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return store.paused() or calls["n"] > 20

    orch.run_forever(should_stop=stop, idle_sleep_s=0.0)
    assert store.paused() and "3 consecutive failures" in (store.get_flag("paused_reason") or "")
    assert any(isinstance(e, LoopPaused) for e in events)


def test_the_lock_file_keeps_two_loops_off_one_store(tmp_path: Path) -> None:
    path = tmp_path / "store.sqlite"
    with LoopLock(path), pytest.raises(LoopAlreadyRunningError):
        LoopLock(path).__enter__()
    with LoopLock(path):  # released on exit
        pass


def test_the_daily_digest_is_sent_once_after_8pm_bucharest_time(tmp_path: Path) -> None:
    from datetime import UTC, datetime, timedelta
    from zoneinfo import ZoneInfo

    from alpha_factory.research.notify import DailyDigest

    backend = FakeBackend({A: ReasonCode.ACCEPTED})
    orch, store, _, _, events = make(tmp_path, [item(A), item(B)], backend=backend)
    orch.run_once()  # trials are stamped with the real time, so build the fake clock on today's date
    bucharest = ZoneInfo("Europe/Bucharest")
    today = datetime.now(bucharest).replace(minute=30, second=0, microsecond=0)
    t = {"now": today.replace(hour=19).astimezone(UTC)}
    orch.clock = lambda: t["now"]
    assert not orch.maybe_digest()  # 19:30 local
    t["now"] = today.replace(hour=20).astimezone(UTC)
    assert orch.maybe_digest() and not orch.maybe_digest()  # once a day
    (digest,) = [e for e in events if isinstance(e, DailyDigest)]
    assert digest.acceptances == 1 and digest.pool_size == 1 and "Reversal" in digest.goal_progress
    assert sum(digest.trials_by_reason.values()) == 2
    t["now"] = t["now"] + timedelta(days=1)  # the next evening
    assert orch.maybe_digest()
