import sqlite3
from pathlib import Path

import pytest
from tests.helpers.store import GOAL, manifest, open_store, proposal, result, sandbox

from alpha_factory.contracts import ReasonCode
from alpha_factory.research.store import ResearchStore

SB, MF = sandbox(), manifest()


def record(store: ResearchStore, batch_id: str, expr: str = "Neg(Pct($close, 2))", **kw):  # type: ignore[no-untyped-def]
    """Store a proposal, then record one trial for it (as the orchestrator does)."""
    goal_id = store.get_batch(batch_id)["goal_id"]
    prop = proposal(batch_id, goal_id, expr)
    store.add_proposals([prop])
    reason = kw.pop("reason", ReasonCode.LOW_IC)
    sign = kw.pop("sign", 1)
    res = result(
        SB,
        prop.expr_hash,
        reason,
        trial_id=kw.pop("trial_id", f"trl_{prop.expr_hash[-12:]}_{reason.value}"),
        sign=sign,
    )
    tid = store.record_trial(
        source="template", sandbox=SB, manifest=MF, reason=reason, proposal=prop, result=res, **kw
    )
    return tid, prop


def test_the_schema_is_created_in_wal_mode(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "s.sqlite")
    assert store._db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"  # pyright: ignore[reportPrivateUsage]
    ResearchStore(tmp_path / "s.sqlite")  # reopening is fine


def test_goal_lifecycle_and_the_one_active_goal_queue(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path / "s.sqlite")
    a, b = store.add_goal(GOAL), store.add_goal(GOAL)
    assert store.active_goal_id() is None
    store.approve_goal(b)
    assert store.active_goal_id() == b  # approval order, not creation order
    store.approve_goal(a)
    assert store.active_goal_id() == b  # a waits in the queue
    store.close_goal(b, "done", "finished")
    assert store.activate_next_goal() == a and store.active_goal_id() == a
    with pytest.raises(KeyError):
        store.approve_goal("gol_missing")


def test_record_trial_inserts_trial_result_and_bumps_the_goal_counter(tmp_path: Path) -> None:
    store, goal_id, batch_id = open_store(tmp_path)
    tid, prop = record(store, batch_id)
    t = store.get_trial(tid)
    assert t.reason == ReasonCode.LOW_IC and t.evaluated and t.family == "short_term_reversal"
    res, _n, ok = store.get_eval(tid) or (None, 0, None)
    assert res is not None and res.expr_hash == prop.expr_hash and ok is None
    assert store.n_evaluated(SB.content_hash()) == 1
    assert next(g for g in store.goals() if g["goal_id"] == goal_id)["trials_used"] == 1
    assert "short_term_reversal" in store.families()


def test_recording_the_same_evaluation_twice_is_a_no_op(tmp_path: Path) -> None:
    store, goal_id, batch_id = open_store(tmp_path)
    first, _ = record(store, batch_id)
    again, _ = record(store, batch_id, trial_id="trl_second")
    assert again == first and store.n_evaluated(SB.content_hash()) == 1
    assert next(g for g in store.goals() if g["goal_id"] == goal_id)["trials_used"] == 1
    h = proposal(batch_id, "g").expr_hash
    assert store.seen(SB.content_hash(), h, "h1", 1)
    assert not store.seen(SB.content_hash(), h, "h1", -1)


def test_a_flipped_sign_is_a_separate_counted_trial(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    _, prop = record(store, batch_id, "Pct($close, 7)", reason=ReasonCode.WRONG_SIGN)
    flipped = result(SB, prop.expr_hash, ReasonCode.LOW_IC, trial_id="trl_flip", sign=-1)
    store.record_trial(
        source="sign_flip",
        sandbox=SB,
        manifest=MF,
        reason=ReasonCode.LOW_IC,
        proposal=prop,
        result=flipped,
    )
    assert store.n_evaluated(SB.content_hash()) == 2


def test_verdicts_that_test_nothing_do_not_count_toward_n(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    for reason in (
        ReasonCode.INVALID_PARSE,
        ReasonCode.TOO_COMPLEX,
        ReasonCode.DUPLICATE,
        ReasonCode.INVALID_SCHEMA,
    ):
        store.record_trial(
            source="template", sandbox=SB, manifest=MF, reason=reason, batch_id=batch_id
        )
        store.record_trial(
            source="template", sandbox=SB, manifest=MF, reason=reason, batch_id=batch_id
        )
    assert store.total_trials() == 8  # every trial is recorded, invalid ones included
    assert store.n_evaluated(SB.content_hash()) == 0
    record(store, batch_id)
    assert store.n_evaluated(SB.content_hash()) == 1


def test_n_is_counted_per_sandbox(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    record(store, batch_id)
    other = SB.model_copy(update={"sandbox_version": "other"})
    assert store.n_evaluated(other.content_hash()) == 0


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE trials SET reason='accepted'",
        "DELETE FROM trials",
        "UPDATE eval_results SET ic_tstat=9",
        "DELETE FROM eval_results",
        "UPDATE llm_calls SET cost_usd=0",
        "DELETE FROM llm_calls",
    ],
)
def test_append_only_tables_reject_updates_and_deletes(tmp_path: Path, sql: str) -> None:
    from datetime import UTC, datetime

    from alpha_factory.contracts import LLMCallRecord

    store, _, batch_id = open_store(tmp_path)
    record(store, batch_id)
    store.add_llm_call(
        LLMCallRecord(
            call_id="llm_1",
            role="proposer",
            provider="p",
            model="m",
            prompt_hash="a" * 64,
            contract_hash=None,
            tokens_in=1,
            tokens_out=1,
            latency_ms=1,
            cost_usd=0.1,
            ok=True,
            error=None,
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )  # fmt: skip
    )
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._db.execute(sql)  # pyright: ignore[reportPrivateUsage]


def test_final_evals_are_append_only_too(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from alpha_factory.contracts import CombinerSpec, FinalEvalReport, FrozenLibrary

    store, _, _ = open_store(tmp_path)
    lib = FrozenLibrary(
        lib_version="lib_v1", sandbox_hash="a" * 64, factors=(),
        combiner=CombinerSpec(kind="rank_average"), frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
    )  # fmt: skip
    store.add_library(lib)
    store.record_final_eval(
        "lib_v1",
        FinalEvalReport(
            request_hash="b" * 64, ic_mean=0.01, ic_tstat=2.0, net_sharpe=0.5, net_sharpe_by_year={2026: 0.5},
            turnover=1.0, cost_share_of_gross=0.5, created_at=datetime(2026, 1, 1, tzinfo=UTC),
        ),
    )  # fmt: skip
    assert len(store.final_evals()) == 1
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store._db.execute("DELETE FROM final_evals")  # pyright: ignore[reportPrivateUsage]


def test_a_failed_transaction_rolls_everything_back(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    with pytest.raises(RuntimeError), store.tx():
        record(store, batch_id)
        raise RuntimeError("crash mid step")
    assert store.total_trials() == 0 and store.n_evaluated(SB.content_hash()) == 0


def test_transactions_nest(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    with store.tx():
        record(store, batch_id)
        with store.tx():
            record(store, batch_id, "Pct($close, 3)")
    assert store.total_trials() == 2


def test_family_stats_use_the_most_recent_evaluated_trials(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    reasons = [ReasonCode.ACCEPTED, ReasonCode.LOW_IC, ReasonCode.LOW_IC, ReasonCode.ACCEPTED]
    for i, r in enumerate(reasons):
        record(store, batch_id, f"Pct($close, {i + 2})", reason=r)
    assert store.family_stats("short_term_reversal", 200) == (2, 4)
    assert store.family_stats("short_term_reversal", 2) == (1, 2)  # the last two: LOW_IC, ACCEPTED
    assert store.family_totals("short_term_reversal") == (4, 2, 0)


def test_beliefs_have_derived_counts_and_evidence_is_idempotent(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    t1, _ = record(store, batch_id, "Pct($close, 2)")
    t2, _ = record(store, batch_id, "Pct($close, 3)")
    b = store.create_belief(
        statement="Reversal is stronger on volume.",
        family="short_term_reversal",
        horizon="h1",
        next_test=None,
    )
    store.add_evidence(b, t1, "support", batch_id)
    store.add_evidence(b, t1, "support", batch_id)  # repeat is dropped by the primary key
    store.add_evidence(b, t2, "contradict", batch_id)
    (belief,) = store.beliefs()
    assert (belief.n_support, belief.n_contradict) == (1, 1)
    store.retire_belief(b)
    assert store.beliefs() == [] and len(store.beliefs(include_retired=True)) == 1


def test_pool_listing_carries_the_expression_and_evidence(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    tid, prop = record(store, batch_id, reason=ReasonCode.ACCEPTED)
    fid = store.add_to_pool(tid)
    (row,) = store.pool()
    assert row.factor_id == fid and row.canonical_expr == prop.canonical_expr
    assert row.lib_factor().sign == 1
    store.remove_from_pool(fid, "replaced by fac_x")
    assert store.pool() == []
