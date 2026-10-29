"""The research store: one SQLite file (WAL), accessed only through this module.

Rows hold full Pydantic records as JSON plus a few indexed columns. trials, eval_results, llm_calls
and final_evals are append-only (triggers). Every trial enters through `record_trial`, so the trial
count behind the gate's deflation cannot be bypassed, and recording the same evaluation twice is a
no-op, which makes re-running a half-finished state safe.
"""

import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from alpha_factory.contracts import (
    BatchDirective,
    Belief,
    DataManifest,
    EvalResult,
    FinalEvalReport,
    FrozenLibrary,
    Horizon,
    LibFactor,
    LLMCallRecord,
    ProposalRecord,
    ReasonCode,
    ResearchGoal,
    SandboxSpec,
)
from alpha_factory.ids import new_id

TrialSource = Literal["llm", "template", "sign_flip", "manual"]
BatchState = Literal[
    "PLANNED", "PROPOSING", "VALIDATING", "EVALUATING", "GATING", "REFLECTING", "DONE", "FAILED"
]
# Verdicts reached before any IC was computed: they test nothing, so they do not count toward N.
NOT_EVALUATED = frozenset(
    {
        ReasonCode.INVALID_SCHEMA,
        ReasonCode.INVALID_PARSE,
        ReasonCode.TOO_COMPLEX,
        ReasonCode.LOOKAHEAD,
        ReasonCode.DUPLICATE,
    }
)
MIGRATIONS = Path(__file__).parent / "migrations"


def now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class TrialRow:
    trial_id: str
    batch_id: str | None
    proposal_id: str | None
    source: str
    expr_hash: str | None
    family: str | None
    horizon: str | None
    sign_used: int | None
    reason: ReasonCode
    evaluated: bool
    created_at: str
    note: str | None = None


@dataclass(frozen=True)
class PoolRow:
    factor_id: str
    trial_id: str
    expr_hash: str
    family: str
    horizon: Horizon
    sign_used: int
    added_at: str
    canonical_expr: str
    ic_tstat: float
    ic_mean: float

    def lib_factor(self) -> LibFactor:
        return LibFactor(
            factor_id=self.factor_id,
            canonical_expr=self.canonical_expr,
            sign=1 if self.sign_used > 0 else -1,
            horizon=self.horizon,
        )


def _trial(r: sqlite3.Row) -> TrialRow:
    return TrialRow(
        r["trial_id"],
        r["batch_id"],
        r["proposal_id"],
        r["source"],
        r["expr_hash"],
        r["family"],
        r["horizon"],
        r["sign_used"],
        ReasonCode(r["reason"]),
        bool(r["evaluated"]),
        r["created_at"],
        r["note"],
    )


class ResearchStore:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._depth = 0
        self._migrate()

    def close(self) -> None:
        self._db.close()

    def _migrate(self) -> None:
        exists = self._db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()
        if exists is None:
            self._db.executescript((MIGRATIONS / "001_init.sql").read_text())

    @contextmanager
    def tx(self) -> Iterator[None]:
        """One transaction. Re-entrant: nested uses join the outermost one."""
        if self._depth == 0:
            self._db.execute("BEGIN IMMEDIATE")
        self._depth += 1
        try:
            yield
        except BaseException:
            self._depth -= 1
            if self._depth == 0:
                self._db.execute("ROLLBACK")
            raise
        else:
            self._depth -= 1
            if self._depth == 0:
                self._db.execute("COMMIT")

    def _q(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self._db.execute(sql, args).fetchall()

    def _one(self, sql: str, args: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        return self._db.execute(sql, args).fetchone()

    # --- flags --------------------------------------------------------------------------------
    def set_flag(self, key: str, value: str) -> None:
        self._db.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=?",
            (key, value, value),
        )

    def get_flag(self, key: str, default: str | None = None) -> str | None:
        r = self._one("SELECT value FROM meta WHERE key=?", (key,))
        return r["value"] if r else default

    def paused(self) -> bool:
        return self.get_flag("paused") == "1"

    # --- goals --------------------------------------------------------------------------------
    def add_goal(self, goal: ResearchGoal) -> str:
        goal_id = new_id("gol")
        self._db.execute(
            "INSERT INTO goals(goal_id, status, goal_json, created_at) VALUES(?, 'pending', ?, ?)",
            (goal_id, goal.model_dump_json(), now()),
        )
        return goal_id

    def approve_goal(self, goal_id: str) -> None:
        """Mark a goal approved. It activates when no other goal is active (queue by approval)."""
        with self.tx():
            if (
                self._one("SELECT 1 FROM goals WHERE goal_id=? AND status='pending'", (goal_id,))
                is None
            ):
                raise KeyError(f"no pending goal {goal_id}")
            self._db.execute("UPDATE goals SET approved_at=? WHERE goal_id=?", (now(), goal_id))
            self.activate_next_goal()

    def activate_next_goal(self) -> str | None:
        with self.tx():
            if self.active_goal_id() is not None:
                return None
            r = self._one(
                "SELECT goal_id FROM goals WHERE status='pending' AND approved_at IS NOT NULL "
                "ORDER BY approved_at LIMIT 1"
            )
            if r is None:
                return None
            self._db.execute("UPDATE goals SET status='active' WHERE goal_id=?", (r["goal_id"],))
            return str(r["goal_id"])

    def close_goal(
        self, goal_id: str, status: Literal["done", "abandoned"], outcome: str = ""
    ) -> None:
        self._db.execute(
            "UPDATE goals SET status=?, closed_at=?, outcome=? WHERE goal_id=? "
            "AND status IN ('pending','active')",
            (status, now(), outcome, goal_id),
        )

    def active_goal_id(self) -> str | None:
        r = self._one("SELECT goal_id FROM goals WHERE status='active'")
        return str(r["goal_id"]) if r else None

    def get_goal(self, goal_id: str) -> ResearchGoal:
        r = self._one("SELECT goal_json FROM goals WHERE goal_id=?", (goal_id,))
        if r is None:
            raise KeyError(goal_id)
        return ResearchGoal.model_validate_json(r["goal_json"])

    def goals(self) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM goals ORDER BY created_at")

    def goal_acceptances(self, goal_id: str) -> int:
        r = self._one(
            "SELECT COUNT(*) c FROM trials t JOIN batches b USING(batch_id) "
            "WHERE b.goal_id=? AND t.reason='accepted'",
            (goal_id,),
        )
        return int(r["c"]) if r else 0

    # --- batches ------------------------------------------------------------------------------
    def create_batch(
        self, goal_id: str, proposer: str, directive: BatchDirective, batch_id: str | None = None
    ) -> str:
        """Create a PLANNED batch. The id may be chosen first: the directive's RNG is seeded by it."""
        batch_id = batch_id or new_id("bat")
        t = now()
        self._db.execute(
            "INSERT INTO batches(batch_id, goal_id, state, proposer, directive_json, created_at, "
            "updated_at) VALUES(?, ?, 'PLANNED', ?, ?, ?, ?)",
            (batch_id, goal_id, proposer, directive.model_dump_json(), t, t),
        )
        return batch_id

    def set_batch(self, batch_id: str, state: BatchState, **fields: str | None) -> None:
        """Move a batch to a state, optionally storing packet_json, response_json, llm_call_id, error."""
        allowed = {"packet_json", "response_json", "llm_call_id", "error"}
        if not set(fields) <= allowed:
            raise ValueError(f"unknown batch fields {set(fields) - allowed}")
        cols = ", ".join(f"{k}=?" for k in fields)
        sql = f"UPDATE batches SET state=?, updated_at=?{', ' + cols if cols else ''} WHERE batch_id=?"
        self._db.execute(sql, (state, now(), *fields.values(), batch_id))

    def get_batch(self, batch_id: str) -> sqlite3.Row:
        r = self._one("SELECT * FROM batches WHERE batch_id=?", (batch_id,))
        if r is None:
            raise KeyError(batch_id)
        return r

    def unfinished_batches(self) -> list[sqlite3.Row]:
        return self._q(
            "SELECT * FROM batches WHERE state NOT IN ('DONE','FAILED') ORDER BY created_at"
        )

    def batches_today(self) -> int:
        day = datetime.now(UTC).date().isoformat()
        return int(self._q("SELECT COUNT(*) c FROM batches WHERE created_at >= ?", (day,))[0]["c"])

    def goal_batches(self, goal_id: str) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM batches WHERE goal_id=? ORDER BY created_at", (goal_id,))

    def batch_accepted(self, batch_id: str) -> int:
        r = self._one(
            "SELECT COUNT(*) c FROM trials WHERE batch_id=? AND reason='accepted'", (batch_id,)
        )
        return int(r["c"]) if r else 0

    # --- proposals and llm calls --------------------------------------------------------------
    def add_proposals(self, records: list[ProposalRecord]) -> None:
        with self.tx():
            for p in records:
                self._db.execute(
                    "INSERT OR IGNORE INTO proposals(proposal_id, batch_id, goal_id, expr_hash, "
                    "family, record_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (
                        p.proposal_id,
                        p.batch_id,
                        p.goal_id,
                        p.expr_hash,
                        p.proposal.family,
                        p.model_dump_json(),
                        now(),
                    ),
                )
                self.ensure_family(p.proposal.family, "llm" if p.llm_call_id else "seed")

    def get_proposal(self, proposal_id: str) -> ProposalRecord:
        r = self._one("SELECT record_json FROM proposals WHERE proposal_id=?", (proposal_id,))
        if r is None:
            raise KeyError(proposal_id)
        return ProposalRecord.model_validate_json(r["record_json"])

    def batch_proposals(self, batch_id: str) -> list[ProposalRecord]:
        rows = self._q(
            "SELECT record_json FROM proposals WHERE batch_id=? ORDER BY rowid", (batch_id,)
        )
        return [ProposalRecord.model_validate_json(r["record_json"]) for r in rows]

    def add_llm_call(self, rec: LLMCallRecord) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO llm_calls(call_id, role, model, tokens_in, tokens_out, cost_usd, "
            "ok, record_json, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rec.call_id,
                rec.role,
                rec.model,
                rec.tokens_in,
                rec.tokens_out,
                rec.cost_usd,
                int(rec.ok),
                rec.model_dump_json(),
                rec.created_at.isoformat(),
            ),
        )

    def proposed_hashes(self) -> set[str]:
        """Expression hashes of every proposal ever stored, so proposers can avoid repeats."""
        return {r["expr_hash"] for r in self._q("SELECT DISTINCT expr_hash FROM proposals")}

    def spend_today(self) -> float:
        day = datetime.now(UTC).date().isoformat()
        r = self._one(
            "SELECT COALESCE(SUM(cost_usd), 0) s FROM llm_calls WHERE created_at >= ?", (day,)
        )
        return float(r["s"]) if r else 0.0

    def llm_spend_by_role(self) -> dict[str, float]:
        rows = self._q("SELECT role, SUM(cost_usd) s FROM llm_calls GROUP BY role")
        return {r["role"]: float(r["s"]) for r in rows}

    # --- trials -------------------------------------------------------------------------------
    def record_trial(
        self,
        *,
        source: TrialSource,
        sandbox: SandboxSpec,
        manifest: DataManifest,
        reason: ReasonCode,
        proposal: ProposalRecord | None = None,
        result: EvalResult | None = None,
        sign_used: int | None = None,
        batch_id: str | None = None,
        family: str | None = None,
        horizon: str | None = None,
        expr_hash: str | None = None,
        n_at_eval: int | None = None,
        holdout_ok: bool | None = None,
        code_commit: str = "unknown",
        note: str | None = None,
    ) -> str:
        """The only way to create a trial. One transaction: insert the trial and its eval result,
        and bump the goal counter. Returns the trial id; an evaluation already recorded under the
        same (sandbox, expression, horizon, sign) returns the existing id and writes nothing."""
        evaluated = reason not in NOT_EVALUATED
        if proposal is not None:
            expr_hash = expr_hash or proposal.expr_hash  # expression + condition: the dedupe key
        if result is not None:
            expr_hash = expr_hash or result.expr_hash
            horizon = horizon or result.horizon
            sign_used = sign_used if sign_used is not None else result.sign_used
            code_commit = result.code_commit
        if proposal is not None:
            batch_id = batch_id or proposal.batch_id
            family = family or proposal.proposal.family
            horizon = horizon or proposal.proposal.horizon
        trial_id = result.trial_id if result is not None else new_id("trl")
        sandbox_hash = sandbox.content_hash()
        with self.tx():
            if evaluated:
                dup = self._one(
                    "SELECT trial_id FROM trials WHERE evaluated=1 AND sandbox_hash=? AND expr_hash=? "
                    "AND horizon=? AND sign_used=?",
                    (sandbox_hash, expr_hash, horizon, sign_used),
                )
                if dup is not None:
                    return str(dup["trial_id"])
            self._db.execute(
                "INSERT INTO trials(trial_id, batch_id, proposal_id, source, expr_hash, family, "
                "horizon, sign_used, sandbox_hash, dataset_hash, code_commit, reason, evaluated, "
                "note, created_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    trial_id,
                    batch_id,
                    proposal.proposal_id if proposal else None,
                    source,
                    expr_hash,
                    family,
                    horizon,
                    sign_used,
                    sandbox_hash,
                    manifest.dataset_hash,
                    code_commit,
                    reason.value,
                    int(evaluated),
                    note,
                    now(),
                ),
            )
            if result is not None:
                self._db.execute(
                    "INSERT INTO eval_results(trial_id, result_json, n_at_eval, ic_tstat, holdout_ok) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        trial_id,
                        result.model_dump_json(),
                        n_at_eval or 0,
                        result.ic_tstat,
                        None if holdout_ok is None else int(holdout_ok),
                    ),
                )
            if family:
                self.ensure_family(family, "llm" if source == "llm" else "seed")
            if batch_id:
                self._db.execute(
                    "UPDATE goals SET trials_used = trials_used + 1 WHERE goal_id = "
                    "(SELECT goal_id FROM batches WHERE batch_id=?)",
                    (batch_id,),
                )
        return trial_id

    def n_evaluated(self, sandbox_hash: str) -> int:
        """N for the deflation formula: trials that reached the IC check under this sandbox."""
        r = self._one(
            "SELECT COUNT(*) c FROM trials WHERE sandbox_hash=? AND evaluated=1", (sandbox_hash,)
        )
        return int(r["c"]) if r else 0

    def seen(self, sandbox_hash: str, expr_hash: str, horizon: str, sign: int) -> bool:
        return (
            self._one(
                "SELECT 1 FROM trials WHERE evaluated=1 AND sandbox_hash=? AND expr_hash=? "
                "AND horizon=? AND sign_used=?",
                (sandbox_hash, expr_hash, horizon, sign),
            )
            is not None
        )

    def trial_for_hash(self, expr_hash: str, horizon: str) -> str | None:
        r = self._one(
            "SELECT trial_id FROM trials WHERE expr_hash=? AND horizon=? AND evaluated=1 "
            "ORDER BY rowid LIMIT 1",
            (expr_hash, horizon),
        )
        return str(r["trial_id"]) if r else None

    def get_trial(self, trial_id: str) -> TrialRow:
        r = self._one("SELECT * FROM trials WHERE trial_id=?", (trial_id,))
        if r is None:
            raise KeyError(trial_id)
        return _trial(r)

    def get_eval(self, trial_id: str) -> tuple[EvalResult, int, bool | None] | None:
        """(result, N used, holdout passed or None) for a trial, if it was evaluated."""
        r = self._one("SELECT * FROM eval_results WHERE trial_id=?", (trial_id,))
        if r is None:
            return None
        ok = None if r["holdout_ok"] is None else bool(r["holdout_ok"])
        return EvalResult.model_validate_json(r["result_json"]), int(r["n_at_eval"]), ok

    def batch_trials(self, batch_id: str) -> list[TrialRow]:
        return [
            _trial(r)
            for r in self._q("SELECT * FROM trials WHERE batch_id=? ORDER BY rowid", (batch_id,))
        ]

    def goal_trials(self, goal_id: str, limit: int | None = None) -> list[TrialRow]:
        """Newest first."""
        sql = (
            "SELECT t.* FROM trials t JOIN batches b USING(batch_id) WHERE b.goal_id=? "
            "ORDER BY t.rowid DESC"
        )
        rows = self._q(sql + (f" LIMIT {int(limit)}" if limit else ""), (goal_id,))
        return [_trial(r) for r in rows]

    def trials_filtered(
        self,
        goal_id: str | None = None,
        reason: str | None = None,
        family: str | None = None,
        limit: int = 50,
    ) -> list[TrialRow]:
        sql, args = "SELECT t.* FROM trials t LEFT JOIN batches b USING(batch_id) WHERE 1=1", []
        if goal_id:
            sql += " AND b.goal_id=?"
            args.append(goal_id)
        if reason:
            sql += " AND t.reason=?"
            args.append(reason)
        if family:
            sql += " AND t.family=?"
            args.append(family)
        sql += f" ORDER BY t.rowid DESC LIMIT {int(limit)}"
        return [_trial(r) for r in self._q(sql, tuple(args))]

    def proposer_trials(self, proposer: str) -> list[TrialRow]:
        """Evaluated trials (sign flips included) from batches run by `proposer`, oldest first."""
        rows = self._q(
            "SELECT t.* FROM trials t JOIN batches b USING(batch_id) "
            "WHERE b.proposer=? AND t.evaluated=1 ORDER BY t.rowid",
            (proposer,),
        )
        return [_trial(r) for r in rows]

    def lib_factors_for_trials(self, trial_ids: list[str]) -> list[LibFactor]:
        out: list[LibFactor] = []
        for tid in trial_ids:
            t = self.get_trial(tid)
            if t.proposal_id is None:
                continue
            prop = self.get_proposal(t.proposal_id)
            out.append(
                LibFactor(
                    factor_id=tid,
                    canonical_expr=prop.canonical_expr,
                    sign=1 if (t.sign_used or 1) > 0 else -1,
                    horizon=prop.proposal.horizon,
                )
            )
        return out

    def reason_counts(self, goal_id: str | None = None) -> dict[str, int]:
        if goal_id:
            rows = self._q(
                "SELECT t.reason, COUNT(*) c FROM trials t JOIN batches b USING(batch_id) "
                "WHERE b.goal_id=? GROUP BY t.reason",
                (goal_id,),
            )
        else:
            rows = self._q("SELECT reason, COUNT(*) c FROM trials GROUP BY reason")
        return {r["reason"]: int(r["c"]) for r in rows}

    def reason_counts_since(self, iso: str) -> dict[str, int]:
        rows = self._q(
            "SELECT reason, COUNT(*) c FROM trials WHERE created_at >= ? GROUP BY reason", (iso,)
        )
        return {r["reason"]: int(r["c"]) for r in rows}

    def spend_since(self, iso: str) -> float:
        r = self._one(
            "SELECT COALESCE(SUM(cost_usd), 0) s FROM llm_calls WHERE created_at >= ?", (iso,)
        )
        return float(r["s"]) if r else 0.0

    def total_trials(self) -> int:
        return int(self._q("SELECT COUNT(*) c FROM trials")[0]["c"])

    # --- families -----------------------------------------------------------------------------
    def ensure_family(self, family: str, created_by: Literal["seed", "llm", "manual"]) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO families(family, status, created_by, created_at) "
            "VALUES(?, 'active', ?, ?)",
            (family, created_by, now()),
        )

    def families(
        self, active_only: bool = True, include: list[str] | tuple[str, ...] = ()
    ) -> list[str]:
        """Bandit arms: the goal's families (created on demand), or every active family."""
        for f in include:
            self.ensure_family(f, "seed")
        rows = self._q("SELECT family, status FROM families ORDER BY family")
        names = [r["family"] for r in rows if not active_only or r["status"] == "active"]
        return [f for f in names if f in include] if include else names

    def families_with_status(self, *statuses: str) -> tuple[str, ...]:
        marks = ",".join("?" for _ in statuses)
        rows = self._q(
            f"SELECT family FROM families WHERE status IN ({marks}) ORDER BY family", statuses
        )
        return tuple(r["family"] for r in rows)

    def set_family_status(
        self, family: str, status: Literal["active", "exhausted", "refuted"]
    ) -> None:
        self._db.execute("UPDATE families SET status=? WHERE family=?", (status, family))

    def family_stats(self, family: str, last_evaluated: int) -> tuple[int, int]:
        """(accepted, evaluated) over the family's most recent `last_evaluated` evaluated trials."""
        rows = self._q(
            "SELECT reason FROM trials WHERE family=? AND evaluated=1 ORDER BY rowid DESC LIMIT ?",
            (family, last_evaluated),
        )
        return sum(r["reason"] == "accepted" for r in rows), len(rows)

    def family_totals(self, family: str) -> tuple[int, int, int]:
        """(evaluated, accepted ever, evaluated since the last acceptance)."""
        rows = self._q(
            "SELECT reason FROM trials WHERE family=? AND evaluated=1 ORDER BY rowid", (family,)
        )
        reasons = [r["reason"] for r in rows]
        last = max((i for i, x in enumerate(reasons) if x == "accepted"), default=-1)
        return len(reasons), sum(x == "accepted" for x in reasons), len(reasons) - 1 - last

    # --- pool ---------------------------------------------------------------------------------
    def add_to_pool(self, trial_id: str, factor_id: str | None = None) -> str:
        t = self.get_trial(trial_id)
        factor_id = factor_id or new_id("fac")
        self._db.execute(
            "INSERT INTO pool(factor_id, trial_id, expr_hash, family, horizon, sign_used, added_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (factor_id, trial_id, t.expr_hash, t.family, t.horizon, t.sign_used, now()),
        )
        return factor_id

    def remove_from_pool(self, factor_id: str, reason: str) -> None:
        self._db.execute(
            "UPDATE pool SET removed_at=?, removed_reason=? WHERE factor_id=? AND removed_at IS NULL",
            (now(), reason, factor_id),
        )

    def pool(self, at: str | None = None) -> list[PoolRow]:
        """The active pool, or the pool as it stood at an ISO timestamp (for re-evaluation)."""
        cond = (
            "p.removed_at IS NULL"
            if at is None
            else "p.added_at <= ? AND (p.removed_at IS NULL OR p.removed_at > ?)"
        )
        args: tuple[str, ...] = () if at is None else (at, at)
        rows = self._q(
            "SELECT p.*, pr.record_json, e.ic_tstat, e.result_json FROM pool p "
            "JOIN trials t ON t.trial_id = p.trial_id "
            "JOIN proposals pr ON pr.proposal_id = t.proposal_id "
            "JOIN eval_results e ON e.trial_id = p.trial_id "
            f"WHERE {cond} ORDER BY p.added_at, p.factor_id",
            args,
        )
        out: list[PoolRow] = []
        for r in rows:
            prop = ProposalRecord.model_validate_json(r["record_json"])
            res = EvalResult.model_validate_json(r["result_json"])
            out.append(
                PoolRow(
                    r["factor_id"],
                    r["trial_id"],
                    r["expr_hash"],
                    r["family"],
                    cast(Horizon, r["horizon"]),
                    r["sign_used"],
                    r["added_at"],
                    prop.canonical_expr,
                    r["ic_tstat"],
                    res.ic_mean,
                )
            )
        return out

    def pool_size(self, horizons: Sequence[str] | None = None) -> int:
        rows = self.pool()
        return sum(1 for r in rows if not horizons or r.horizon in horizons)

    def pool_added_since(self, since: str) -> int:
        r = self._one(
            "SELECT COUNT(*) c FROM pool WHERE added_at > ? OR removed_at > ?", (since, since)
        )
        return int(r["c"]) if r else 0

    # --- beliefs ------------------------------------------------------------------------------
    def create_belief(
        self,
        *,
        statement: str,
        family: str,
        horizon: str,
        next_test: str | None,
        supersedes: str | None = None,
    ) -> str:
        belief_id, t = new_id("blf"), now()
        self._db.execute(
            "INSERT INTO beliefs(belief_id, family, horizon, statement, next_test, supersedes, "
            "created_at, updated_at) VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (belief_id, family, horizon, statement, next_test, supersedes, t, t),
        )
        return belief_id

    def add_evidence(
        self,
        belief_id: str,
        trial_id: str,
        direction: Literal["support", "contradict"],
        batch_id: str,
    ) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO belief_evidence(belief_id, trial_id, direction, batch_id) "
            "VALUES(?, ?, ?, ?)",
            (belief_id, trial_id, direction, batch_id),
        )
        self._db.execute("UPDATE beliefs SET updated_at=? WHERE belief_id=?", (now(), belief_id))

    def update_next_test(self, belief_id: str, next_test: str) -> None:
        self._db.execute(
            "UPDATE beliefs SET next_test=?, updated_at=? WHERE belief_id=?",
            (next_test, now(), belief_id),
        )

    def retire_belief(self, belief_id: str) -> None:
        self._db.execute(
            "UPDATE beliefs SET retired=1, updated_at=? WHERE belief_id=?", (now(), belief_id)
        )

    def beliefs(
        self, families: list[str] | None = None, include_retired: bool = False
    ) -> list[Belief]:
        """Beliefs with counts derived from the evidence table (never incremented)."""
        rows = self._q(
            "SELECT b.*, "
            "(SELECT COUNT(*) FROM belief_evidence e WHERE e.belief_id=b.belief_id AND "
            " e.direction='support') ns, "
            "(SELECT COUNT(*) FROM belief_evidence e WHERE e.belief_id=b.belief_id AND "
            " e.direction='contradict') nc FROM beliefs b ORDER BY b.created_at DESC, b.rowid DESC"
        )
        out = []
        for r in rows:
            if r["retired"] and not include_retired:
                continue
            if families is not None and r["family"] not in families:
                continue
            out.append(
                Belief(
                    belief_id=r["belief_id"],
                    statement=r["statement"],
                    family=r["family"],
                    horizon=r["horizon"],
                    n_support=r["ns"],
                    n_contradict=r["nc"],
                    supersedes=r["supersedes"],
                    next_test=r["next_test"],
                    retired=bool(r["retired"]),
                    created_at=datetime.fromisoformat(r["created_at"]),
                    updated_at=datetime.fromisoformat(r["updated_at"]),
                )
            )
        return out

    def belief_ids(self) -> set[str]:
        return {r["belief_id"] for r in self._q("SELECT belief_id FROM beliefs WHERE retired=0")}

    def evidence_for(self, belief_id: str) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM belief_evidence WHERE belief_id=?", (belief_id,))

    def save_reflection(self, batch_id: str, record_json: str) -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO reflections(batch_id, record_json, created_at) VALUES(?, ?, ?)",
            (batch_id, record_json, now()),
        )

    # --- libraries and final evals ------------------------------------------------------------
    def add_library(self, lib: FrozenLibrary, report_json: str | None = None) -> None:
        self._db.execute(
            "INSERT INTO factor_libs(lib_version, content_hash, lib_json, report_json, frozen_at) "
            "VALUES(?, ?, ?, ?, ?)",
            (
                lib.lib_version,
                lib.content_hash(),
                lib.model_dump_json(),
                report_json,
                lib.frozen_at.isoformat(),
            ),
        )

    def get_library(self, lib_version: str) -> FrozenLibrary:
        r = self._one("SELECT lib_json FROM factor_libs WHERE lib_version=?", (lib_version,))
        if r is None:
            raise KeyError(lib_version)
        return FrozenLibrary.model_validate_json(r["lib_json"])

    def libraries(self) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM factor_libs ORDER BY frozen_at")

    def next_lib_version(self) -> str:
        return f"lib_v{len(self.libraries()) + 1}"

    def record_final_eval(self, lib_version: str, report: FinalEvalReport) -> None:
        self._db.execute(
            "INSERT INTO final_evals(request_hash, lib_version, report_json, imported_at) "
            "VALUES(?, ?, ?, ?)",
            (report.request_hash, lib_version, report.model_dump_json(), now()),
        )

    def final_evals(self) -> list[sqlite3.Row]:
        return self._q("SELECT * FROM final_evals ORDER BY imported_at")
