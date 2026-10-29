"""The orchestrator: one batch at a time through a fixed sequence of states.

PLANNED -> PROPOSING -> VALIDATING -> EVALUATING -> GATING -> REFLECTING -> DONE (or FAILED).
Every transition commits in one transaction together with the writes of that state, so a crash
or reboot resumes at the stored state with nothing lost or duplicated. Re-running a state is
safe: recording the same evaluation twice is a no-op, and a rolled-back state left no trace.
"""

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from zoneinfo import ZoneInfo

import numpy as np
import structlog
from pydantic import ValidationError

from alpha_factory.agents.proposer import Proposer
from alpha_factory.compute.specs import EvalBatchRequest, EvalSpec, PoolLooRequest
from alpha_factory.compute.worker import EvalBackend
from alpha_factory.contracts import (
    BatchDirective,
    DataManifest,
    FactorProposal,
    LibFactor,
    ProposalRecord,
    RawProposalBatch,
    ReasonCode,
    SandboxSpec,
)
from alpha_factory.dsl import DSLError, compile_expr
from alpha_factory.eval.gate import required_tstat
from alpha_factory.gitinfo import git_commit
from alpha_factory.ids import new_id
from alpha_factory.research.bandit import make_directive, seed_for, update_family_status
from alpha_factory.research.config import LoopConfig
from alpha_factory.research.notify import (
    Acceptance,
    DailyDigest,
    FreezeSuggested,
    GoalEnded,
    LoopPaused,
    NotifyEvent,
)
from alpha_factory.research.store import ResearchStore, now

log = structlog.get_logger(__name__)
SLUG = re.compile(r"^[a-z][a-z0-9_]{2,39}$")
HORIZONS = ("h1", "h2", "h4")


class EvaluationError(RuntimeError):
    """The worker could not evaluate one or more specs."""


class BudgetExceededError(RuntimeError):
    """The daily LLM budget is used up."""


class Reflector(Protocol):
    def reflect(self, batch_id: str) -> None: ...


@dataclass
class Orchestrator:
    store: ResearchStore
    cfg: LoopConfig
    backend: EvalBackend
    sandbox: SandboxSpec
    manifest: DataManifest
    proposers: dict[str, Proposer]
    notify: Callable[[NotifyEvent], None] = lambda _e: None
    reflector: Reflector | None = None
    fault: Callable[[str], None] = lambda _point: None  # tests inject crashes here
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    digest_hour: int = 20  # local time, Europe/Bucharest

    # ---- planning -------------------------------------------------------------------------
    def plan_batch(self, goal_id: str) -> str:
        batch_id = new_id("bat")
        rng = np.random.default_rng(seed_for(batch_id))
        goal = self.store.get_goal(goal_id)
        directive = make_directive(goal, self.store, self.cfg, rng)
        self.store.create_batch(goal_id, self.cfg.proposer, directive, batch_id=batch_id)
        return batch_id

    # ---- one state transition ----------------------------------------------------------------
    def advance(self, batch_id: str) -> str:
        """Perform the current state's work and move to the next state. Returns the new state."""
        state = self.store.get_batch(batch_id)["state"]
        self.fault(f"before:{state}")
        match state:
            case "PLANNED":
                self.store.set_batch(batch_id, "PROPOSING")
            case "PROPOSING":
                self._propose(batch_id)
            case "VALIDATING":
                self._validate(batch_id)
            case "EVALUATING":
                self._evaluate(batch_id)
            case "GATING":
                self._gate(batch_id)
            case "REFLECTING":
                self._reflect(batch_id)
            case _:
                return str(state)
        self.fault(f"after:{state}")
        return str(self.store.get_batch(batch_id)["state"])

    def run_batch(self, batch_id: str) -> str:
        while True:
            state = str(self.store.get_batch(batch_id)["state"])
            if state in ("DONE", "FAILED"):
                break
            if self.store.paused():
                return state
            self.advance(batch_id)
        self._after_batch(batch_id)
        return str(self.store.get_batch(batch_id)["state"])

    # ---- PROPOSING -----------------------------------------------------------------------------
    def _propose(self, batch_id: str) -> None:
        b = self.store.get_batch(batch_id)
        if b["response_json"]:  # a saved response is never requested again
            self.store.set_batch(batch_id, "VALIDATING")
            return
        if b["proposer"] == "llm" and self.store.spend_today() >= self.cfg.daily_llm_budget_usd:
            raise BudgetExceededError(
                f"daily LLM budget ${self.cfg.daily_llm_budget_usd:.2f} reached"
            )
        goal = self.store.get_goal(b["goal_id"])
        directive = BatchDirective.model_validate_json(b["directive_json"])
        out = self.proposers[b["proposer"]].propose(directive, goal, self.store, seed_for(batch_id))
        with self.store.tx():
            for call in out.llm_calls:
                self.store.add_llm_call(call)
            call_id = out.llm_calls[0].call_id if out.llm_calls else None
            if out.raw is None:
                self.store.set_batch(
                    batch_id,
                    "FAILED",
                    error=out.failure or "no proposals",
                    packet_json=out.packet_json,
                    response_json=out.response_text,
                    llm_call_id=call_id,
                )
            else:
                self.store.set_batch(
                    batch_id,
                    "VALIDATING",
                    response_json=out.raw.model_dump_json(),
                    packet_json=out.packet_json,
                    llm_call_id=call_id,
                )

    # ---- VALIDATING ----------------------------------------------------------------------------
    def _validate(self, batch_id: str) -> None:
        b = self.store.get_batch(batch_id)
        raw = RawProposalBatch.model_validate_json(b["response_json"])
        goal_id = b["goal_id"]
        source = "llm" if b["proposer"] == "llm" else "template"
        pool_ids = {p.factor_id for p in self.store.pool()}
        sb_hash = self.sandbox.content_hash()
        records: list[ProposalRecord] = []
        invalid: list[tuple[ReasonCode, dict[str, object], str, str | None, str | None]] = []
        in_batch: set[tuple[str, str, int]] = set()
        for item in raw.proposals:
            fam = item.get("family") if isinstance(item.get("family"), str) else None
            fam = fam if fam and SLUG.match(fam) else None
            hor = item.get("horizon") if item.get("horizon") in HORIZONS else None
            try:
                fp = FactorProposal.model_validate(item, context={"pool_ids": pool_ids})
            except ValidationError as e:
                invalid.append((ReasonCode.INVALID_SCHEMA, item, e.errors()[0]["msg"], fam, hor))
                continue
            try:
                compiled = compile_expr(fp.expression, condition=fp.condition)
            except DSLError as e:
                invalid.append((e.reason, item, e.message, fam, hor))
                continue
            sign = 1 if fp.expected_sign == "positive" else -1
            key = (compiled.full_hash, fp.horizon, sign)
            earlier = self.store.trial_for_hash(compiled.full_hash, fp.horizon)
            if key in in_batch or self.store.seen(sb_hash, compiled.full_hash, fp.horizon, sign):
                invalid.append(
                    (
                        ReasonCode.DUPLICATE,
                        item,
                        f"same canonical expression as {earlier or 'an earlier proposal in this batch'}",
                        fam,
                        hor,
                    )
                )
                continue
            in_batch.add(key)
            records.append(
                ProposalRecord(
                    proposal_id=new_id("prp"),
                    batch_id=batch_id,
                    goal_id=goal_id,
                    llm_call_id=b["llm_call_id"],
                    proposal=fp,
                    canonical_expr=compiled.expression,
                    expr_hash=compiled.full_hash,
                    ast_depth=compiled.ast_depth,
                    windows_used=compiled.windows_used,
                    created_at=_utcnow(),
                )
            )
        with self.store.tx():
            self.store.add_proposals(records)
            for reason, item, msg, fam, hor in invalid:
                self.store.record_trial(
                    source=source,
                    sandbox=self.sandbox,
                    manifest=self.manifest,
                    reason=reason,
                    batch_id=batch_id,
                    family=fam,
                    horizon=hor,
                    code_commit=git_commit(),
                    note=json.dumps({"expression": item.get("expression"), "message": msg}),
                )
            self.store.set_batch(batch_id, "EVALUATING")

    # ---- EVALUATING ----------------------------------------------------------------------------
    def _spec(self, p: ProposalRecord, sign: int, pool_exprs: dict[str, str]) -> EvalSpec:
        return EvalSpec(
            trial_key=f"{p.proposal_id}|{p.proposal.horizon}|{sign}",
            canonical_expr=p.canonical_expr,
            condition=p.proposal.condition,
            horizon=p.proposal.horizon,
            sign=1 if sign > 0 else -1,
            parent_exprs=tuple(pool_exprs[x] for x in p.proposal.parents if x in pool_exprs),
            refutations=tuple(p.proposal.refutations),
        )

    def _request(self, specs: list[EvalSpec], n_evaluated: int) -> EvalBatchRequest:
        return EvalBatchRequest(
            sandbox_hash=self.sandbox.content_hash(),
            dataset_hash=self.manifest.dataset_hash,
            code_commit=git_commit(),
            n_evaluated=n_evaluated,
            pool=tuple(p.lib_factor() for p in self.store.pool()),
            specs=tuple(specs),
        )

    def _evaluate(self, batch_id: str) -> None:
        b = self.store.get_batch(batch_id)
        source = "llm" if b["proposer"] == "llm" else "template"
        records = {p.proposal_id: p for p in self.store.batch_proposals(batch_id)}
        pool_exprs = {p.factor_id: p.canonical_expr for p in self.store.pool()}
        n0 = self.store.n_evaluated(self.sandbox.content_hash())
        specs = [
            self._spec(p, 1 if p.proposal.expected_sign == "positive" else -1, pool_exprs)
            for p in records.values()
        ]
        outcomes = self.backend.evaluate(self._request(specs, n0)).outcomes if specs else ()
        self.fault("evaluated")
        errors = [o for o in outcomes if o.error]
        if errors:
            raise EvaluationError("; ".join(f"{o.trial_key}: {o.error}" for o in errors))
        flips: list[EvalSpec] = []
        for o in outcomes:
            if o.result is not None and o.result.reason == ReasonCode.WRONG_SIGN:
                p = records[o.trial_key.split("|")[0]]
                flips.append(self._spec(p, -o.result.sign_used, pool_exprs))
        flip_out = (
            self.backend.evaluate(self._request(flips, n0 + len(specs))).outcomes if flips else ()
        )
        errors = [o for o in flip_out if o.error]
        if errors:
            raise EvaluationError("; ".join(f"{o.trial_key}: {o.error}" for o in errors))
        with self.store.tx():
            for kind, group in (("main", outcomes), ("flip", flip_out)):
                for o in group:
                    assert o.result is not None
                    p = records[o.trial_key.split("|")[0]]
                    self.store.record_trial(
                        source="sign_flip" if kind == "flip" else source,
                        sandbox=self.sandbox,
                        manifest=self.manifest,
                        reason=o.result.reason,
                        proposal=p,
                        result=o.result,
                        n_at_eval=o.n_at_eval,
                        holdout_ok=o.holdout_ok,
                    )
            self.fault("recorded")
            self.store.set_batch(batch_id, "GATING")

    # ---- GATING --------------------------------------------------------------------------------
    def _gate(self, batch_id: str) -> None:
        store, cfg = self.store, self.cfg
        trials = store.batch_trials(batch_id)
        pool = {p.factor_id: p for p in store.pool()}
        remove: dict[str, str] = {}
        add: list[tuple[str, str]] = []  # (trial_id, new factor_id)
        for t in trials:
            if not t.evaluated or t.proposal_id is None:
                continue
            ev = store.get_eval(t.trial_id)
            if ev is None:
                continue
            res, _, holdout_ok = ev
            if t.reason == ReasonCode.ACCEPTED and holdout_ok is not False:
                add.append((t.trial_id, new_id("fac")))
            elif t.reason == ReasonCode.CORRELATED and not res.flags and holdout_ok:
                old_id = res.max_corr_pool_id
                old = pool.get(old_id) if old_id else None
                if (
                    old
                    and old_id not in remove
                    and res.ic_tstat >= cfg.replace_t_ratio * old.ic_tstat
                ):
                    fid = new_id("fac")
                    remove[old_id] = f"replaced by {fid}"  # type: ignore[index]
                    add.append((t.trial_id, fid))
        # the pool after these changes, and the cap
        after: dict[str, LibFactor] = {
            k: v.lib_factor() for k, v in pool.items() if k not in remove
        }
        for trial_id, fid in add:
            tr = store.get_trial(trial_id)
            prop = store.get_proposal(tr.proposal_id)  # type: ignore[arg-type]
            after[fid] = LibFactor(
                factor_id=fid,
                canonical_expr=prop.canonical_expr,
                sign=1 if (tr.sign_used or 1) > 0 else -1,
                horizon=prop.proposal.horizon,
            )
        while len(after) > cfg.pool_cap:
            loo = self.backend.pool_loo(
                PoolLooRequest(
                    sandbox_hash=self.sandbox.content_hash(),
                    dataset_hash=self.manifest.dataset_hash,
                    pool=tuple(after.values()),
                )
            ).combined_ic_without
            worst = max(loo, key=lambda k: (loo[k], k))  # removal hurts the combined IC least
            after.pop(worst)
            if worst in pool:
                remove[worst] = "cap"
            else:  # a factor admitted in this very step: just do not add it
                add = [(t, f) for t, f in add if f != worst]
        with store.tx():
            for fid, why in remove.items():
                store.remove_from_pool(fid, why)
            for trial_id, fid in add:
                store.add_to_pool(trial_id, fid)
            for fam in {t.family for t in trials if t.family}:
                update_family_status(store, fam)
            store.set_batch(batch_id, "REFLECTING")
        self._announce(batch_id, [t for t, _ in add])

    def _announce(self, batch_id: str, accepted_trials: list[str]) -> None:
        for trial_id in accepted_trials:
            res, n, _ = self.store.get_eval(trial_id) or (None, 0, None)
            if res is None:
                continue
            fid = next((p.factor_id for p in self.store.pool() if p.trial_id == trial_id), trial_id)
            tr = self.store.get_trial(trial_id)
            self.notify(
                Acceptance(
                    factor_id=fid,
                    family=tr.family or "",
                    horizon=res.horizon,
                    tstat=res.ic_tstat,
                    t_required=required_tstat(n, self.sandbox.gate),
                    worst_fold=res.ic_worst_fold,
                    net_sharpe=res.net_sharpe_val,
                    expression=self.store.get_proposal(tr.proposal_id).canonical_expr,  # type: ignore[arg-type]
                )
            )
        last = self.store.libraries()
        since = last[-1]["frozen_at"] if last else "1970-01-01"
        changed = self.store.pool_added_since(since)
        flag = f"freeze_notified_{len(last)}"
        if changed >= self.cfg.freeze_suggestion_delta and self.store.get_flag(flag) is None:
            self.store.set_flag(flag, "1")
            self.notify(
                FreezeSuggested(
                    factors_changed=changed, last_library=last[-1]["lib_version"] if last else None
                )
            )

    # ---- REFLECTING ----------------------------------------------------------------------------
    def _reflect(self, batch_id: str) -> None:
        b = self.store.get_batch(batch_id)
        if b["proposer"] == "llm" and self.reflector is not None:
            self.reflector.reflect(batch_id)  # applies belief updates in its own transaction
        self.store.set_batch(batch_id, "DONE")

    # ---- goals ---------------------------------------------------------------------------------
    def _after_batch(self, batch_id: str) -> None:
        self._check_goal(str(self.store.get_batch(batch_id)["goal_id"]))

    def _check_goal(self, goal_id: str) -> None:
        """End the goal if its budget is used or it has stalled. Safe to call at any time."""
        goal = self.store.get_goal(goal_id)
        row = next(g for g in self.store.goals() if g["goal_id"] == goal_id)
        if row["status"] != "active":
            return
        batches = [x for x in self.store.goal_batches(goal_id) if x["state"] == "DONE"]
        k = self.cfg.goal.stop_after_batches_without_accept
        why = None
        if row["trials_used"] >= goal.budget_trials:
            why = "trial budget used"
        elif len(batches) >= k and not any(
            self.store.batch_accepted(x["batch_id"]) for x in batches[-k:]
        ):
            why = f"{k} batches without an acceptance"
        if why:
            accepted = self.store.goal_acceptances(goal_id)
            self.store.close_goal(goal_id, "done", why)
            self.store.activate_next_goal()
            self.notify(
                GoalEnded(
                    title=goal.title,
                    why=why,
                    trials_used=int(row["trials_used"]),
                    acceptances=accepted,
                    beliefs_changed=0,
                )
            )

    # ---- the long-running loop -------------------------------------------------------------------
    def run_once(self) -> bool:
        """Resume an unfinished batch or run a new one. False when there is nothing to do."""
        if self.store.paused():
            return False
        unfinished = self.store.unfinished_batches()
        if unfinished:
            self.run_batch(str(unfinished[0]["batch_id"]))
            return True
        goal_id = self.store.active_goal_id() or self.store.activate_next_goal()
        if goal_id is not None:
            self._check_goal(goal_id)  # a crash after DONE must not let one more batch start
            goal_id = self.store.active_goal_id() or self.store.activate_next_goal()
        if goal_id is None or self.store.batches_today() >= self.cfg.max_batches_per_day:
            return False
        self.run_batch(self.plan_batch(goal_id))
        return True

    def run_forever(
        self, should_stop: Callable[[], bool] = lambda: False, idle_sleep_s: float = 30.0
    ) -> None:
        failures = 0
        while not should_stop():
            try:
                self.maybe_digest()
                progressed = self.run_once()
                failures = 0
            except BudgetExceededError as e:
                self.pause(str(e))
                continue
            except Exception as e:
                failures += 1
                log.error("batch step failed", error=repr(e), consecutive=failures)
                if failures >= self.cfg.max_consecutive_failures:
                    self.pause(f"{failures} consecutive failures: {e!r}")
                    failures = 0
                time.sleep(min(idle_sleep_s, 5.0))
                continue
            if not progressed:
                time.sleep(idle_sleep_s)

    def maybe_digest(self) -> bool:
        """Send the daily digest once a day, after 20:00 Bucharest time. True if one was sent."""
        local = self.clock().astimezone(ZoneInfo("Europe/Bucharest"))
        day = local.date().isoformat()
        if local.hour < self.digest_hour or self.store.get_flag("digest_sent_for") == day:
            return False
        since = local.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC).isoformat()
        goal_id = self.store.active_goal_id()
        progress = "No active goal."
        if goal_id:
            g = next(x for x in self.store.goals() if x["goal_id"] == goal_id)
            goal = self.store.get_goal(goal_id)
            progress = f"Goal '{goal.title}': {g['trials_used']}/{goal.budget_trials} trials."
        by_reason = self.store.reason_counts_since(since)
        self.notify(
            DailyDigest(
                trials_by_reason=by_reason,
                acceptances=by_reason.get("accepted", 0),
                pool_size=len(self.store.pool()),
                llm_spend_usd=self.store.spend_since(since),
                goal_progress=progress,
            )
        )
        self.store.set_flag("digest_sent_for", day)
        return True

    def pause(self, reason: str) -> None:
        self.store.set_flag("paused", "1")
        self.store.set_flag("paused_reason", reason)
        self.notify(LoopPaused(reason=reason))


def _utcnow():  # type: ignore[no-untyped-def]
    from datetime import UTC, datetime

    return datetime.now(UTC)


__all__ = ["BudgetExceededError", "EvaluationError", "Orchestrator", "now"]
