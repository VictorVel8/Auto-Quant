"""The reflector: one call per LLM batch that turns results into belief updates.

Belief counts are always recomputed from the evidence table, never incremented, so they cannot
drift. Evidence ids are validated against this batch's trials, so a belief can only cite trials
from the batch that produced the reflection.
"""

import json

import structlog

from alpha_factory.agents import prompts
from alpha_factory.agents.feedback import feedback_for
from alpha_factory.agents.flow import call_with_repair
from alpha_factory.agents.llm_client import LLMClient
from alpha_factory.contracts import Reflection, SandboxSpec
from alpha_factory.research.store import ResearchStore

log = structlog.get_logger(__name__)


def reflector_input(store: ResearchStore, batch_id: str, sandbox: SandboxSpec) -> dict[str, object]:
    """The beliefs in play and each trial of the batch with its hypothesis and rounded feedback."""
    trials = store.batch_trials(batch_id)
    families = sorted({t.family for t in trials if t.family})
    results = []
    for t in trials:
        fb = feedback_for(t, sandbox, store)
        ev = store.get_eval(t.trial_id)
        hyp = mech = ""
        if t.proposal_id:
            p = store.get_proposal(t.proposal_id).proposal
            hyp, mech = p.hypothesis, p.mechanism
        results.append(
            {
                **fb.model_dump(mode="json"),
                "hypothesis": hyp,
                "mechanism": mech,
                "hypothesis_refuted": bool(ev and ev[0].hypothesis_refuted),
                "triggered": [
                    o.check.value for o in (ev[0].refutations if ev else ()) if o.triggered
                ],
            }
        )
    return {"beliefs": store.beliefs(families=families), "results": results}


def apply_reflection(store: ResearchStore, batch_id: str, reflection: Reflection) -> int:
    """Apply belief updates in one transaction. Returns how many beliefs were created or changed."""
    refuted = {
        t.trial_id: bool((ev := store.get_eval(t.trial_id)) and ev[0].hypothesis_refuted)
        for t in store.batch_trials(batch_id)
    }
    changed = 0
    with store.tx():
        for u in reflection.updates:
            if u.action == "create":
                bid = store.create_belief(
                    statement=u.statement or "", family=u.family, horizon=u.horizon, next_test=u.next_test
                )  # fmt: skip
                store.ensure_family(u.family, "llm")
                for tid in u.evidence:
                    store.add_evidence(bid, tid, "support", batch_id)
            elif u.action in ("support", "contradict"):
                assert u.belief_id is not None
                for tid in u.evidence:
                    # A rejection that did not refute the hypothesis says the expression failed,
                    # not the mechanism (reflector rule 4): it cannot count against a belief.
                    if u.action == "contradict" and not refuted.get(tid, False):
                        log.info("contradiction dropped: hypothesis not refuted", trial=tid)
                        continue
                    store.add_evidence(u.belief_id, tid, u.action, batch_id)
                if u.next_test:
                    store.update_next_test(u.belief_id, u.next_test)
            elif u.action == "refine":
                assert u.belief_id is not None
                bid = store.create_belief(
                    statement=u.statement or "", family=u.family, horizon=u.horizon,
                    next_test=u.next_test, supersedes=u.belief_id,
                )  # fmt: skip
                for tid in u.evidence:
                    store.add_evidence(bid, tid, "support", batch_id)
                store.retire_belief(u.belief_id)
            else:  # retire
                assert u.belief_id is not None
                store.retire_belief(u.belief_id)
            changed += 1
        store.save_reflection(batch_id, reflection.model_dump_json())
    return changed


class LLMReflector:
    def __init__(self, client: LLMClient, store: ResearchStore, sandbox: SandboxSpec) -> None:
        self.client, self.store, self.sandbox = client, store, sandbox

    def reflect(self, batch_id: str) -> None:
        store = self.store
        ids = {t.trial_id for t in store.batch_trials(batch_id)}
        ctx = {"batch_trial_ids": ids, "belief_ids": store.belief_ids()}
        user = prompts.render(
            f"{prompts.REFLECTOR}.user", **reflector_input(store, batch_id, self.sandbox)
        )
        res = call_with_repair(
            self.client, role="reflector", system=prompts.render(f"{prompts.REFLECTOR}.system"),
            user=user, output_model=Reflection,
            parse=lambda text: Reflection.model_validate(json.loads(text), context=ctx),
        )  # fmt: skip
        with store.tx():
            for call in res.calls:
                store.add_llm_call(call)
        if res.value is None:  # a failed reflection costs the batch its learning, not its trials
            log.warning("reflection failed", batch=batch_id, error=res.error)
            return
        apply_reflection(store, batch_id, res.value)
