"""Context packet assembly: deterministic rules over the store, within a token budget.

The same store state always produces the same packet. Tokens are estimated at 3.5 characters each.
When over budget, trimming goes: older recent rows, then graveyard entries, then exemplars. Beliefs
and the directive are never cut.
"""

import json
from collections import Counter

from alpha_factory.agents.feedback import feedback_for, round_ic
from alpha_factory.contracts import (
    BatchDirective,
    BeliefStatus,
    BeliefView,
    ContextPacket,
    FactorProposal,
    PoolFactorView,
    ReasonCode,
    ResearchGoal,
    SandboxSpec,
    TrialDigest,
)
from alpha_factory.dsl.registry import render_docs
from alpha_factory.hashing import sha256_canonical
from alpha_factory.research.store import ResearchStore

TOKEN_BUDGET = 12_000
CHARS_PER_TOKEN = 3.5
MAX_BELIEFS, MAX_POOL, MAX_RECENT, MAX_GRAVEYARD, MAX_EXEMPLARS = 20, 30, 40, 30, 3
ORDER = {BeliefStatus.CONTESTED: 0, BeliefStatus.UNTESTED: 1, BeliefStatus.SUPPORTED: 2}


def contract_text(sandbox: SandboxSpec) -> str:
    """The DSL reference and label definitions. Identical across calls, so providers can cache it."""
    labels = "; ".join(f"{s.name} = {s.hours}h" for s in sandbox.labels)
    return (
        f"{render_docs()}\n\nHorizons (forward return from the next bar's open): {labels}.\n"
        "A factor scores every stock at each 30-minute bar. expected_sign is the sign of the IC "
        "between the expression and the forward return. Family slugs are snake_case, 3 to 40 chars."
    )


def estimate_tokens(packet: ContextPacket) -> int:
    return int(len(packet.model_dump_json()) / CHARS_PER_TOKEN)


def _beliefs(
    store: ResearchStore, goal: ResearchGoal, directive: BatchDirective
) -> tuple[BeliefView, ...]:
    families = set(goal.families) | set(directive.explore_families)
    live = [b for b in store.beliefs(families=sorted(families)) if b.status in ORDER]
    live.sort(key=lambda b: (ORDER[b.status], -b.created_at.timestamp(), b.belief_id))
    return tuple(
        BeliefView(
            belief_id=b.belief_id,
            statement=b.statement,
            status=b.status,
            evidence=f"{b.n_support} for, {b.n_contradict} against",
            next_test=b.next_test,
        )
        for b in live[:MAX_BELIEFS]
    )


def _pool(store: ResearchStore, goal: ResearchGoal) -> tuple[PoolFactorView, ...]:
    rows = store.pool()
    if (
        len(rows) > MAX_POOL
    ):  # best t-stat per cluster (family stands in for it), then goal families
        best: dict[str, object] = {}
        for r in sorted(rows, key=lambda r: (-r.ic_tstat, r.factor_id)):
            best.setdefault(r.family, r)
        keep = list(best.values())
        rest = [r for r in rows if r not in keep and r.family in goal.families]
        rows = ([*keep, *rest])[:MAX_POOL]  # type: ignore[list-item]
    out = []
    for r in rows:
        prop = store.get_proposal(store.get_trial(r.trial_id).proposal_id or "")
        out.append(
            PoolFactorView(
                factor_id=r.factor_id,
                family=r.family,
                horizon=r.horizon,
                expression=r.canonical_expr,
                mechanism=prop.proposal.mechanism,
                ic=round_ic(r.ic_mean),
                cluster=r.family,
            )
        )
    return tuple(out)


def _digest(store: ResearchStore, goal_id: str) -> TrialDigest:
    trials = store.goal_trials(goal_id)
    by_reason: Counter[ReasonCode] = Counter(t.reason for t in trials)
    patterns: list[str] = []
    fam_reasons: dict[str, Counter[str]] = {}
    fam_accept: Counter[str] = Counter()
    parse_errors: Counter[str] = Counter()
    for t in trials:
        if t.family:
            fam_reasons.setdefault(t.family, Counter())[t.reason.value] += 1
            fam_accept[t.family] += t.reason == ReasonCode.ACCEPTED
        if t.reason == ReasonCode.INVALID_PARSE and t.note:
            parse_errors[str(json.loads(t.note).get("message"))] += 1
    for fam in sorted(fam_reasons):
        reason, k = sorted(fam_reasons[fam].items(), key=lambda kv: (-kv[1], kv[0]))[0]
        patterns.append(f"{fam}: mostly {reason} ({k} of {sum(fam_reasons[fam].values())})")
    if parse_errors:
        msg, k = sorted(parse_errors.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        patterns.append(f"most common parse error ({k} times): {msg}")
    for fam in sorted(fam_reasons):
        n = sum(fam_reasons[fam].values())
        if n >= 20 and fam_accept[fam] == 0:
            patterns.append(f"{fam}: no acceptance after {n} trials")
    return TrialDigest(n_trials=len(trials), by_reason=dict(by_reason), patterns=patterns)


def _graveyard(store: ResearchStore, goal_id: str) -> tuple[str, ...]:
    refuted = [
        b.statement
        for b in store.beliefs(include_retired=False)
        if b.status == BeliefStatus.REFUTED
    ]
    dup: Counter[str] = Counter()
    for t in store.goal_trials(goal_id):
        if t.reason == ReasonCode.DUPLICATE and t.note:
            expr = json.loads(t.note).get("expression")
            if expr:
                dup[str(expr)] += 1
    repeated = [f"expression proposed {n + 1} times: {e}" for e, n in sorted(dup.items()) if n >= 1]
    return tuple([*sorted(refuted), *repeated][:MAX_GRAVEYARD])


def _exemplars(store: ResearchStore, goal: ResearchGoal) -> tuple[FactorProposal, ...]:
    rows = [r for r in store.pool() if not goal.families or r.family in goal.families]
    rows.sort(key=lambda r: (-r.ic_tstat, r.factor_id))
    out = []
    for r in rows[:MAX_EXEMPLARS]:
        tr = store.get_trial(r.trial_id)
        out.append(store.get_proposal(tr.proposal_id or "").proposal)
    return tuple(out)


def build_packet(
    store: ResearchStore,
    goal_id: str,
    goal: ResearchGoal,
    directive: BatchDirective,
    sandbox: SandboxSpec,
    batch_id: str,
) -> ContextPacket:
    contract = contract_text(sandbox)
    recent = [
        feedback_for(t, sandbox, store) for t in reversed(store.goal_trials(goal_id, MAX_RECENT))
    ]
    packet = ContextPacket(
        packet_id=f"pkt_{batch_id}",
        contract_hash=sha256_canonical({"contract": contract}),
        contract=contract,
        goal=goal,
        directive=directive,
        beliefs=_beliefs(store, goal, directive),
        pool=_pool(store, goal),
        recent=tuple(recent),
        digest=_digest(store, goal_id),
        graveyard=_graveyard(store, goal_id),
        exemplars=_exemplars(store, goal),
    )
    # trim, in the plan's order: older recent rows, then graveyard entries, then exemplars
    while estimate_tokens(packet) > TOKEN_BUDGET:
        if packet.recent:
            packet = packet.model_copy(update={"recent": packet.recent[1:]})
        elif packet.graveyard:
            packet = packet.model_copy(update={"graveyard": packet.graveyard[:-1]})
        elif packet.exemplars:
            packet = packet.model_copy(update={"exemplars": packet.exemplars[:-1]})
        else:
            break
    return packet
