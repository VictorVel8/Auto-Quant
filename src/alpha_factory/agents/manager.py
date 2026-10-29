"""The research manager: drafts goals from a brief of the store. You approve or edit them."""

import json
from pathlib import Path

import yaml

from alpha_factory.agents import prompts
from alpha_factory.agents.feedback import round_ic
from alpha_factory.agents.flow import call_with_repair
from alpha_factory.agents.llm_client import LLMClient
from alpha_factory.contracts import GoalProposals, ResearchGoal
from alpha_factory.research.store import ResearchStore


def brief(store: ResearchStore) -> str:
    """brief.md: family statistics, beliefs by status, the pool, recent digests and closed goals.

    No sealed numbers: only what the loop itself has seen.
    """
    lines = ["# Research brief", "", "## Families"]
    for fam in store.families(active_only=False):
        evaluated, accepted, since = store.family_totals(fam)
        status = store.families_with_status("exhausted", "refuted")
        tag = "exhausted/refuted" if fam in status else "active"
        lines.append(
            f"- {fam} [{tag}]: {evaluated} evaluated, {accepted} accepted, {since} since the last acceptance"
        )
    lines += ["", "## Beliefs"]
    for b in store.beliefs():
        lines.append(
            f"- [{b.status.value}] ({b.family}, {b.horizon}) {b.statement} ({b.n_support} for, {b.n_contradict} against)"
        )
    lines += ["", "## Pool"]
    for p in store.pool():
        lines.append(
            f"- {p.factor_id} [{p.family}, {p.horizon}] IC {round_ic(p.ic_mean)}: {p.canonical_expr}"
        )
    lines += ["", "## Trials by reason"]
    for reason, n in sorted(store.reason_counts().items()):
        lines.append(f"- {reason}: {n}")
    lines += ["", "## Goals"]
    for g in store.goals():
        goal = ResearchGoal.model_validate_json(g["goal_json"])
        lines.append(
            f"- [{g['status']}] {goal.title}: {goal.question} ({g['trials_used']}/{goal.budget_trials} trials; {g['outcome'] or 'open'})"
        )
    return "\n".join(lines) + "\n"


def propose_goals(client: LLMClient, store: ResearchStore) -> list[str]:
    """Ask the manager model for one to three pending goals and save them."""
    res = call_with_repair(
        client, role="manager", system=prompts.render(f"{prompts.MANAGER}.system"),
        user=prompts.render(f"{prompts.MANAGER}.user", brief=brief(store)),
        output_model=GoalProposals, parse=GoalProposals.model_validate_json,
    )  # fmt: skip
    with store.tx():
        for call in res.calls:
            store.add_llm_call(call)
        if res.value is None:
            raise RuntimeError(f"the manager returned no usable goals: {res.error}")
        return [store.add_goal(g) for g in res.value.goals]


def goals_from_file(store: ResearchStore, path: Path) -> list[str]:
    """Add goals from a YAML file (a list of goals, or {goals: [...]}), for example one a chat
    model wrote from brief.md or your own bootstrap goals."""
    raw = yaml.safe_load(path.read_text())
    items = raw["goals"] if isinstance(raw, dict) else raw
    parsed = [ResearchGoal.model_validate(g) for g in items]
    with store.tx():
        return [store.add_goal(g) for g in parsed]


def dump(goal: ResearchGoal) -> str:
    return json.dumps(goal.model_dump(mode="json"), indent=2)
