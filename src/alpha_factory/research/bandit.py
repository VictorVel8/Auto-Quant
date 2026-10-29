"""Where effort goes is decided by code, never by the LLM."""

from hashlib import sha256

import numpy as np

from alpha_factory.contracts import BatchDirective, ResearchGoal
from alpha_factory.research.config import LoopConfig
from alpha_factory.research.store import ResearchStore

EXHAUSTED_MIN_TRIALS = 60
EXHAUSTED_WINDOW = 40
REFUTED_MIN_TRIALS = 40


def seed_for(batch_id: str) -> int:
    """A stable integer seed from a batch id, so a directive or template batch can be reproduced."""
    return int(sha256(batch_id.encode()).hexdigest()[:8], 16)


def make_directive(
    goal: ResearchGoal, store: ResearchStore, cfg: LoopConfig, rng: np.random.Generator
) -> BatchDirective:
    """Thompson sampling over the goal's families by recent acceptance, then a fixed recipe."""
    arms = store.families(active_only=True, include=goal.families)
    scores: dict[str, float] = {}
    for fam in arms:
        acc, n = store.family_stats(fam, last_evaluated=cfg.bandit_window)
        scores[fam] = float(rng.beta(1 + acc, 1 + (n - acc)))
    ranked = sorted(scores, key=lambda f: (-scores[f], f))
    mix = cfg.mix_for(store.pool_size(goal.horizons))
    return BatchDirective(
        n_new=mix.new,
        n_mutate=mix.mutate,
        n_refine=mix.refine,
        n_combine=mix.combine,
        explore_families=tuple(ranked[:2]),
        avoid_families=store.families_with_status("exhausted", "refuted"),
        min_outside_families=min(2, mix.new),
    )


def update_family_status(store: ResearchStore, family: str) -> str:
    """Refresh a family's status after a batch. Returns the (possibly new) status."""
    current = store.families_with_status("exhausted", "refuted")
    if family in current:
        return "exhausted" if family in store.families_with_status("exhausted") else "refuted"
    evaluated, accepted, since_last = store.family_totals(family)
    beliefs = store.beliefs(families=[family], include_retired=False)
    if (
        evaluated >= REFUTED_MIN_TRIALS
        and accepted == 0
        and beliefs
        and all(b.status.value == "refuted" for b in beliefs)
    ):
        store.set_family_status(family, "refuted")
        return "refuted"
    if evaluated >= EXHAUSTED_MIN_TRIALS and since_last >= EXHAUSTED_WINDOW:
        store.set_family_status(family, "exhausted")
        return "exhausted"
    return "active"
