from pathlib import Path

import numpy as np
import pytest
from tests.helpers.store import GOAL, manifest, open_store, proposal, result, sandbox

from alpha_factory.agents.template import CONDITIONS, TEMPLATES, TemplateProposer, mutate_expression
from alpha_factory.contracts import BatchDirective, FactorProposal, ReasonCode, ResearchGoal
from alpha_factory.dsl import compile_expr
from alpha_factory.research.bandit import make_directive, seed_for, update_family_status
from alpha_factory.research.config import LoopConfig
from alpha_factory.research.store import ResearchStore

SB, MF = sandbox(), manifest()
GOAL3 = ResearchGoal(
    title="Three families", question="Which of these families predicts returns at h1?",
    families=["short_term_reversal", "volume_shock", "overnight_gap"], horizons=["h1", "h2"],
    budget_trials=100, stop_when="One accepted factor.",
)  # fmt: skip


def directive(new: int = 4, mutate: int = 0, refine: int = 0, combine: int = 0) -> BatchDirective:
    return BatchDirective(
        n_new=new, n_mutate=mutate, n_refine=refine, n_combine=combine,
        explore_families=("short_term_reversal",), avoid_families=(), min_outside_families=1,
    )  # fmt: skip


def seed_pool(store: ResearchStore, batch_id: str, specs: list[tuple[str, str, int]]) -> list[str]:
    """specs: (expression, family, sign). Adds accepted factors to the pool."""
    goal_id = store.get_batch(batch_id)["goal_id"]
    ids = []
    for i, (expr, family, sign) in enumerate(specs):
        prop = proposal(batch_id, goal_id, expr, family, n=i)
        store.add_proposals([prop])
        res = result(
            SB,
            prop.expr_hash,
            ReasonCode.ACCEPTED,
            trial_id=f"trl_pool_{prop.expr_hash[-10:]}",
            sign=sign,
        )
        tid = store.record_trial(
            source="template",
            sandbox=SB,
            manifest=MF,
            reason=ReasonCode.ACCEPTED,
            proposal=prop,
            result=res,
        )
        ids.append(store.add_to_pool(tid))
    return ids


def test_every_template_compiles_for_several_windows() -> None:
    for items in TEMPLATES.values():
        for tpl, sign, _ in items:
            assert sign in ("positive", "negative")
            for n, m, k in [(3, 5, 2), (13, 26, 3), (26, 65, 5)]:  # TsZScore and Corr need n >= 3
                compile_expr(tpl.format(n=n, m=m, k=k))


def test_a_template_batch_is_valid_typed_and_reproducible(tmp_path: Path) -> None:
    store, _, _ = open_store(tmp_path)
    out = TemplateProposer().propose(directive(new=12), GOAL3, store, rng_seed=7)
    assert out.raw is not None and len(out.raw.proposals) == 12
    for item in out.raw.proposals:
        FactorProposal.model_validate(
            item
        )  # text lengths, parents-by-type and refutations all pass
        compile_expr(item["expression"], condition=item["condition"])
        assert item["horizon"] in GOAL3.horizons and item["family"] in GOAL3.families
    again = TemplateProposer().propose(directive(new=12), GOAL3, store, rng_seed=7)
    assert again.raw is not None and again.raw.proposals == out.raw.proposals
    other = TemplateProposer().propose(directive(new=12), GOAL3, store, rng_seed=8)
    assert other.raw is not None and other.raw.proposals != out.raw.proposals


def test_expressions_within_a_batch_and_across_batches_are_distinct(tmp_path: Path) -> None:
    store, _, _ = open_store(tmp_path)
    p = TemplateProposer()
    first = p.propose(directive(new=12), GOAL3, store, 1)
    second = p.propose(directive(new=12), GOAL3, store, 2)
    exprs = [i["expression"] for o in (first, second) if o.raw for i in o.raw.proposals]
    assert len(set(exprs)) == len(exprs) == 24


def test_derived_proposals_come_from_the_pool_with_valid_parents(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    ids = seed_pool(store, batch_id, [("Neg(Pct($close, 5))", "short_term_reversal", 1),
                                      ("Div($volume, Mean($volume, 26))", "volume_shock", 1),
                                      ("Pct($close, 65)", "momentum", -1)])  # fmt: skip
    out = TemplateProposer().propose(
        directive(new=2, mutate=3, refine=2, combine=2), GOAL3, store, 11
    )
    assert out.raw is not None
    kinds = [i["type"] for i in out.raw.proposals]
    assert kinds.count("mutate") == 3 and kinds.count("refine") == 2 and kinds.count("combine") == 2
    for item in out.raw.proposals:
        proposal_ = FactorProposal.model_validate(
            item, context={"pool_ids": set(ids)}
        )  # parents exist
        if item["type"] == "refine":
            assert item["condition"] in [
                compile_expr("$ret", condition=c).condition for c in CONDITIONS
            ]
        if item["type"] == "combine":
            assert len(proposal_.parents) == 2 and "Add(" in item["expression"]
            assert len({p for p in proposal_.parents}) == 2


def test_with_an_empty_pool_derived_slots_fall_back_to_new(tmp_path: Path) -> None:
    store, _, _ = open_store(tmp_path)
    out = TemplateProposer().propose(
        directive(new=2, mutate=2, refine=1, combine=1), GOAL3, store, 3
    )
    assert out.raw is not None and {i["type"] for i in out.raw.proposals} == {"new"}


@pytest.mark.parametrize(
    "expr",
    ["Neg(Pct($close, 5))", "Mean(Div($volume, Mean($volume, 26)), 8)", "Rank(Delta($close, 13))"],
)
def test_mutations_change_the_expression_and_stay_valid(expr: str) -> None:
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(25):
        m = mutate_expression(expr, rng)
        assert m != compile_expr(expr).expression
        compile_expr(m)
        seen.add(m)
    assert len(seen) >= 4


def test_seed_is_stable_per_batch_id() -> None:
    assert seed_for("bat_abc") == seed_for("bat_abc") != seed_for("bat_abd")


def test_directive_follows_the_pool_size_table_and_is_reproducible(tmp_path: Path) -> None:
    store, _, batch_id = open_store(tmp_path)
    cfg = LoopConfig()
    d0 = make_directive(GOAL3, store, cfg, np.random.default_rng(seed_for("b1")))
    assert (d0.n_new, d0.n_mutate, d0.n_refine, d0.n_combine) == (12, 0, 0, 0)
    assert d0.min_outside_families == 2 and len(d0.explore_families) == 2
    assert d0 == make_directive(GOAL3, store, cfg, np.random.default_rng(seed_for("b1")))
    seed_pool(store, batch_id, [("Neg(Pct($close, 5))", "short_term_reversal", 1)])
    d1 = make_directive(GOAL3, store, cfg, np.random.default_rng(1))
    assert (d1.n_new, d1.n_mutate, d1.n_refine, d1.n_combine) == (6, 4, 2, 0)
    seed_pool(store, batch_id, [("Pct($close, 65)", "momentum", 1)])
    d2 = make_directive(GOAL3, store, cfg, np.random.default_rng(1))
    assert (d2.n_new, d2.n_mutate, d2.n_refine, d2.n_combine) == (4, 4, 2, 2)


def test_batch_size_scales_the_mix_and_always_sums_up() -> None:
    for size in (6, 8, 12, 16):
        for pool in (0, 1, 5):
            m = LoopConfig(batch_size=size).mix_for(pool)
            assert (
                m.new + m.mutate + m.refine + m.combine == size
                and min(m.new, m.mutate, m.refine, m.combine) >= 0
            )


def test_thompson_sampling_prefers_the_family_that_keeps_getting_accepted(tmp_path: Path) -> None:
    store, goal_id, batch_id = open_store(tmp_path)
    for i in range(40):  # volume_shock accepts often, the others never
        prop = proposal(batch_id, goal_id, f"Pct($close, {i + 2})", "volume_shock", n=i)
        store.add_proposals([prop])
        r = ReasonCode.ACCEPTED if i % 2 == 0 else ReasonCode.LOW_IC
        store.record_trial(
            source="template",
            sandbox=SB,
            manifest=MF,
            reason=r,
            proposal=prop,
            result=result(SB, prop.expr_hash, r, trial_id=f"trl_v{i}"),
        )
    for field_, fam in (("open", "short_term_reversal"), ("high", "overnight_gap")):
        for i in range(40):  # the others have a record of failing (distinct expressions)
            prop = proposal(batch_id, goal_id, f"Pct(${field_}, {i + 2})", fam, n=i + 100)
            store.add_proposals([prop])
            res = result(SB, prop.expr_hash, ReasonCode.LOW_IC, trial_id=f"trl_{fam[:3]}{i}")
            store.record_trial(
                source="template", sandbox=SB, manifest=MF, reason=ReasonCode.LOW_IC,
                proposal=prop, result=res,
            )  # fmt: skip
    firsts = [
        make_directive(GOAL3, store, LoopConfig(), np.random.default_rng(s)).explore_families[0]
        for s in range(50)
    ]
    assert firsts.count("volume_shock") > 30


def test_families_become_exhausted_and_stay_avoided(tmp_path: Path) -> None:
    store, goal_id, batch_id = open_store(tmp_path)

    def add(n: int, reason: ReasonCode, start: int) -> None:
        for i in range(start, start + n):
            prop = proposal(batch_id, goal_id, f"Pct($close, {i + 2})", "volume_shock", n=i)
            store.add_proposals([prop])
            store.record_trial(
                source="template",
                sandbox=SB,
                manifest=MF,
                reason=reason,
                proposal=prop,
                result=result(SB, prop.expr_hash, reason, trial_id=f"trl_e{i}"),
            )

    add(1, ReasonCode.ACCEPTED, 0)
    add(30, ReasonCode.LOW_IC, 1)
    assert update_family_status(store, "volume_shock") == "active"  # 31 trials
    add(30, ReasonCode.LOW_IC, 31)  # 61 trials, 60 since the acceptance
    assert update_family_status(store, "volume_shock") == "exhausted"
    d = make_directive(GOAL3, store, LoopConfig(), np.random.default_rng(0))
    assert "volume_shock" in d.avoid_families
    store.set_family_status("volume_shock", "active")  # af family activate
    assert (
        "volume_shock"
        not in make_directive(GOAL3, store, LoopConfig(), np.random.default_rng(0)).avoid_families
    )


def test_a_family_with_no_acceptance_and_only_refuted_beliefs_is_refuted(tmp_path: Path) -> None:
    store, goal_id, batch_id = open_store(tmp_path)
    trials = []
    for i in range(40):
        prop = proposal(batch_id, goal_id, f"Pct($close, {i + 2})", "overnight_gap", n=i)
        store.add_proposals([prop])
        trials.append(
            store.record_trial(
                source="template",
                sandbox=SB,
                manifest=MF,
                reason=ReasonCode.LOW_IC,
                proposal=prop,
                result=result(SB, prop.expr_hash, ReasonCode.LOW_IC, trial_id=f"trl_r{i}"),
            )
        )
    assert update_family_status(store, "overnight_gap") == "active"  # no beliefs yet
    b = store.create_belief(
        statement="Gaps do not fill by midday.",
        family="overnight_gap",
        horizon="h1",
        next_test=None,
    )
    for t in trials[:5]:
        store.add_evidence(b, t, "contradict", batch_id)
    assert update_family_status(store, "overnight_gap") == "refuted"
    assert GOAL.families == ["short_term_reversal"]
