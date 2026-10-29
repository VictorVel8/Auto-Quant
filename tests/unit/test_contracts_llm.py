"""Valid and invalid examples plus JSON Schema snapshots for every LLMModel."""

import inspect
from typing import Any

import pytest
from pydantic import ValidationError
from syrupy.assertion import SnapshotAssertion

from alpha_factory import contracts
from alpha_factory.contracts import (
    BeliefUpdate,
    FactorProposal,
    GoalProposals,
    LLMModel,
    ProposalBatch,
    Reflection,
    ResearchGoal,
)

PROPOSAL: dict[str, Any] = {
    "type": "new",
    "parents": [],
    "family": "short_term_reversal",
    "horizon": "h1",
    "hypothesis": "Stocks that fell over the last hour rebound over the next hour.",
    "mechanism": "Liquidity providers demand a premium to absorb order-flow imbalance.",
    "expected_sign": "positive",
    "expression": "-Pct($close, 2)",
    "condition": None,
    "refutations": ["cost_killed"],
    "novelty": "Baseline reversal.",
}

BELIEF_UPDATE: dict[str, Any] = {
    "action": "create",
    "belief_id": None,
    "statement": "Reversal is stronger when volume is high.",
    "family": "short_term_reversal",
    "horizon": "h1",
    "evidence": ["trl_1"],
    "next_test": None,
}

GOAL: dict[str, Any] = {
    "title": "Reversal at h1",
    "question": "Does short-term reversal survive costs at h1?",
    "families": ["short_term_reversal"],
    "horizons": ["h1"],
    "budget_trials": 100,
    "stop_when": "Two accepted factors.",
}

# Every LLMModel must be listed here with a valid example and at least one invalid one.
EXAMPLES: dict[type[LLMModel], tuple[dict[str, Any], list[dict[str, Any]]]] = {
    FactorProposal: (
        PROPOSAL,
        [
            {**PROPOSAL, "unknown": 1},
            {**PROPOSAL, "type": "mutate"},  # needs 1 parent
            {**PROPOSAL, "type": "combine", "parents": ["a"]},  # needs 2 to 3
            {**PROPOSAL, "family": "Bad Family"},
            {**PROPOSAL, "horizon": "h3"},
            {**PROPOSAL, "hypothesis": "too short"},
            {**PROPOSAL, "refutations": []},
            {**PROPOSAL, "refutations": ["not_concentrated_in_condition"]},  # no condition
        ],
    ),
    ProposalBatch: (
        {"rationale": "Follows the directive.", "proposals": [PROPOSAL]},
        [
            {"rationale": "x", "proposals": []},
            {"rationale": "x", "proposals": [PROPOSAL] * 17},
            {"rationale": "x" * 601, "proposals": [PROPOSAL]},
        ],
    ),
    BeliefUpdate: (
        BELIEF_UPDATE,
        [
            {**BELIEF_UPDATE, "belief_id": "blf_1"},  # create with an id
            {**BELIEF_UPDATE, "action": "support"},  # support without an id
            {**BELIEF_UPDATE, "statement": None},  # create without a statement
            {**BELIEF_UPDATE, "evidence": []},
        ],
    ),
    Reflection: (
        {"summary": "Reversal works at h1.", "updates": [BELIEF_UPDATE]},
        [{"summary": "x", "updates": [BELIEF_UPDATE] * 11}],
    ),
    GoalProposals: (
        {"goals": [GOAL]},
        [{"goals": []}, {"goals": [GOAL] * 4}, {"goals": [{**GOAL, "budget_trials": 5}]}],
    ),
    ResearchGoal: (
        GOAL,
        [
            {**GOAL, "budget_trials": 5},
            {**GOAL, "horizons": []},
            {**GOAL, "families": ["a_family"] * 6},
        ],
    ),
}


def _llm_models() -> list[type[LLMModel]]:
    return [
        m
        for _, m in inspect.getmembers(contracts, inspect.isclass)
        if issubclass(m, LLMModel) and m is not LLMModel
    ]


def test_every_llm_model_has_examples() -> None:
    assert set(_llm_models()) == set(EXAMPLES)


@pytest.mark.parametrize("model", list(EXAMPLES), ids=lambda m: m.__name__)
def test_valid_example(model: type[LLMModel]) -> None:
    valid, _ = EXAMPLES[model]
    assert model.model_validate(valid)


@pytest.mark.parametrize(
    ("model", "bad"),
    [(m, b) for m, (_, bads) in EXAMPLES.items() for b in bads],
    ids=lambda v: v.__name__ if isinstance(v, type) else "",
)
def test_invalid_example(model: type[LLMModel], bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        model.model_validate(bad)


@pytest.mark.parametrize("model", list(EXAMPLES), ids=lambda m: m.__name__)
def test_json_schema_snapshot(model: type[LLMModel], snapshot: SnapshotAssertion) -> None:
    assert model.model_json_schema() == snapshot


def test_models_are_frozen() -> None:
    p = FactorProposal.model_validate(PROPOSAL)
    with pytest.raises(ValidationError):
        p.family = "other_family"  # type: ignore[misc]


def test_whitespace_is_stripped() -> None:
    p = FactorProposal.model_validate({**PROPOSAL, "expression": "  -Pct($close, 2)  "})
    assert p.expression == "-Pct($close, 2)"


def test_schema_nesting_is_shallow() -> None:
    """LLM-facing schemas stay at most 2 levels deep (batch -> proposal -> flat fields)."""
    schema = ProposalBatch.model_json_schema()
    defs = schema["$defs"]
    assert set(defs) == {"FactorProposal", "RefutationCheck"}
    for prop in defs["FactorProposal"]["properties"].values():
        assert prop.get("type") != "object"


def test_parents_must_have_been_shown() -> None:
    bad = {**PROPOSAL, "type": "mutate", "parents": ["fac_unknown"]}
    with pytest.raises(ValidationError, match="unknown parents"):
        FactorProposal.model_validate(bad, context={"pool_ids": {"fac_1"}})
    ok = FactorProposal.model_validate(bad, context={"pool_ids": {"fac_unknown"}})
    assert ok.parents == ["fac_unknown"]


def test_belief_update_cross_references() -> None:
    support = {**BELIEF_UPDATE, "action": "support", "belief_id": "blf_1", "statement": None}
    with pytest.raises(ValidationError, match="current batch"):
        BeliefUpdate.model_validate(support, context={"batch_trial_ids": {"trl_9"}})
    with pytest.raises(ValidationError, match="unknown belief"):
        BeliefUpdate.model_validate(support, context={"belief_ids": {"blf_2"}})
    BeliefUpdate.model_validate(
        support, context={"batch_trial_ids": {"trl_1"}, "belief_ids": {"blf_1"}}
    )
