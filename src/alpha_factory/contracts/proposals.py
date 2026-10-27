"""Proposer output: what the LLM writes, and what code stores about it."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationInfo, model_validator

from alpha_factory.contracts.base import Horizon, LLMModel, Record, Sha256, Slug


class RefutationCheck(StrEnum):
    SUBSUMED_BY_BASELINE = "subsumed_by_baseline"
    SIGN_UNSTABLE = "sign_unstable"
    NOT_CONCENTRATED_IN_CONDITION = "not_concentrated_in_condition"  # needs `condition`
    WEAK_AT_DECLARED_HORIZON = "weak_at_declared_horizon"
    COST_KILLED = "cost_killed"
    CORRELATED_WITH_PARENT = "correlated_with_parent"


class FactorProposal(LLMModel):
    type: Literal["new", "mutate", "refine", "combine"] = Field(
        description="new: no parents. mutate or refine: exactly 1 parent. combine: 2 to 3 parents."
    )
    parents: list[str] = Field(
        default_factory=list,
        max_length=3,
        description="Ids of pool factors listed in the packet. Never invent ids.",
    )
    family: Slug = Field(description="A family slug from the packet, or a new snake_case slug.")
    horizon: Horizon = Field(description="Forward-return label this factor targets.")
    hypothesis: str = Field(
        min_length=20, max_length=400, description="One falsifiable claim about future returns."
    )
    mechanism: str = Field(
        min_length=20,
        max_length=400,
        description="Why the effect exists: who trades, under what pressure.",
    )
    expected_sign: Literal["positive", "negative"] = Field(
        description="Sign of the IC between the expression and the forward return."
    )
    expression: str = Field(
        min_length=3,
        max_length=300,
        description="DSL expression using only the listed fields and operators.",
    )
    condition: str | None = Field(
        default=None,
        max_length=200,
        description="Optional boolean DSL expression marking where the effect should be strongest.",
    )
    refutations: list[RefutationCheck] = Field(
        min_length=1, max_length=4, description="Results that would show the hypothesis is wrong."
    )
    novelty: str = Field(
        max_length=200, description="What this adds compared with its parents and the pool."
    )

    @model_validator(mode="after")
    def _parents_match_type(self) -> "FactorProposal":
        n = len(self.parents)
        ok = {"new": n == 0, "mutate": n == 1, "refine": n == 1, "combine": 2 <= n <= 3}[self.type]
        if not ok:
            raise ValueError(f"type={self.type} cannot have {n} parents")
        if (
            RefutationCheck.NOT_CONCENTRATED_IN_CONDITION in self.refutations
            and self.condition is None
        ):
            raise ValueError("not_concentrated_in_condition requires a condition")
        return self

    @model_validator(mode="after")
    def _parents_were_shown(self, info: ValidationInfo) -> "FactorProposal":
        ctx: dict[str, Any] = info.context or {}
        shown: set[str] | None = ctx.get("pool_ids")
        if shown is not None and not set(self.parents) <= shown:
            raise ValueError(f"unknown parents: {sorted(set(self.parents) - shown)}")
        return self


class ProposalBatch(LLMModel):
    """What the proposer returns. Its JSON Schema is what gets sent."""

    rationale: str = Field(
        max_length=600, description="How this batch follows the directive and the current beliefs."
    )
    proposals: list[FactorProposal] = Field(min_length=1, max_length=16)


class RawProposalBatch(BaseModel):
    """Loose envelope, validated first, so one bad item can't sink the batch."""

    rationale: str
    proposals: list[dict[str, Any]]


class ProposalRecord(Record):
    proposal_id: str  # "prp_" + ULID
    batch_id: str
    goal_id: str
    llm_call_id: str | None  # None for template proposals
    proposal: FactorProposal
    canonical_expr: str
    expr_hash: Sha256  # dedupe key
    ast_depth: int
    windows_used: tuple[int, ...]
    created_at: datetime
