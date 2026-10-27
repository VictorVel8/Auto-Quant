"""Beliefs: claims about mechanisms, with evidence counts computed by code."""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, ValidationInfo, computed_field, model_validator

from alpha_factory.contracts.base import Horizon, LLMModel, Record, Slug


class BeliefUpdate(LLMModel):
    action: Literal["create", "support", "contradict", "refine", "retire"]
    belief_id: str | None = Field(
        default=None, description="Required unless action is create. An id from the packet."
    )
    statement: str | None = Field(
        default=None,
        max_length=300,
        description="Required for create and refine. One claim about a mechanism in a context.",
    )
    family: Slug
    horizon: Horizon
    evidence: list[str] = Field(
        min_length=1, max_length=20, description="Trial ids from THIS batch only."
    )
    next_test: str | None = Field(
        default=None, max_length=200, description="One concrete follow-up experiment."
    )

    @model_validator(mode="after")
    def _fields_match_action(self, info: ValidationInfo) -> "BeliefUpdate":
        if (self.action == "create") != (self.belief_id is None):
            raise ValueError("belief_id is required unless action is create")
        if self.action in ("create", "refine") and not self.statement:
            raise ValueError("create and refine need a statement")
        ctx: dict[str, Any] = info.context or {}
        if "batch_trial_ids" in ctx and not set(self.evidence) <= ctx["batch_trial_ids"]:
            raise ValueError("evidence must come from the current batch")
        if self.belief_id and "belief_ids" in ctx and self.belief_id not in ctx["belief_ids"]:
            raise ValueError(f"unknown belief {self.belief_id}")
        return self


class Reflection(LLMModel):
    summary: str = Field(max_length=500, description="What this batch taught, in 2 to 3 sentences.")
    updates: list[BeliefUpdate] = Field(max_length=10)


class BeliefStatus(StrEnum):
    UNTESTED = "untested"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    REFUTED = "refuted"
    RETIRED = "retired"


class Belief(Record):
    belief_id: str
    statement: str
    family: str
    horizon: Horizon
    n_support: int = 0
    n_contradict: int = 0
    supersedes: str | None = None
    next_test: str | None = None
    retired: bool = False
    created_at: datetime
    updated_at: datetime

    @computed_field
    @property
    def posterior_mean(self) -> float:  # Beta(1, 1) prior
        return (1 + self.n_support) / (2 + self.n_support + self.n_contradict)

    @computed_field
    @property
    def status(self) -> BeliefStatus:
        if self.retired:
            return BeliefStatus.RETIRED
        if self.n_support + self.n_contradict < 3:
            return BeliefStatus.UNTESTED
        if self.posterior_mean >= 0.65:
            return BeliefStatus.SUPPORTED
        if self.posterior_mean <= 0.35:
            return BeliefStatus.REFUTED
        return BeliefStatus.CONTESTED
