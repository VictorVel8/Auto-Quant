"""The research goal, the bandit's directive and the packet the LLM is shown."""

from pydantic import BaseModel, Field

from alpha_factory.contracts.base import Horizon, LLMModel, Record, Sha256, Slug
from alpha_factory.contracts.beliefs import BeliefStatus
from alpha_factory.contracts.evaluation import EvalFeedback, ReasonCode
from alpha_factory.contracts.proposals import FactorProposal


class ResearchGoal(LLMModel):
    """Written by the research manager, approved by you."""

    title: str = Field(max_length=80)
    question: str = Field(max_length=400, description="The research question this goal answers.")
    families: list[Slug] = Field(max_length=5)
    horizons: list[Horizon] = Field(min_length=1)
    budget_trials: int = Field(ge=20, le=500)
    stop_when: str = Field(max_length=200, description="What result ends this goal early.")


class GoalProposals(LLMModel):
    """What the research manager returns: one to three candidate goals."""

    goals: list[ResearchGoal] = Field(
        min_length=1, max_length=3, description="Distinct goals, most promising first."
    )


class BatchDirective(Record):
    """Computed by the bandit, never by an LLM."""

    n_new: int
    n_mutate: int
    n_refine: int
    n_combine: int
    explore_families: tuple[str, ...]
    avoid_families: tuple[str, ...]
    min_outside_families: int


class PoolFactorView(BaseModel):
    factor_id: str
    family: str
    horizon: Horizon
    expression: str
    mechanism: str
    ic: float
    cluster: str


class BeliefView(BaseModel):
    belief_id: str
    statement: str
    status: BeliefStatus
    evidence: str  # "4 for, 1 against"
    next_test: str | None


class TrialDigest(BaseModel):
    n_trials: int
    by_reason: dict[ReasonCode, int]
    patterns: list[str]


class ContextPacket(Record):
    packet_id: str
    contract_hash: Sha256
    contract: str
    goal: ResearchGoal
    directive: BatchDirective
    beliefs: tuple[BeliefView, ...] = Field(max_length=20)
    pool: tuple[PoolFactorView, ...] = Field(max_length=30)
    recent: tuple[EvalFeedback, ...] = Field(max_length=40)
    digest: TrialDigest
    graveyard: tuple[str, ...] = Field(max_length=30)
    exemplars: tuple[FactorProposal, ...] = Field(max_length=3)
