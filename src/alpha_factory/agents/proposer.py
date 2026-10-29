"""The proposer interface. Template and LLM proposers return the same thing."""

from dataclasses import dataclass, field
from typing import Protocol

from alpha_factory.contracts import BatchDirective, LLMCallRecord, RawProposalBatch, ResearchGoal
from alpha_factory.research.store import ResearchStore


@dataclass(frozen=True)
class ProposalOutcome:
    """A batch of loosely validated proposals (or why there is none), plus what produced it.

    `raw` is the loose envelope: items stay dicts so one bad item cannot sink the batch.
    """

    raw: RawProposalBatch | None
    failure: str | None = None
    packet_json: str | None = None
    response_text: str | None = None
    llm_calls: tuple[LLMCallRecord, ...] = field(default_factory=tuple)


class Proposer(Protocol):
    def propose(
        self, directive: BatchDirective, goal: ResearchGoal, store: ResearchStore, rng_seed: int
    ) -> ProposalOutcome: ...
