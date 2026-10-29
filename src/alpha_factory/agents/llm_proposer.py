"""The LLM proposer: packet -> prompt -> one validated, repairable call -> a loose proposal batch."""

from alpha_factory.agents import prompts
from alpha_factory.agents.flow import call_with_repair
from alpha_factory.agents.llm_client import LLMClient
from alpha_factory.agents.packet import build_packet
from alpha_factory.agents.proposer import ProposalOutcome
from alpha_factory.contracts import (
    BatchDirective,
    ProposalBatch,
    RawProposalBatch,
    ResearchGoal,
    SandboxSpec,
)
from alpha_factory.research.store import ResearchStore


class LLMProposer:
    def __init__(self, client: LLMClient, sandbox: SandboxSpec) -> None:
        self.client, self.sandbox = client, sandbox

    def propose(
        self, directive: BatchDirective, goal: ResearchGoal, store: ResearchStore, rng_seed: int
    ) -> ProposalOutcome:  # fmt: skip
        goal_id = store.active_goal_id()
        if goal_id is None:
            return ProposalOutcome(raw=None, failure="no active goal")
        packet = build_packet(store, goal_id, goal, directive, self.sandbox, f"{rng_seed:08x}")
        system = prompts.render(f"{prompts.PROPOSER}.system")
        exemplars = [e.model_dump(mode="json") for e in packet.exemplars]
        user = prompts.render(
            f"{prompts.PROPOSER}.user", packet=packet, goal=goal, exemplars=exemplars
        )
        res = call_with_repair(
            self.client, role="proposer", system=system, user=user, output_model=ProposalBatch,
            parse=RawProposalBatch.model_validate_json, contract_hash=packet.contract_hash,
        )  # fmt: skip
        return ProposalOutcome(
            raw=res.value, failure=res.error, packet_json=packet.model_dump_json(),
            response_text=res.text, llm_calls=res.calls,
        )  # fmt: skip
