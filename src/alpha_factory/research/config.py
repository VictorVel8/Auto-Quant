"""configs/loop.yaml and configs/llm.yaml as typed models."""

from typing import Literal

from alpha_factory.contracts.base import Record


class Mix(Record):
    new: int
    mutate: int
    refine: int
    combine: int


class GoalConfig(Record):
    stop_after_batches_without_accept: int = 10


class EvaluationConfig(Record):
    backend: Literal["modal", "local"] = "local"
    max_parallel: int = 12


class LoopConfig(Record):
    proposer: Literal["template", "llm"] = "template"
    batch_size: int = 12
    max_batches_per_day: int = 40
    daily_llm_budget_usd: float = 2.0
    max_consecutive_failures: int = 3
    goal: GoalConfig = GoalConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    bandit_window: int = 200
    pool_cap: int = 40
    replace_t_ratio: float = 1.2
    freeze_suggestion_delta: int = 5

    def mix_for(self, pool_size: int) -> Mix:
        """How a batch splits across proposal types, by active pool size (the plan's table)."""
        base = (12, 0, 0, 0) if pool_size == 0 else (6, 4, 2, 0) if pool_size == 1 else (4, 4, 2, 2)
        scaled = [round(x * self.batch_size / 12) for x in base]
        scaled[0] += self.batch_size - sum(scaled)  # rounding goes to new proposals
        return Mix(new=scaled[0], mutate=scaled[1], refine=scaled[2], combine=scaled[3])


class LLMConfig(Record):
    base_url_env: str = "LLM_BASE_URL"
    supports_json_schema: bool = True
    models: dict[str, str]
    prices_usd_per_mtok: dict[str, dict[str, float]]
    temperature: dict[str, float]
    max_output_tokens: dict[str, int]
    timeout_s: int = 120
    retries: int = 3
