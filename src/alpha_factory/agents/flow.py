"""The call flow shared by every LLM request: call, validate, repair once, give up cleanly."""

from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import ValidationError

from alpha_factory.agents import prompts
from alpha_factory.agents.llm_client import LLMClient, LLMError, Role
from alpha_factory.contracts import LLMCallRecord, LLMModel


@dataclass(frozen=True)
class Validated[T]:
    value: T | None
    text: str  # the last response text
    calls: tuple[LLMCallRecord, ...] = field(default_factory=tuple)
    error: str | None = None


def call_with_repair[T](
    client: LLMClient,
    *,
    role: Role,
    system: str,
    user: str,
    output_model: type[LLMModel],
    parse: Callable[[str], T],
    contract_hash: str | None = None,
) -> Validated[T]:
    """Call the model and validate the reply. On failure make one repair call that includes the
    validation errors; a second failure returns no value (the caller marks the batch failed)."""
    calls: list[LLMCallRecord] = []
    text = ""
    message = user
    for attempt in range(2):
        try:
            resp = client.complete_json(
                role=role if attempt == 0 else "repair", system=system, user=message,
                output_model=output_model,
            )  # fmt: skip
        except LLMError as e:
            calls += [c.model_copy(update={"contract_hash": contract_hash}) for c in e.calls]
            return Validated(None, text, tuple(calls), str(e))
        calls += [c.model_copy(update={"contract_hash": contract_hash}) for c in resp.calls]
        text = resp.text
        try:
            return Validated(parse(text), text, tuple(calls))
        except (ValidationError, ValueError) as e:
            errors = str(e)[:1500]
            if attempt == 1:
                return Validated(None, text, tuple(calls), f"invalid after repair: {errors}")
            message = (
                f"{user}\n\n## YOUR PREVIOUS RESPONSE\n{text}\n\n"
                f"{prompts.render(prompts.REPAIR, errors=errors)}"
            )
    raise AssertionError("unreachable")
