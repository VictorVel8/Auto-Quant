"""A scripted LLM client and an httpx mock provider."""

import json
from datetime import UTC, datetime

import httpx

from alpha_factory.agents.llm_client import LLMError, LLMResponse
from alpha_factory.contracts import LLMCallRecord, LLMModel
from alpha_factory.research.config import LLMConfig

CFG = LLMConfig(
    models={"proposer": "m-prop", "reflector": "m-refl", "manager": "m-man"},
    prices_usd_per_mtok={m: {"input": 1.0, "output": 2.0} for m in ("m-prop", "m-refl", "m-man")},
    temperature={"proposer": 0.8, "reflector": 0.2, "manager": 0.4},
    max_output_tokens={"proposer": 6000, "reflector": 2000, "manager": 3000},
    retries=3,
)


def call_record(
    role: str = "proposer", ok: bool = True, cost: float = 0.01, error: str | None = None
) -> LLMCallRecord:
    return LLMCallRecord(
        call_id=f"llm_{abs(hash((role, error, cost, datetime.now(UTC)))):x}", role=role,  # type: ignore[arg-type]
        provider="test", model="m", prompt_hash="a" * 64, contract_hash=None, tokens_in=100,
        tokens_out=50, latency_ms=5, cost_usd=cost, ok=ok, error=error, created_at=datetime.now(UTC),
    )  # fmt: skip


class ScriptedLLM:
    """Returns the queued texts in order and remembers every request."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.requests: list[dict[str, str]] = []

    def complete_json(
        self, *, role: str, system: str, user: str, output_model: type[LLMModel]
    ) -> LLMResponse:
        self.requests.append(
            {"role": role, "system": system, "user": user, "model": output_model.__name__}
        )
        if not self.texts:
            raise LLMError(
                "no more scripted responses", (call_record(role, ok=False, error="exhausted"),)
            )
        return LLMResponse(text=self.texts.pop(0), calls=(call_record(role),))


def provider(
    responses: list[httpx.Response | Exception], seen: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.MockTransport(handler)


def ok(content: object, tin: int = 1000, tout: int = 500) -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout},
        },
    )
