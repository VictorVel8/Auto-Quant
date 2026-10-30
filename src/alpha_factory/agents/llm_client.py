"""One OpenAI-compatible LLM client behind a protocol (Meta API, Fireworks, anything similar).

Every attempt, failed or not, becomes an LLMCallRecord with tokens and cost. With
supports_json_schema the request uses strict JSON-schema output; otherwise plain JSON mode with the
schema in the prompt. Strict mode needs every property required, so `to_strict_schema` rewrites the
Pydantic schema; that rewritten schema is what the snapshot tests cover.
"""

import json
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import httpx

from alpha_factory.contracts import LLMCallRecord, LLMModel
from alpha_factory.hashing import sha256_canonical
from alpha_factory.ids import new_id
from alpha_factory.research.config import LLMConfig

Role = Literal["proposer", "reflector", "repair", "manager"]
RETRYABLE = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """The provider failed after the configured retries. `calls` holds every attempt made."""

    def __init__(self, message: str, calls: tuple[LLMCallRecord, ...] = ()) -> None:
        super().__init__(message)
        self.calls = calls


@dataclass(frozen=True)
class LLMResponse:
    text: str  # the model's JSON text ("" when every attempt failed)
    calls: tuple[LLMCallRecord, ...] = field(default_factory=tuple)  # every attempt, in order


class LLMClient(Protocol):
    def complete_json(
        self, *, role: Role, system: str, user: str, output_model: type[LLMModel]
    ) -> LLMResponse: ...


def to_strict_schema(model: type[LLMModel]) -> dict[str, Any]:
    """The Pydantic JSON Schema rewritten for strict structured output.

    Every property is required (optional ones become nullable), additionalProperties is false, and
    defaults are dropped, which strict mode does not accept.
    """
    schema = model.model_json_schema()

    def nullable(prop: dict[str, Any]) -> dict[str, Any]:
        if "anyOf" in prop:
            if not any(x.get("type") == "null" for x in prop["anyOf"]):
                prop["anyOf"] = [*prop["anyOf"], {"type": "null"}]
        elif "$ref" in prop:
            title = prop.pop("title", None)
            ref = {"$ref": prop.pop("$ref")}
            prop["anyOf"] = [ref, {"type": "null"}]
            if title:
                prop["title"] = title
        elif isinstance(prop.get("type"), str):
            prop["type"] = [prop["type"], "null"]
        elif isinstance(prop.get("type"), list) and "null" not in prop["type"]:
            prop["type"].append("null")
        return prop

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object" and "properties" in node:
                required = set(node.get("required", []))
                for name, prop in node["properties"].items():
                    if name not in required:
                        nullable(prop)
                node["required"] = list(node["properties"])
                node["additionalProperties"] = False
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


class OpenAICompatClient:
    def __init__(
        self,
        cfg: LLMConfig,
        base_url: str,
        api_key: str,
        *,
        provider: str = "openai-compatible",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.cfg, self.provider = cfg, provider
        self._sleep, self._rng = sleep, rng or random.Random(0)
        self._http = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=cfg.timeout_s,
            transport=transport,
        )

    def _cost(self, model: str, tokens_in: int, tokens_out: int) -> float:
        price = self.cfg.prices_usd_per_mtok.get(model, {"input": 0.0, "output": 0.0})
        return (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6

    def _body(
        self, role: Role, system: str, user: str, output_model: type[LLMModel]
    ) -> dict[str, Any]:
        key = "proposer" if role == "repair" else role
        body: dict[str, Any] = {
            "model": self.cfg.models[key],
            "temperature": self.cfg.temperature[key],
            "max_tokens": self.cfg.max_output_tokens[key],
        }
        if key in self.cfg.reasoning_effort:
            body["reasoning_effort"] = self.cfg.reasoning_effort[key]
        if self.cfg.supports_json_schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__,
                    "schema": to_strict_schema(output_model),
                    "strict": True,
                },
            }
            body["messages"] = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
            body["response_format"] = {"type": "json_object"}
            schema = json.dumps(output_model.model_json_schema())
            body["messages"] = [
                {
                    "role": "system",
                    "content": f"{system}\n\nReturn JSON matching this schema:\n{schema}",
                },
                {"role": "user", "content": user},
            ]
        return body

    def complete_json(
        self, *, role: Role, system: str, user: str, output_model: type[LLMModel]
    ) -> LLMResponse:
        body = self._body(role, system, user, output_model)
        model = str(body["model"])
        prompt_hash = sha256_canonical({"system": system, "user": user})
        calls: list[LLMCallRecord] = []
        last_error = ""
        for attempt in range(self.cfg.retries + 1):
            t0 = time.perf_counter()
            text, tin, tout, ok, err = "", 0, 0, False, None
            retry = False
            try:
                r = self._http.post("/chat/completions", json=body)
                if r.status_code in RETRYABLE:
                    err, retry = f"HTTP {r.status_code}", True
                elif r.status_code >= 400:
                    err = f"HTTP {r.status_code}: {r.text[:200]}"  # a bad request will not improve
                else:
                    data = r.json()
                    text = data["choices"][0]["message"]["content"] or ""
                    usage = data.get("usage") or {}
                    tin, tout = (
                        int(usage.get("prompt_tokens", 0)),
                        int(usage.get("completion_tokens", 0)),
                    )
                    ok = True
            except httpx.TransportError as e:
                err, retry = f"{type(e).__name__}: {e}", True
            except (KeyError, IndexError, TypeError, ValueError) as e:
                err = f"malformed response: {e!r}"
            calls.append(
                LLMCallRecord(
                    call_id=new_id("llm"),
                    role=role,
                    provider=self.provider,
                    model=model,
                    prompt_hash=prompt_hash,
                    contract_hash=None,
                    tokens_in=tin,
                    tokens_out=tout,
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                    cost_usd=self._cost(model, tin, tout),
                    ok=ok,
                    error=err,
                    created_at=datetime.now(UTC),
                )
            )
            if ok:
                return LLMResponse(text=text, calls=tuple(calls))
            last_error = err or "unknown error"
            if not retry or attempt == self.cfg.retries:
                break
            self._sleep(min(30.0, 2.0**attempt) * (0.5 + self._rng.random()))
        raise LLMError(
            f"{role} call failed after {len(calls)} attempts: {last_error}", tuple(calls)
        )
