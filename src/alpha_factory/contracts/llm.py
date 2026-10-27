"""Accounting record for one LLM call."""

from datetime import datetime
from typing import Literal

from alpha_factory.contracts.base import Record, Sha256


class LLMCallRecord(Record):
    call_id: str
    role: Literal["proposer", "reflector", "repair", "manager"]
    provider: str
    model: str
    prompt_hash: Sha256
    contract_hash: Sha256 | None
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    ok: bool
    error: str | None
    created_at: datetime
