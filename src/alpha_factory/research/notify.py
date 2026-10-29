"""Discord notifications: one notify(event) function, a typed event and a template per event.

At 40 batches a day a message per batch would be noise, so routine progress goes into one daily
digest. A missing webhook logs instead of failing.
"""

from typing import Literal

import httpx
import structlog
from pydantic import BaseModel, ConfigDict

log = structlog.get_logger(__name__)
DISCORD_LIMIT = 1900


class _Event(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Acceptance(_Event):
    kind: Literal["acceptance"] = "acceptance"
    factor_id: str
    family: str
    horizon: str
    tstat: float
    t_required: float
    worst_fold: float
    net_sharpe: float | None
    expression: str


class LoopPaused(_Event):
    kind: Literal["loop_paused"] = "loop_paused"
    reason: str


class GoalEnded(_Event):
    kind: Literal["goal_ended"] = "goal_ended"
    title: str
    why: str
    trials_used: int
    acceptances: int
    beliefs_changed: int


class FreezeSuggested(_Event):
    kind: Literal["freeze_suggested"] = "freeze_suggested"
    factors_changed: int
    last_library: str | None


class DailyDigest(_Event):
    kind: Literal["daily_digest"] = "daily_digest"
    trials_by_reason: dict[str, int]
    acceptances: int
    pool_size: int
    llm_spend_usd: float
    goal_progress: str


NotifyEvent = Acceptance | LoopPaused | GoalEnded | FreezeSuggested | DailyDigest


def render(e: NotifyEvent) -> str:
    match e:
        case Acceptance():
            sharpe = "n/a" if e.net_sharpe is None else f"{e.net_sharpe:.1f}"
            return (
                f"Accepted {e.factor_id} [{e.family}, {e.horizon}] t {e.tstat:.1f} "
                f"(req {e.t_required:.1f}), worst fold {e.worst_fold:.3f}, net Sharpe {sharpe}. "
                f"Expr: {e.expression}"
            )
        case LoopPaused():
            return f"Loop paused. Reason: {e.reason}"
        case GoalEnded():
            return (
                f"Goal ended: {e.title} ({e.why}). {e.trials_used} trials, {e.acceptances} accepted, "
                f"{e.beliefs_changed} beliefs changed. Approve the next goal with `af goal approve`."
            )
        case FreezeSuggested():
            last = e.last_library or "no library yet"
            return (
                f"The pool changed by {e.factors_changed} factors since {last}. "
                "Consider `af lib freeze`."
            )
        case DailyDigest():
            reasons = ", ".join(f"{k} {v}" for k, v in sorted(e.trials_by_reason.items())) or "none"
            return (
                f"Daily digest: trials by reason: {reasons}. Accepted today: {e.acceptances}. "
                f"Pool: {e.pool_size}. LLM spend: ${e.llm_spend_usd:.2f}. {e.goal_progress}"
            )


class Notifier:
    def __init__(
        self, webhook_url: str | None, transport: httpx.BaseTransport | None = None
    ) -> None:
        self.webhook_url = webhook_url
        self._transport = transport
        self.sent: list[str] = []  # what was rendered, for tests and logs

    def notify(self, event: NotifyEvent) -> None:
        text = render(event)[:DISCORD_LIMIT]
        self.sent.append(text)
        if not self.webhook_url:
            log.info("notify (no webhook configured)", message=text)
            return
        try:
            with httpx.Client(transport=self._transport, timeout=10) as client:
                client.post(self.webhook_url, json={"content": text}).raise_for_status()
        except httpx.HTTPError as e:  # a failed notification must never stop the loop
            log.warning("discord notification failed", error=str(e))
