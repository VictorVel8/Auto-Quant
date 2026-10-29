"""The gate holdout: pass or fail, never a number.

Runs last and only for trials that passed every other check, which keeps holdout lookups rare.
The holdout IC is computed here and discarded: callers get a result whose reason is either
unchanged (ACCEPTED) or HOLDOUT_FAIL, and nothing else leaves this function.
"""

import polars as pl

from alpha_factory.contracts.evaluation import EvalResult, ReasonCode
from alpha_factory.contracts.sandbox import SandboxSpec
from alpha_factory.data.marketdata import MarketData
from alpha_factory.dsl import CompiledFactor
from alpha_factory.eval.context import EvalContext, build_context
from alpha_factory.eval.stats import daily_ic, intraday_ics


def holdout_passes(ic_holdout: float, ic_search: float, min_ratio: float) -> bool:
    """Same sign as the search IC, and at least `min_ratio` of it."""
    if ic_search <= 0:  # IC is already signed, so a non-positive search IC cannot be accepted
        return False
    return ic_holdout > 0 and ic_holdout >= min_ratio * ic_search


def holdout_ok(
    result: EvalResult,
    factor: CompiledFactor,
    sandbox: SandboxSpec,
    *,
    data: MarketData | None = None,
    context: EvalContext | None = None,
) -> bool:
    """Does the holdout agree with the search result? A bool: the IC itself is dropped."""
    ctx = context
    if ctx is None:
        if data is None:
            raise ValueError("holdout_ok needs either data or a context")
        ctx = build_context(data, sandbox, zone="holdout")
    if ctx.zone != "holdout":
        raise ValueError("the holdout check needs a holdout context")
    scores = (
        factor.run(ctx.panel.df)
        .filter(pl.col("session_date").is_between(ctx.start, ctx.end))
        .with_columns((pl.col("score") * result.sign_used).alias("score"))
    )
    ics = intraday_ics(scores, ctx.labels, f"y_{result.horizon}", ctx.min_names)
    daily = daily_ic(ics, 1)
    ic_holdout = float(daily["ic"].mean()) if daily.height else 0.0  # type: ignore[arg-type]
    return holdout_passes(ic_holdout, result.ic_mean, sandbox.gate.holdout_min_ic_ratio)


def gate_holdout(
    result: EvalResult,
    factor: CompiledFactor,
    sandbox: SandboxSpec,
    *,
    data: MarketData | None = None,
    context: EvalContext | None = None,
) -> EvalResult:
    """Return `result` unchanged if the holdout agrees, else the same result with HOLDOUT_FAIL.

    Only ACCEPTED results are checked.
    """
    if result.reason != ReasonCode.ACCEPTED:
        return result
    ok = holdout_ok(result, factor, sandbox, data=data, context=context)
    return result if ok else result.model_copy(update={"reason": ReasonCode.HOLDOUT_FAIL})
