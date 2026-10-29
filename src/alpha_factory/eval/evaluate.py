"""evaluate(): one expression, one horizon, one sign -> an EvalResult with a reason code."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from time import perf_counter
from typing import Literal

import polars as pl

from alpha_factory.backtest.engine import rebalance_bars, simulate
from alpha_factory.contracts.base import Horizon
from alpha_factory.contracts.evaluation import EvalResult, ReasonCode, RefutationOutcome
from alpha_factory.contracts.proposals import RefutationCheck
from alpha_factory.contracts.sandbox import SandboxSpec
from alpha_factory.data.marketdata import MarketData
from alpha_factory.dsl import CompiledFactor
from alpha_factory.eval.context import EvalContext, build_context, min_condition_names
from alpha_factory.eval.gate import ic_floor, required_tstat
from alpha_factory.eval.incremental import incremental_ics
from alpha_factory.eval.pool import Pool, max_corr_with_pool, mean_rank_corr
from alpha_factory.eval.regimes import ic_by_regime
from alpha_factory.eval.stats import (
    by_year,
    daily_ic,
    fold_stats,
    icir,
    intraday_ics,
    series_stats,
)
from alpha_factory.eval.verdict import Checks, ic_verdict, later_failures, refutations
from alpha_factory.gitinfo import git_commit
from alpha_factory.ids import new_id


@dataclass(frozen=True)
class TrialInfo:
    """What the orchestrator knows about the trial; the evaluator only reads it."""

    trial_id: str | None = None
    proposal_id: str | None = None
    n_trials: int = 1  # trials that reached the IC check, this one included (sets the t bar)
    null_floor: Mapping[str, float] | None = None  # per-horizon IC floor from null calibration
    refutations: tuple[RefutationCheck, ...] = ()  # those the proposal declared
    parent_ids: tuple[str, ...] = ()
    code_commit: str | None = None


def _label(h: Horizon) -> str:
    return f"y_{h}"


def search_daily_ic(
    scores: pl.DataFrame, ctx: EvalContext, horizon: Horizon, sign: int
) -> pl.DataFrame:
    """Signed daily IC (session_date, ic) over the search folds."""
    ics = intraday_ics(scores, ctx.labels, _label(horizon), ctx.min_names)
    daily = daily_ic(ics, sign)
    in_folds = pl.any_horizontal(
        [pl.col("session_date").is_between(f.val.start, f.val.end) for f in ctx.sandbox.folds]
    )
    return daily.filter(in_folds)


def turnover(scores: pl.DataFrame, sandbox: SandboxSpec, min_names: int) -> float:
    """1 - mean Spearman correlation of ranks between consecutive rebalance times in a session."""
    bars = rebalance_bars(sandbox.portfolio.rebalance_times, sandbox.bar_minutes)
    corrs: list[float] = []
    for b0, b1 in pairwise(bars):
        a = scores.filter(pl.col("bar") == b0).select("tix", "session_date", "score")
        b = scores.filter(pl.col("bar") == b1).select(
            "tix", "session_date", pl.col("score").alias("y")
        )
        j = a.join(b, on=["tix", "session_date"]).drop_nulls()
        r = (
            j.with_columns(
                pl.col("score").rank("average").over("session_date").alias("rs"),
                pl.col("y").rank("average").over("session_date").alias("ry"),
            )
            .group_by("session_date")
            .agg(pl.len().alias("n"), pl.corr("rs", "ry").alias("c"))
            .filter((pl.col("n") >= min_names) & pl.col("c").is_finite())
        )
        # group_by returns groups in no fixed order and float addition is not associative, so
        # sort before summing (and sum exactly) for a result that is bit-for-bit reproducible
        corrs += r.sort("session_date")["c"].to_list()
    return 1.0 - math.fsum(corrs) / len(corrs) if corrs else 1.0


def evaluate(
    factor: CompiledFactor,
    horizon: Horizon,
    *,
    sign: Literal[1, -1],
    pool: Pool,
    sandbox: SandboxSpec,
    data: MarketData | None = None,
    context: EvalContext | None = None,
    trial: TrialInfo | None = None,
) -> EvalResult:
    """Score a factor on the search folds and assign one reason code.

    Pass `context` (from build_context) to reuse the panel and labels across a batch; otherwise
    one is built from `data`. Evaluation stops after LOW_IC and WRONG_SIGN; past that point every
    check runs so the flags are complete. The holdout is not touched here: see gate_holdout.
    """
    started = perf_counter()
    trial = trial or TrialInfo()
    if context is None:
        if data is None:
            raise ValueError("evaluate needs either data or a context")
        context = build_context(data, sandbox)
    ctx = context
    gate = sandbox.gate
    sandbox_hash = sandbox.content_hash()
    if sandbox_hash != ctx.sandbox.content_hash():
        raise ValueError("the context was built for a different sandbox")

    run = factor.run(ctx.panel.df)
    scores = run.filter(pl.col("session_date").is_between(ctx.start, ctx.end)).with_columns(
        (pl.col("score") * sign).alias("score")
    )
    daily = search_daily_ic(scores, ctx, horizon, 1)  # `scores` already carries the sign
    overall = series_stats(daily["ic"].to_numpy())
    folds = fold_stats(daily, ctx.sandbox.folds)
    worst = min((f.ic_mean for f in folds if f.n_days), default=0.0)
    years = by_year(daily)

    t_req = required_tstat(trial.n_trials, gate)
    floor = ic_floor(gate, horizon, trial.null_floor)
    verdict = ic_verdict(overall.mean, overall.tstat, t_req, floor)

    def finish(
        reason: ReasonCode,
        flags: tuple[ReasonCode, ...] = (),
        *,
        ic_by_horizon: dict[Horizon, float] | None = None,
        turnover_v: float = 0.0,
        max_corr: tuple[float, str | None] = (0.0, None),
        incremental: float = 0.0,
        regimes: dict[str, float] | None = None,
        cond: tuple[float | None, float | None] = (None, None),
        net_sharpe: float | None = None,
        outcomes: tuple[RefutationOutcome, ...] = (),
        refuted: bool = False,
    ) -> EvalResult:
        return EvalResult(
            trial_id=trial.trial_id or new_id("trl"),
            proposal_id=trial.proposal_id,
            expr_hash=factor.expr_hash,
            dataset_hash=ctx.dataset_hash,
            sandbox_hash=sandbox_hash,
            code_commit=trial.code_commit or git_commit(),
            horizon=horizon,
            sign_used=sign,
            folds=folds,
            ic_mean=overall.mean,
            ic_tstat=overall.tstat,
            ic_worst_fold=worst,
            icir=icir(daily["ic"].to_numpy()),
            ic_by_year=years,
            ic_by_horizon=ic_by_horizon or {horizon: overall.mean},
            turnover=turnover_v,
            max_corr_pool=max_corr[0],
            max_corr_pool_id=max_corr[1],
            incremental_ic=incremental,
            ic_by_regime=regimes or {},
            condition_ic_in=cond[0],
            condition_ic_out=cond[1],
            net_sharpe_val=net_sharpe,
            reason=reason,
            flags=flags,
            refutations=outcomes,
            hypothesis_refuted=refuted,
            runtime_s=perf_counter() - started,
        )

    if verdict != "ok":  # short-circuit: the declared checks that would follow are not run
        reason = ReasonCode.WRONG_SIGN if verdict == "wrong_sign" else ReasonCode.LOW_IC
        outcomes = refutations(
            trial.refutations, reason=reason, flags=(), fold_means=[], ic_by_year={},
            ic_by_horizon={}, horizon=horizon, condition_ic_in=None, condition_ic_out=None,
            net_sharpe_val=None, parent_corr=None, evaluated=False,
        )  # fmt: skip
        return finish(reason, outcomes=outcomes, refuted=verdict == "wrong_sign")

    # --- every remaining check runs -----------------------------------------------------------
    ic_by_h: dict[Horizon, float] = {horizon: overall.mean}
    for spec in ctx.sandbox.labels:
        if spec.name != horizon:
            d = search_daily_ic(scores, ctx, spec.name, 1)
            ic_by_h[spec.name] = float(d["ic"].mean()) if d.height else 0.0  # type: ignore[arg-type]

    inc = incremental_ics(scores, ctx.baselines, ctx.labels, f"d_{horizon}", ctx.min_names)
    incremental = float(daily_ic(inc, 1)["ic"].mean()) if inc.height else 0.0  # type: ignore[arg-type]

    corr, corr_id = max_corr_with_pool(scores, pool, ctx.panel.df, ctx.min_names)
    regimes = ic_by_regime(daily, ctx.regimes)

    cond_in = cond_out = None
    if factor.condition_column is not None and "cond" in run.columns:
        cond_min = min_condition_names(ctx.sandbox)
        flagged = scores.join(run.select("tix", "ts", "cond"), on=["tix", "ts"]).filter(
            pl.col("cond").is_not_null()
        )
        for want in (True, False):
            sub = flagged.filter(pl.col("cond") == want)
            d = daily_ic(intraday_ics(sub, ctx.labels, _label(horizon), cond_min), 1)
            value = float(d["ic"].mean()) if d.height else None  # type: ignore[arg-type]
            if want:
                cond_in = value
            else:
                cond_out = value

    sim = ctx.sim_inputs()
    ids = ctx.panel.ticker_ids()
    back = simulate(
        scores.join(ids, on="tix").select("ticker_id", "ts", "score"),
        sim.bars,
        sim.members,
        sim.sectors,
        ctx.sandbox,
    )
    net_sharpe = back.summary.net_sharpe if back.summary.n_days else None

    checks = Checks(worst, years, incremental, corr, regimes, net_sharpe)
    failing = later_failures(checks, gate)
    reason = failing[0] if failing else ReasonCode.ACCEPTED
    flags = tuple(failing[1:])

    parent_corr: float | None = None
    if RefutationCheck.CORRELATED_WITH_PARENT in trial.refutations:
        sub = scores.filter(pl.col("tsi") % 4 == 0)
        vals = [
            mean_rank_corr(
                sub, pool.scores(e, ctx.panel.df).filter(pl.col("tsi") % 4 == 0), ctx.min_names
            )
            for e in pool.entries
            if e.factor_id in trial.parent_ids
        ]
        parent_corr = max((abs(v) for v in vals if v is not None), default=None)

    outcomes = refutations(
        trial.refutations,
        reason=reason,
        flags=flags,
        fold_means=[f.ic_mean for f in folds],
        ic_by_year=years,
        ic_by_horizon=ic_by_h,
        horizon=horizon,
        condition_ic_in=cond_in,
        condition_ic_out=cond_out,
        net_sharpe_val=net_sharpe,
        parent_corr=parent_corr,
        evaluated=True,
    )
    return finish(
        reason,
        flags,
        ic_by_horizon=ic_by_h,
        turnover_v=turnover(scores, ctx.sandbox, ctx.min_names),
        max_corr=(corr, corr_id),
        incremental=incremental,
        regimes=regimes,
        cond=(cond_in, cond_out),
        net_sharpe=net_sharpe,
        outcomes=outcomes,
        refuted=any(o.triggered for o in outcomes),
    )
