"""The evaluator end to end on the synthetic market (planted reversal, IC 0.03 at h1)."""

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import Literal, cast

import numpy as np
import polars as pl
import pytest

from alpha_factory.contracts.base import Horizon
from alpha_factory.contracts.evaluation import GateSpec, ReasonCode
from alpha_factory.contracts.proposals import RefutationCheck
from alpha_factory.contracts.sandbox import FinalEvalRequest, SandboxSpec
from alpha_factory.data import MarketData
from alpha_factory.dsl import CompiledFactor, compile_expr
from alpha_factory.eval.calibration import (
    NullCalibration,
    label_shuffle_check,
    run_canary,
    run_null,
    shuffle_labels,
)
from alpha_factory.eval.context import EvalContext, build_context
from alpha_factory.eval.evaluate import TrialInfo, evaluate
from alpha_factory.eval.holdout import gate_holdout
from alpha_factory.eval.pool import Pool, PoolEntry
from alpha_factory.eval.regimes import regime_table
from alpha_factory.sandbox.final_eval import FinalEvalError, FinalEvalLedger, final_eval
from alpha_factory.settings import Settings
from alpha_factory.synthetic.dataset import synthetic_sandbox, write_datasets, zones
from alpha_factory.synthetic.market import SyntheticMarket

PLANTED = "-Pct($close, 2)"


@pytest.fixture(scope="module")
def world(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[SyntheticMarket, Path, MarketData]]:
    market = SyntheticMarket()
    root = tmp_path_factory.mktemp("eval")
    write_datasets(market, root)
    yield market, root, MarketData("dev", settings=Settings(data_dir=root))


@pytest.fixture(scope="module")
def sandbox(world: tuple[SyntheticMarket, Path, MarketData]) -> SandboxSpec:
    market, _, md = world
    return synthetic_sandbox(market, md.manifest.dataset_hash)


@pytest.fixture(scope="module")
def ctx(world: tuple[SyntheticMarket, Path, MarketData], sandbox: SandboxSpec) -> EvalContext:
    return build_context(world[2], sandbox)


def ev(
    expr: str,
    ctx: EvalContext,
    sign: Literal[1, -1] = 1,
    h: Horizon = "h1",
    pool: Pool | None = None,
    **kw,
):  # type: ignore[no-untyped-def]
    trial = kw.pop("trial", TrialInfo())
    factor = compile_expr(expr, condition=kw.pop("condition", None))
    return evaluate(
        factor, h, sign=sign, pool=pool or Pool(), sandbox=ctx.sandbox, context=ctx, trial=trial
    )  # type: ignore[arg-type]


# --- planted signal ----------------------------------------------------------------------------
def test_planted_ic_is_recovered_within_one_hundredth(ctx: EvalContext) -> None:
    r = ev(PLANTED, ctx)
    assert r.ic_mean == pytest.approx(0.03, abs=0.01)
    assert r.ic_tstat > 10
    assert all(f.ic_mean > 0.015 and f.n_days > 50 for f in r.folds) and len(r.folds) == 4
    assert r.ic_worst_fold == min(f.ic_mean for f in r.folds)
    assert (
        r.ic_by_horizon["h1"] > r.ic_by_horizon["h2"] > r.ic_by_horizon["h4"] > 0
    )  # decay profile
    assert set(r.ic_by_horizon) == {"h1", "h2", "h4"}
    assert r.sandbox_hash == ctx.sandbox.content_hash() and r.dataset_hash == ctx.dataset_hash
    assert {"vol_high", "vol_low", "trend_up", "trend_down"} == set(r.ic_by_regime)


def test_the_planted_factor_is_subsumed_by_the_one_bar_reversal_baseline(ctx: EvalContext) -> None:
    r = ev(PLANTED, ctx)
    assert r.reason == ReasonCode.SUBSUMED and r.incremental_ic < 0.003
    assert ReasonCode.COST_KILLED in r.flags  # turnover is ~1 and the spread eats the edge
    assert r.turnover > 0.9 and r.net_sharpe_val is not None and r.net_sharpe_val < 0


def test_a_baseline_has_no_incremental_ic_but_an_unrelated_signal_would(ctx: EvalContext) -> None:
    assert ev("-$ret", ctx).incremental_ic == 0.0  # exactly a baseline: nothing left over


def test_wrong_sign_is_recorded_and_the_flipped_trial_matches(ctx: EvalContext) -> None:
    wrong = ev("Pct($close, 2)", ctx, sign=1)
    assert wrong.reason == ReasonCode.WRONG_SIGN and wrong.hypothesis_refuted
    right = ev("Pct($close, 2)", ctx, sign=-1)
    assert right.ic_mean == pytest.approx(-wrong.ic_mean) and right.ic_tstat == pytest.approx(
        -wrong.ic_tstat
    )
    assert right.reason != ReasonCode.WRONG_SIGN and not right.hypothesis_refuted


def test_no_signal_is_low_ic_and_evaluation_stops_early(ctx: EvalContext) -> None:
    r = ev("$volume", ctx)
    assert r.reason == ReasonCode.LOW_IC and r.flags == () and not r.hypothesis_refuted
    assert r.net_sharpe_val is None and set(r.ic_by_horizon) == {"h1"} and r.ic_by_regime == {}


def test_the_bar_rises_with_the_trial_count_and_the_null_floor(ctx: EvalContext) -> None:
    assert ev(PLANTED, ctx, trial=TrialInfo(n_trials=1)).reason != ReasonCode.LOW_IC
    assert (
        ev(PLANTED, ctx, trial=TrialInfo(n_trials=10**60)).reason == ReasonCode.LOW_IC
    )  # t_req ~ 16
    assert ev(PLANTED, ctx, trial=TrialInfo(null_floor={"h1": 0.05})).reason == ReasonCode.LOW_IC


def test_evaluation_is_reproducible(ctx: EvalContext) -> None:
    a, b = ev(PLANTED, ctx), ev(PLANTED, ctx)
    drop = {"trial_id", "runtime_s"}
    assert a.model_dump(exclude=drop) == b.model_dump(exclude=drop)


def test_the_declared_refutations_are_computed_and_set_the_flag(ctx: EvalContext) -> None:
    decl = (
        RefutationCheck.COST_KILLED,
        RefutationCheck.SUBSUMED_BY_BASELINE,
        RefutationCheck.SIGN_UNSTABLE,
    )
    r = ev(PLANTED, ctx, trial=TrialInfo(refutations=decl))
    got = {o.check: o.triggered for o in r.refutations}
    assert got == {
        RefutationCheck.COST_KILLED: True,
        RefutationCheck.SUBSUMED_BY_BASELINE: True,
        RefutationCheck.SIGN_UNSTABLE: False,
    }
    assert r.hypothesis_refuted


def test_conditional_ic(ctx: EvalContext) -> None:
    r = ev(PLANTED, ctx, condition="$volume > Mean($volume, 26)")
    assert r.condition_ic_in is not None and r.condition_ic_out is not None
    assert r.condition_ic_in == pytest.approx(0.027, abs=0.012)  # volume is unrelated to the effect
    assert ev(PLANTED, ctx).condition_ic_in is None


# --- pool ---------------------------------------------------------------------------------------
def test_correlation_with_the_pool(ctx: EvalContext) -> None:
    twin = PoolEntry("fac_twin", compile_expr("Neg(Delta($close, 2))"))
    other = PoolEntry("fac_other", compile_expr("$volume"))
    pool = Pool((other, twin))
    r = ev(PLANTED, ctx, pool=pool)
    assert r.max_corr_pool_id == "fac_twin" and r.max_corr_pool > 0.8
    assert ReasonCode.CORRELATED in (r.reason, *r.flags)
    solo = ev(PLANTED, ctx, pool=Pool((other,)))
    assert solo.max_corr_pool < 0.1
    parent = ev(
        PLANTED,
        ctx,
        pool=pool,
        trial=TrialInfo(
            refutations=(RefutationCheck.CORRELATED_WITH_PARENT,), parent_ids=("fac_twin",)
        ),
    )
    assert parent.refutations[0].triggered


# --- accepted path and the holdout ---------------------------------------------------------------
@pytest.fixture(scope="module")
def lenient(world: tuple[SyntheticMarket, Path, MarketData], sandbox: SandboxSpec) -> EvalContext:
    """The same sandbox with a gate that lets the planted factor through."""
    gate = GateSpec(min_incremental_ic=-1.0, min_net_sharpe_val=-100.0, max_corr_pool=1.0)
    return build_context(world[2], sandbox.model_copy(update={"gate": gate}))


def test_a_factor_that_passes_everything_is_accepted_and_the_holdout_confirms_it(
    world: tuple[SyntheticMarket, Path, MarketData], lenient: EvalContext
) -> None:
    r = ev(PLANTED, lenient)
    assert r.reason == ReasonCode.ACCEPTED and r.flags == ()
    confirmed = gate_holdout(r, compile_expr(PLANTED), lenient.sandbox, data=world[2])
    assert confirmed.reason == ReasonCode.ACCEPTED and confirmed == r


def test_the_holdout_fails_a_factor_whose_effect_does_not_carry_over(
    world: tuple[SyntheticMarket, Path, MarketData], lenient: EvalContext
) -> None:
    r = ev(PLANTED, lenient)
    strict = lenient.sandbox.model_copy(
        update={"gate": lenient.sandbox.gate.model_copy(update={"holdout_min_ic_ratio": 5.0})}
    )
    out = gate_holdout(r, compile_expr(PLANTED), strict, data=world[2])
    assert out.reason == ReasonCode.HOLDOUT_FAIL
    assert out.model_dump(exclude={"reason"}) == r.model_dump(
        exclude={"reason"}
    )  # no holdout number
    assert (
        gate_holdout(ev("$volume", lenient), compile_expr("$volume"), strict, data=world[2]).reason
        == ReasonCode.LOW_IC
    )


def test_search_never_reads_holdout_rows(ctx: EvalContext, sandbox: SandboxSpec) -> None:
    last_search = max(f.val.end for f in sandbox.folds)
    assert cast(date, ctx.panel.df["session_date"].max()) <= last_search
    assert cast(date, ctx.labels["session_date"].max()) <= last_search
    assert ctx.end < sandbox.gate_holdout.start


def test_contexts_refuse_the_wrong_data(
    world: tuple[SyntheticMarket, Path, MarketData], sandbox: SandboxSpec
) -> None:
    _, root, md = world
    with pytest.raises(ValueError, match="different dataset"):
        build_context(md, sandbox.model_copy(update={"dataset_hash": "f" * 64}))
    with pytest.raises(PermissionError):
        build_context(MarketData("full", settings=Settings(data_dir=root), sealed=True), sandbox)


# --- regimes are causal --------------------------------------------------------------------------
def test_regime_states_do_not_depend_on_later_sessions(ctx: EvalContext) -> None:
    full = regime_table(ctx.panel.df)
    cut_date = full["session_date"][full.height - 40]
    early = regime_table(ctx.panel.df.filter(pl.col("session_date") <= cut_date))
    both = early.join(full, on="session_date", suffix="_full").drop_nulls("vol_high_full")
    assert both.height > 50
    assert both.filter(pl.col("vol_high") != pl.col("vol_high_full")).is_empty()
    assert both.filter(pl.col("trend_up") != pl.col("trend_up_full")).is_empty()


# --- calibration ---------------------------------------------------------------------------------
def test_null_calibration_shape_and_label_shuffle(ctx: EvalContext, tmp_path: Path) -> None:
    cal = run_null(ctx, n=8, seed=3)
    assert set(cal.ic_floor) == {"h1", "h2", "h4"} and all(v >= 0 for v in cal.ic_floor.values())
    assert cal.sandbox_hash == ctx.sandbox.content_hash() and cal.dataset_hash == ctx.dataset_hash
    path = tmp_path / "null.json"
    path.write_text(cal.model_dump_json())
    assert NullCalibration.model_validate_json(path.read_text()) == cal
    assert abs(label_shuffle_check(ctx, "h1", n_expressions=10, seed=2)) < 0.002


def test_shuffled_labels_keep_their_values_per_timestamp(ctx: EvalContext) -> None:
    s = shuffle_labels(ctx.labels, ["y_h1"], seed=1)
    a = ctx.labels.group_by("ts").agg(pl.col("y_h1").drop_nulls().sum().alias("s")).sort("ts")
    b = s.group_by("ts").agg(pl.col("y_h1").drop_nulls().sum().alias("s")).sort("ts")
    assert np.allclose(a["s"].to_numpy(), b["s"].to_numpy())
    assert not s.sort("ts", "tix")["y_h1"].equals(ctx.labels.sort("ts", "tix")["y_h1"])


def test_the_leakage_canary_is_flagged_by_the_causality_test(
    world: tuple[SyntheticMarket, Path, MarketData], ctx: EvalContext
) -> None:
    report = run_canary(ctx, world[2], floor=0.02)
    assert report.flagged_lookahead
    # Synthetic volume carries no information about returns, so the AQuA canary's IC stays low;
    # the forward-return canary leaks the label itself and must be flagged and score high.
    assert abs(report.ic) < 0.01
    assert report.leak_flagged_lookahead
    assert report.leak_scores_high
    assert report.passed


# --- final evaluation ----------------------------------------------------------------------------
class FactorLibrary:
    """A stand-in for a frozen library: one factor as the 'combined' signal."""

    version: str = "factor_lib_v1"
    horizon: Horizon = "h1"

    def __init__(self, factor: CompiledFactor) -> None:
        self.factor = factor

    def combiner_hash(self) -> str:
        return self.factor.expr_hash

    def score(self, panel: pl.DataFrame) -> pl.DataFrame:
        return self.factor.run(panel).select("tix", "ts", "session_date", "score")


def test_final_eval_runs_once_and_refuses_repeats_and_mismatches(
    world: tuple[SyntheticMarket, Path, MarketData], sandbox: SandboxSpec, tmp_path: Path
) -> None:
    _, root, _ = world
    lib = FactorLibrary(compile_expr(PLANTED))
    request = FinalEvalRequest(
        lib_version=lib.version,
        combiner_hash=lib.combiner_hash(),
        sandbox_hash=sandbox.content_hash(),
    )
    ledger = FinalEvalLedger(tmp_path / "ledger.jsonl")
    full = MarketData("full", settings=Settings(data_dir=root), sealed=True)

    report = final_eval(request, library=lib, ledger=ledger, sandbox=sandbox, data=full)
    assert report.request_hash == request.request_hash() and ledger.count() == 1
    assert report.ic_mean == pytest.approx(0.03, abs=0.012) and report.ic_tstat > 5
    assert set(report.net_sharpe_by_year) and report.cost_share_of_gross > 0

    with pytest.raises(FinalEvalError, match="already evaluated"):
        final_eval(request, library=lib, ledger=ledger, sandbox=sandbox, data=full)
    assert ledger.count() == 1
    assert (
        json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[0])["request_hash"]
        == report.request_hash
    )

    other = FinalEvalRequest(
        lib_version="factor_lib_v2",
        combiner_hash=lib.combiner_hash(),
        sandbox_hash=sandbox.content_hash(),
    )
    with pytest.raises(FinalEvalError, match="does not match"):
        final_eval(other, library=lib, ledger=ledger, sandbox=sandbox, data=full)
    stale = FinalEvalRequest(
        lib_version=lib.version, combiner_hash=lib.combiner_hash(), sandbox_hash="0" * 64
    )
    with pytest.raises(FinalEvalError, match="different sandbox"):
        final_eval(stale, library=lib, ledger=ledger, sandbox=sandbox, data=full)
    assert ledger.count() == 1


def test_final_eval_cannot_run_without_the_full_dataset(
    sandbox: SandboxSpec, tmp_path: Path
) -> None:
    lib = FactorLibrary(compile_expr(PLANTED))
    request = FinalEvalRequest(
        lib_version=lib.version,
        combiner_hash=lib.combiner_hash(),
        sandbox_hash=sandbox.content_hash(),
    )
    ledger = FinalEvalLedger(tmp_path / "ledger.jsonl")
    with pytest.raises(Exception, match=r"full|build"):  # no data dir here: MarketData refuses
        final_eval(request, library=lib, ledger=ledger, sandbox=sandbox)
    assert ledger.count() == 0


def test_the_sealed_window_is_outside_everything_the_search_can_see(
    sandbox: SandboxSpec, world: tuple[SyntheticMarket, Path, MarketData]
) -> None:
    z = zones(world[0])
    assert sandbox.sealed_test == z["sealed"]
    dev = world[2]
    assert max(dev.sessions()) < sandbox.sealed_test.start
