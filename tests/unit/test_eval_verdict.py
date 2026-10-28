import pytest

from alpha_factory.contracts.evaluation import GateSpec, ReasonCode
from alpha_factory.contracts.proposals import RefutationCheck
from alpha_factory.eval.gate import ic_floor, required_tstat
from alpha_factory.eval.holdout import holdout_passes
from alpha_factory.eval.verdict import (
    Checks,
    ic_verdict,
    later_failures,
    refutations,
    years_positive_fraction,
)

GATE = GateSpec()


@pytest.mark.parametrize(
    ("n", "expected"),
    [(1, 3.0), (10, 3.0), (100, 3.4808), (1_000, 4.0556), (10_000, 4.5648)],
)
def test_required_tstat_follows_the_plan_table(n: int, expected: float) -> None:
    # 3.0 (floor), about 3.5, about 4.1, about 4.6
    assert required_tstat(n, GATE) == pytest.approx(expected, abs=1e-3)


def test_deflation_can_be_switched_off_and_never_drops_below_the_floor() -> None:
    assert required_tstat(10_000, GateSpec(deflate_by_trials=False)) == 3.0
    assert required_tstat(0, GATE) == 3.0
    assert required_tstat(50, GateSpec(min_ic_tstat=5.0)) == 5.0


def test_ic_floor_prefers_the_gate_then_the_calibration() -> None:
    assert ic_floor(GateSpec(min_ic_mean=0.01), "h1", {"h1": 0.05}) == 0.01
    assert ic_floor(GateSpec(), "h1", {"h1": 0.05, "h2": 0.02}) == 0.05
    assert ic_floor(GateSpec(), "h4", {"h1": 0.05}) == 0.0
    assert ic_floor(GateSpec(), "h1", None) == 0.0


def test_ic_verdict_separates_low_ic_from_wrong_sign() -> None:
    assert ic_verdict(0.02, 4.0, 3.0, 0.01) == "ok"
    assert ic_verdict(0.02, 2.0, 3.0, 0.01) == "low_ic"  # right sign, not significant
    assert ic_verdict(0.005, 4.0, 3.0, 0.01) == "low_ic"  # significant, under the floor
    assert ic_verdict(-0.02, -4.0, 3.0, 0.01) == "wrong_sign"  # the reversed sign would pass
    assert ic_verdict(-0.005, -4.0, 3.0, 0.01) == "low_ic"  # reversed passes t but not the floor
    assert ic_verdict(0.0, 0.0, 3.0, 0.0) == "low_ic"


def good() -> Checks:
    return Checks(
        ic_worst_fold=0.01, ic_by_year={2022: 0.02, 2023: 0.02, 2024: 0.01, 2025: 0.02},
        incremental_ic=0.01, max_corr_pool=0.3,
        ic_by_regime={"vol_high": 0.02, "vol_low": 0.01, "trend_up": 0.02, "trend_down": 0.01},
        net_sharpe_val=1.0,
    )  # fmt: skip


def test_a_clean_factor_has_no_failures() -> None:
    assert later_failures(good(), GATE) == []


def test_each_later_check_fails_on_its_own() -> None:
    g = good()
    cases = {
        ReasonCode.UNSTABLE: Checks(**{**g.__dict__, "ic_worst_fold": -0.001}),
        ReasonCode.SUBSUMED: Checks(**{**g.__dict__, "incremental_ic": 0.002}),
        ReasonCode.CORRELATED: Checks(**{**g.__dict__, "max_corr_pool": 0.71}),
        ReasonCode.REGIME_ONLY: Checks(
            **{**g.__dict__, "ic_by_regime": {"vol_high": -0.001, "vol_low": 0.02}}
        ),
        ReasonCode.COST_KILLED: Checks(**{**g.__dict__, "net_sharpe_val": -0.1}),
    }  # fmt: skip
    for code, checks in cases.items():
        assert later_failures(checks, GATE) == [code]


def test_unstable_also_when_too_few_years_have_the_right_sign() -> None:
    years = {2022: 0.02, 2023: -0.01, 2024: -0.01, 2025: 0.02}  # 2 of 4 < 0.75
    assert years_positive_fraction(years) == 0.5
    assert later_failures(Checks(**{**good().__dict__, "ic_by_year": years}), GATE) == [
        ReasonCode.UNSTABLE
    ]
    three_of_four = {2022: 0.02, 2023: -0.01, 2024: 0.01, 2025: 0.02}
    assert later_failures(Checks(**{**good().__dict__, "ic_by_year": three_of_four}), GATE) == []


def test_precedence_first_failure_is_the_reason_the_rest_are_flags() -> None:
    bad = Checks(
        ic_worst_fold=-0.01, ic_by_year={2022: -0.01}, incremental_ic=0.0, max_corr_pool=0.9,
        ic_by_regime={"vol_high": -0.01}, net_sharpe_val=-1.0,
    )  # fmt: skip
    assert later_failures(bad, GATE) == [
        ReasonCode.UNSTABLE, ReasonCode.SUBSUMED, ReasonCode.CORRELATED,
        ReasonCode.REGIME_ONLY, ReasonCode.COST_KILLED,
    ]  # fmt: skip


def test_no_backtest_means_no_cost_verdict() -> None:
    assert later_failures(Checks(**{**good().__dict__, "net_sharpe_val": None}), GATE) == []


@pytest.mark.parametrize(
    ("holdout", "search", "ratio", "ok"),
    [(0.02, 0.03, 0.5, True), (0.014, 0.03, 0.5, False), (0.015, 0.03, 0.5, True),
     (-0.01, 0.03, 0.5, False), (0.0, 0.03, 0.5, False), (0.02, 0.0, 0.5, False)],
)  # fmt: skip
def test_holdout_needs_the_same_sign_and_half_the_search_ic(
    holdout: float, search: float, ratio: float, ok: bool
) -> None:
    assert holdout_passes(holdout, search, ratio) is ok


def refute(declared, **over):  # type: ignore[no-untyped-def]
    base = dict(
        reason=ReasonCode.ACCEPTED, flags=(), fold_means=[0.02, 0.01], ic_by_year={2022: 0.02},
        ic_by_horizon={"h1": 0.02, "h2": 0.015, "h4": 0.01}, horizon="h1", condition_ic_in=0.03,
        condition_ic_out=0.01, net_sharpe_val=1.0, parent_corr=0.5, evaluated=True,
    )  # fmt: skip
    out = refutations(tuple(declared), **{**base, **over})
    return {o.check: o.triggered for o in out}


def test_each_refutation_check() -> None:
    C = RefutationCheck
    assert refute([C.SUBSUMED_BY_BASELINE]) == {C.SUBSUMED_BY_BASELINE: False}
    assert refute([C.SUBSUMED_BY_BASELINE], reason=ReasonCode.SUBSUMED)[C.SUBSUMED_BY_BASELINE]
    assert refute([C.SUBSUMED_BY_BASELINE], flags=(ReasonCode.SUBSUMED,))[C.SUBSUMED_BY_BASELINE]
    assert refute([C.SIGN_UNSTABLE], fold_means=[0.02, -0.001])[C.SIGN_UNSTABLE]
    assert refute([C.SIGN_UNSTABLE], ic_by_year={2022: 0.02, 2023: -0.01})[C.SIGN_UNSTABLE]
    assert not refute([C.SIGN_UNSTABLE])[C.SIGN_UNSTABLE]
    assert not refute([C.NOT_CONCENTRATED_IN_CONDITION])[C.NOT_CONCENTRATED_IN_CONDITION]
    assert refute([C.NOT_CONCENTRATED_IN_CONDITION], condition_ic_in=0.01)[
        C.NOT_CONCENTRATED_IN_CONDITION
    ]
    assert refute([C.NOT_CONCENTRATED_IN_CONDITION], condition_ic_in=None)[
        C.NOT_CONCENTRATED_IN_CONDITION
    ]
    # another horizon more than 1.5x the declared one
    assert not refute([C.WEAK_AT_DECLARED_HORIZON])[C.WEAK_AT_DECLARED_HORIZON]
    assert refute([C.WEAK_AT_DECLARED_HORIZON], ic_by_horizon={"h1": 0.01, "h2": 0.02})[
        C.WEAK_AT_DECLARED_HORIZON
    ]
    assert not refute([C.WEAK_AT_DECLARED_HORIZON], ic_by_horizon={"h1": 0.01, "h2": 0.015})[
        C.WEAK_AT_DECLARED_HORIZON
    ]
    assert refute([C.COST_KILLED], net_sharpe_val=-0.01)[C.COST_KILLED]
    assert not refute([C.COST_KILLED], net_sharpe_val=None)[C.COST_KILLED]
    assert refute([C.CORRELATED_WITH_PARENT], parent_corr=0.81)[C.CORRELATED_WITH_PARENT]
    assert not refute([C.CORRELATED_WITH_PARENT], parent_corr=0.8)[C.CORRELATED_WITH_PARENT]


def test_checks_are_not_run_when_the_evaluation_stopped_early() -> None:
    out = refute([RefutationCheck.COST_KILLED], evaluated=False, net_sharpe_val=-9.0)
    assert out == {RefutationCheck.COST_KILLED: False}
