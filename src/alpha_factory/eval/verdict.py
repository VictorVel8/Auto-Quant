"""Verdict precedence and refutation checks.

Each trial gets exactly one reason: the first failing check in plan order. Later failures are
flags. Check 6 (LOW_IC) and 7 (WRONG_SIGN) are one decision: when the declared sign fails check 6
but the reversed sign would pass it, the verdict is WRONG_SIGN rather than LOW_IC.
"""

from dataclasses import dataclass
from typing import Literal

from alpha_factory.contracts.base import Horizon
from alpha_factory.contracts.evaluation import GateSpec, ReasonCode, RefutationOutcome
from alpha_factory.contracts.proposals import RefutationCheck

CORRELATED_WITH_PARENT = 0.8
WEAK_HORIZON_RATIO = 1.5


def passes_ic(ic_mean: float, tstat: float, t_req: float, floor: float) -> bool:
    """Check 6: the t-stat clears the (deflated) bar and the mean IC clears the floor."""
    return tstat >= t_req and ic_mean >= floor


def ic_verdict(
    ic_mean: float, tstat: float, t_req: float, floor: float
) -> Literal["ok", "low_ic", "wrong_sign"]:
    if passes_ic(ic_mean, tstat, t_req, floor):
        return "ok"
    return "wrong_sign" if passes_ic(-ic_mean, -tstat, t_req, floor) else "low_ic"


@dataclass(frozen=True)
class Checks:
    ic_worst_fold: float
    ic_by_year: dict[int, float]
    incremental_ic: float
    max_corr_pool: float
    ic_by_regime: dict[str, float]
    net_sharpe_val: float | None


def years_positive_fraction(ic_by_year: dict[int, float]) -> float:
    return sum(v > 0 for v in ic_by_year.values()) / len(ic_by_year) if ic_by_year else 0.0


def later_failures(c: Checks, gate: GateSpec) -> list[ReasonCode]:
    """Checks 8 to 12, in order. The first is the reason and the rest are flags."""
    out: list[ReasonCode] = []
    if (
        c.ic_worst_fold < gate.min_ic_worst_fold
        or years_positive_fraction(c.ic_by_year) < gate.min_years_positive_frac
    ):
        out.append(ReasonCode.UNSTABLE)
    if c.incremental_ic < gate.min_incremental_ic:
        out.append(ReasonCode.SUBSUMED)
    if c.max_corr_pool > gate.max_corr_pool:
        out.append(ReasonCode.CORRELATED)
    if any(v < 0 for v in c.ic_by_regime.values()):
        out.append(ReasonCode.REGIME_ONLY)
    if c.net_sharpe_val is not None and c.net_sharpe_val < gate.min_net_sharpe_val:
        out.append(ReasonCode.COST_KILLED)
    return out


def refutations(
    declared: tuple[RefutationCheck, ...],
    *,
    reason: ReasonCode,
    flags: tuple[ReasonCode, ...],
    fold_means: list[float],
    ic_by_year: dict[int, float],
    ic_by_horizon: dict[Horizon, float],
    horizon: Horizon,
    condition_ic_in: float | None,
    condition_ic_out: float | None,
    net_sharpe_val: float | None,
    parent_corr: float | None,
    evaluated: bool,
) -> tuple[RefutationOutcome, ...]:
    """Which declared hypothesis-refuting results actually occurred."""
    out: list[RefutationOutcome] = []
    for check in declared:
        triggered, detail = False, "not evaluated"
        if evaluated:
            match check:
                case RefutationCheck.SUBSUMED_BY_BASELINE:
                    triggered = ReasonCode.SUBSUMED in (reason, *flags)
                    detail = "incremental IC below the minimum"
                case RefutationCheck.SIGN_UNSTABLE:
                    triggered = any(m < 0 for m in fold_means) or any(
                        v < 0 for v in ic_by_year.values()
                    )
                    detail = "a fold or a year has the opposite sign"
                case RefutationCheck.NOT_CONCENTRATED_IN_CONDITION:
                    triggered = (
                        condition_ic_in is None
                        or condition_ic_out is None
                        or not condition_ic_in > condition_ic_out
                    )
                    detail = f"IC in {condition_ic_in} vs out {condition_ic_out}"
                case RefutationCheck.WEAK_AT_DECLARED_HORIZON:
                    mine = ic_by_horizon.get(horizon, 0.0)
                    triggered = any(
                        v > WEAK_HORIZON_RATIO * mine
                        for h, v in ic_by_horizon.items()
                        if h != horizon
                    )
                    detail = "another horizon's IC is more than 1.5x the declared one"
                case RefutationCheck.COST_KILLED:
                    triggered = net_sharpe_val is not None and net_sharpe_val < 0
                    detail = f"net Sharpe {net_sharpe_val}"
                case RefutationCheck.CORRELATED_WITH_PARENT:
                    triggered = parent_corr is not None and parent_corr > CORRELATED_WITH_PARENT
                    detail = f"correlation with the most similar parent {parent_corr}"
        out.append(RefutationOutcome(check=check, triggered=triggered, detail=detail))
    return tuple(out)
