"""EvalFeedback: the rounded projection the LLM sees. Built by code only, from stored results.

The LLM never sees raw metrics. Notes are templated per reason code, with numbers rounded the same
way as the feedback fields (IC to 0.005, correlation to 0.05).
"""

import json

from alpha_factory.contracts import EvalFeedback, EvalResult, Horizon, ReasonCode, SandboxSpec
from alpha_factory.eval.gate import required_tstat
from alpha_factory.research.store import ResearchStore, TrialRow


def round_ic(x: float) -> float:
    return round(round(x / 0.005) * 0.005, 3)


def round_corr(x: float) -> float:
    return round(round(x / 0.05) * 0.05, 2)


def years_positive(r: EvalResult) -> str:
    n = len(r.ic_by_year)
    return f"{sum(v > 0 for v in r.ic_by_year.values())}/{n}"


def _note_fields(t: TrialRow) -> dict[str, str | None]:
    return json.loads(t.note) if t.note else {}


def note_for(
    t: TrialRow, r: EvalResult | None, n_at_eval: int, sandbox: SandboxSpec, store: ResearchStore
) -> str:
    """One line explaining a verdict, from a template per reason."""
    reason = t.reason
    if (
        r is None
    ):  # invalid_schema, invalid_parse, too_complex, duplicate: the checker's own message
        return str(_note_fields(t).get("message") or reason.value)
    match reason:
        case ReasonCode.LOW_IC:
            t_req = required_tstat(n_at_eval, sandbox.gate)
            return f"t-stat {r.ic_tstat:.1f} vs required {t_req:.1f}"
        case ReasonCode.WRONG_SIGN:
            return "the reversed sign passes; a sign-flip trial was run"
        case ReasonCode.UNSTABLE:
            bad = next((f for f in r.folds if f.ic_mean < 0), None)
            if bad is not None:
                return f"{bad.fold_id} has the opposite sign"
            n = len(r.ic_by_year)
            return f"{sum(v > 0 for v in r.ic_by_year.values())} of {n} years with the right sign"
        case ReasonCode.SUBSUMED:
            return f"incremental IC {round_ic(r.incremental_ic):.3f} over the six baselines"
        case ReasonCode.CORRELATED:
            return f"correlation {round_corr(r.max_corr_pool):.2f} with {r.max_corr_pool_id}"
        case ReasonCode.REGIME_ONLY:
            state = next((k for k, v in sorted(r.ic_by_regime.items()) if v < 0), "a regime")
            return f"wrong sign in {state}"
        case ReasonCode.COST_KILLED:
            return "negative net Sharpe after costs"
        case ReasonCode.HOLDOUT_FAIL:
            return "failed the holdout check"  # no number, by design
        case ReasonCode.ACCEPTED:
            fid = next(
                (p.factor_id for p in store.pool() if p.trial_id == t.trial_id), "a pool factor"
            )
            return f"accepted as {fid}"
        case _:
            return reason.value


def feedback_for(t: TrialRow, sandbox: SandboxSpec, store: ResearchStore) -> EvalFeedback:
    ev = store.get_eval(t.trial_id)
    horizon: Horizon = t.horizon if t.horizon in ("h1", "h2", "h4") else "h1"  # type: ignore[assignment]
    family = t.family or "unknown"
    if ev is None:
        expression = str(_note_fields(t).get("expression") or "")
        return EvalFeedback(
            trial_id=t.trial_id, family=family, horizon=horizon, expression=expression[:300],
            reason=t.reason, ic=0.0, ic_worst_fold=0.0, years_positive="0/0", corr_to_pool=0.0,
            note=note_for(t, None, 0, sandbox, store),
        )  # fmt: skip
    result, n, _ = ev
    expression = store.get_proposal(t.proposal_id).canonical_expr if t.proposal_id else ""
    return EvalFeedback(
        trial_id=t.trial_id, family=family, horizon=horizon, expression=expression,
        reason=t.reason, ic=round_ic(result.ic_mean), ic_worst_fold=round_ic(result.ic_worst_fold),
        years_positive=years_positive(result), corr_to_pool=round_corr(result.max_corr_pool),
        note=note_for(t, result, n, sandbox, store),
    )  # fmt: skip
