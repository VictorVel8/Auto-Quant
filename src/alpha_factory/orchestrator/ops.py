"""Operations on stored trials: re-evaluation from the stored inputs."""

from dataclasses import dataclass

from alpha_factory.compute.specs import EvalBatchRequest, EvalSpec
from alpha_factory.compute.worker import EvalBackend
from alpha_factory.contracts import EvalResult, SandboxSpec
from alpha_factory.research.store import ResearchStore

IGNORED = {"trial_id", "runtime_s", "code_commit", "proposal_id"}


@dataclass(frozen=True)
class Reevaluation:
    stored: EvalResult
    recomputed: EvalResult
    differences: dict[str, tuple[object, object]]

    @property
    def identical(self) -> bool:
        return not self.differences


def reevaluate(
    store: ResearchStore,
    backend: EvalBackend,
    sandbox: SandboxSpec,
    dataset_hash: str,
    trial_id: str,
) -> Reevaluation:
    """Re-run a stored trial from its stored inputs: the proposal, the sign, the sandbox and data
    hashes, the trial count N it was judged under, and the pool as it stood at that moment."""
    trial = store.get_trial(trial_id)
    stored_eval = store.get_eval(trial_id)
    if stored_eval is None or trial.proposal_id is None:
        raise ValueError(f"{trial_id} was never evaluated")
    stored, n_at_eval, _ = stored_eval
    prop = store.get_proposal(trial.proposal_id)
    pool = store.pool(at=trial.created_at)
    by_id = {p.factor_id: p.canonical_expr for p in pool}
    sign = trial.sign_used or 1
    spec = EvalSpec(
        trial_key=f"{prop.proposal_id}|{prop.proposal.horizon}|{sign}",
        canonical_expr=prop.canonical_expr,
        condition=prop.proposal.condition,
        horizon=prop.proposal.horizon,
        sign=1 if sign > 0 else -1,
        parent_exprs=tuple(by_id[x] for x in prop.proposal.parents if x in by_id),
        refutations=tuple(prop.proposal.refutations),
    )
    request = EvalBatchRequest(
        sandbox_hash=sandbox.content_hash(),
        dataset_hash=dataset_hash,
        code_commit=stored.code_commit,
        n_evaluated=n_at_eval - 1,  # the worker adds the batch size (1) back
        pool=tuple(p.lib_factor() for p in pool),
        specs=(spec,),
    )
    (outcome,) = backend.evaluate(request).outcomes
    if outcome.result is None:
        raise RuntimeError(outcome.error)
    a = stored.model_dump(exclude=IGNORED)
    b = outcome.result.model_dump(exclude=IGNORED)
    diff = {k: (a[k], b[k]) for k in a if a[k] != b[k]}
    return Reevaluation(stored, outcome.result, diff)
