"""The gate's thresholds: the deflated t-stat bar and the data-driven IC floor."""

from collections.abc import Mapping
from statistics import NormalDist

from alpha_factory.contracts.base import Horizon
from alpha_factory.contracts.evaluation import GateSpec

ALPHA = 0.05


def required_tstat(n_trials: int, gate: GateSpec, alpha: float = ALPHA) -> float:
    """t_req(N) = max(t_floor, Phi^-1(1 - alpha / (2N))): the bar rises with the trials tested.

    N counts trials that reached the IC check under the current sandbox hash, this one included.
    A Bonferroni-style correction: conservative, because many trials are correlated.
    """
    if not gate.deflate_by_trials:
        return gate.min_ic_tstat
    n = max(n_trials, 1)
    # Phi^-1(1 - p) = -Phi^-1(p): the small-tail form stays exact for very large N
    return max(gate.min_ic_tstat, -NormalDist().inv_cdf(alpha / (2 * n)))


def ic_floor(gate: GateSpec, horizon: Horizon, null_floor: Mapping[str, float] | None) -> float:
    """min_ic_mean from the gate, else the null-calibrated 99th percentile for the horizon.

    With neither, the floor is 0: the t-stat bar alone decides.
    """
    if gate.min_ic_mean is not None:
        return gate.min_ic_mean
    return float((null_floor or {}).get(horizon, 0.0))
