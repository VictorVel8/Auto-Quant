"""Random valid expressions, for null calibration and Plan 4's template proposer.

Generation is by construction (types and argument kinds are respected), then the result is
checked; size limits can still fail, in which case the draw is repeated.
"""

import numpy as np

from alpha_factory.dsl import ast as A
from alpha_factory.dsl.canonical import canonicalise, to_string
from alpha_factory.dsl.checker import check
from alpha_factory.dsl.errors import DSLError
from alpha_factory.dsl.registry import FIELDS, MIN_WINDOW, REGISTRY, WINDOW_ONE_OK, WINDOWS

_GROUP_WEIGHT = {"ts": 0.45, "cs": 0.2, "elementwise": 0.2, "session": 0.05, "logic": 0.1}
_SMALL_CONSTANTS = (0.5, 1.0, 2.0, 5.0, 10.0)


def _ops(returns: str, depth: int) -> list[str]:
    """Operators that fit: one with a bool argument needs room for a comparison below it."""
    return [
        n
        for n, s in REGISTRY.items()
        if s.returns == returns
        and (depth >= 3 or "bool" not in s.args)
        and (returns == "float" or depth >= 2)
    ]


def _pick(rng: np.random.Generator, names: list[str]) -> str:
    w = np.array([_GROUP_WEIGHT[REGISTRY[n].group] for n in names])
    return names[int(rng.choice(len(names), p=w / w.sum()))]


def _window(rng: np.random.Generator, op: str) -> A.Num:
    lo = MIN_WINDOW.get(op, 1 if op in WINDOW_ONE_OK else 2)
    return A.Num(float(rng.choice([w for w in WINDOWS if w >= lo])))


def _gen(rng: np.random.Generator, want: str, depth: int, consts: list[int]) -> A.Node:
    if want == "float" and (depth <= 1 or rng.random() < 0.15):
        return A.Field(str(rng.choice(FIELDS)))
    name = _pick(rng, _ops(want, depth))
    spec = REGISTRY[name]
    args: list[A.Node] = []
    for i, kind in enumerate(spec.args):
        if kind in ("float", "bool"):
            if (
                kind == "float"
                and consts[0] < 2
                and spec.group == "elementwise"
                and rng.random() < 0.15
            ):
                consts[0] += 1
                args.append(A.Num(float(rng.choice(_SMALL_CONSTANTS))))
            else:
                args.append(_gen(rng, kind, depth - 1, consts))
        elif kind == "window":
            args.append(_window(rng, name))
        else:
            allowed = spec.choices[i] if i < len(spec.choices) else None
            args.append(A.Num(float(rng.choice(allowed if allowed else (-1.0, 1.0, 2.0)))))
    if name in ("Clip",):  # keep lo <= hi
        lo, hi = sorted(a.value for a in args[1:] if isinstance(a, A.Num))
        args[1], args[2] = A.Num(lo), A.Num(hi)
    return A.Call(name, tuple(args))


def random_expression(rng: np.random.Generator, max_depth: int = 4) -> str:
    """A random valid expression in canonical form, with depth at most `max_depth`."""
    for _ in range(100):
        tree = _gen(rng, "float", max_depth, [0])
        try:
            check(tree, "float")
        except DSLError:
            continue
        canon = canonicalise(tree)
        try:
            check(canon, "float")  # folding can create constants with too many digits
        except DSLError:
            continue
        return to_string(canon)
    raise RuntimeError("could not draw a valid random expression")
