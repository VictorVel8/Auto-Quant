"""The template proposer: hand-written templates and mutations, no LLM.

It makes the whole loop work without a model and is later the baseline the LLM has to beat.
Everything is seeded by the batch id, so a batch can be reproduced exactly.
"""

from dataclasses import dataclass

import numpy as np

from alpha_factory.agents.proposer import ProposalOutcome
from alpha_factory.contracts import BatchDirective, Horizon, RawProposalBatch, ResearchGoal
from alpha_factory.dsl import DSLError, compile_expr, random_expression
from alpha_factory.dsl import ast as A
from alpha_factory.dsl.canonical import canonicalise, to_string
from alpha_factory.dsl.checker import check
from alpha_factory.dsl.parser import parse
from alpha_factory.dsl.registry import REGISTRY, WINDOWS
from alpha_factory.research.store import PoolRow, ResearchStore

# family -> (expression template, expected sign, one line of intuition). {n} {m} are windows.
TEMPLATES: dict[str, list[tuple[str, str, str]]] = {
    "short_term_reversal": [
        ("-Pct($close, {n})", "positive", "recent losers rebound as liquidity providers unwind"),
        ("-Rank(Pct($close, {n}))", "positive", "the weakest recent performers bounce back"),
        ("-ZScore(Delta($close, {n}))", "positive", "large recent moves partially revert"),
        ("-($close / Mean($close, {n}) - 1)", "positive", "price above its mean is pulled back"),
        ("-TsZScore($close, {n})", "positive", "stretched prices revert toward their own history"),
    ],
    "volume_shock": [
        (
            "Rank($volume / Mean($volume, {n})) * Sign($ret)",
            "positive",
            "abnormal volume confirms the direction of the move",
        ),
        (
            "SLog($volume / Mean($volume, {n})) * Sign(Pct($close, {m}))",
            "negative",
            "volume spikes mark exhaustion of a move",
        ),
        (
            "-Sign($ret) * ($volume / Mean($volume, {n}))",
            "positive",
            "heavy volume against a move signals reversal",
        ),
        ("Corr($volume, $close, {n})", "positive", "price rising with volume reflects real demand"),
    ],
    "overnight_gap": [
        ("$dayopen / $prevclose - 1", "negative", "opening gaps tend to fill during the morning"),
        ("-Rank($dayopen / $prevclose - 1)", "positive", "the largest gaps up are faded by midday"),
        (
            "($close - $dayopen) / $prevclose",
            "negative",
            "the part of the gap already filled predicts the rest",
        ),
        (
            "If($bar <= {k}, $dayopen / $prevclose - 1, 0 * $ret)",
            "negative",
            "gap effects are strongest early in the session",
        ),
    ],
    "momentum": [
        ("Pct($close, {n})", "positive", "winners keep winning over multi-day horizons"),
        (
            "Rank(Pct($close, {n})) - Rank(Pct($close, {m}))",
            "positive",
            "acceleration of recent trend predicts continuation",
        ),
        ("Decay($ret, {n})", "positive", "recent returns weighted toward the newest persist"),
    ],
    "intraday_trend": [
        ("Slope($close, {n})", "positive", "a steady intraday drift keeps going"),
        ("SessMean($ret)", "positive", "the session's average return so far persists"),
        (
            "SessSum($ret) / (Std($ret, {n}) + 0.001)",
            "positive",
            "risk-adjusted drift in the day predicts the next hours",
        ),
    ],
    "volatility": [
        ("-Std($ret, {n})", "positive", "calm stocks outperform turbulent ones intraday"),
        (
            "-($high - $low) / $close",
            "positive",
            "wide bars signal uncertainty that is later compensated",
        ),
    ],
}
CONDITIONS = [
    "$volume > Mean($volume, 130)",
    "$volume < Mean($volume, 130)",
    "$bar <= 3",
    "$bar >= 9",
    "Abs($ret) > Mean(Abs($ret), 65)",
    "Std($ret, 26) > Mean(Std($ret, 26), 130)",
]
SWAP_GROUPS = [("Mean", "EMA", "Decay"), ("Rank", "ZScore"), ("Min", "Max"), ("Std", "Skew")]
WRAPS = ("Rank", "SectorNeutral", "TsZScore")


@dataclass(frozen=True)
class Draw:
    """One proposal before it is turned into a dict."""

    type: str
    parents: list[str]
    family: str
    horizon: Horizon
    expression: str
    condition: str | None
    sign: str
    why: str


def _windows_in(node: A.Node) -> list[tuple[tuple[int, ...], int]]:
    """Paths to window literals: (path of arg indices, value)."""
    out: list[tuple[tuple[int, ...], int]] = []

    def walk(n: A.Node, path: tuple[int, ...]) -> None:
        if isinstance(n, A.Call):
            spec = REGISTRY[n.name]
            for i, (kind, a) in enumerate(zip(spec.args, n.args, strict=True)):
                if kind == "window" and isinstance(a, A.Num):
                    out.append(((*path, i), int(a.value)))
                else:
                    walk(a, (*path, i))

    walk(node, ())
    return out


def _replace(node: A.Node, path: tuple[int, ...], new: A.Node) -> A.Node:
    if not path:
        return new
    assert isinstance(node, A.Call)
    args = list(node.args)
    args[path[0]] = _replace(args[path[0]], path[1:], new)
    return A.Call(node.name, tuple(args))


def _swap_ops(node: A.Node, rng: np.random.Generator) -> A.Node:
    calls: list[tuple[int, ...]] = []

    def find(n: A.Node, path: tuple[int, ...]) -> None:
        if isinstance(n, A.Call):
            if any(n.name in g for g in SWAP_GROUPS):
                calls.append(path)
            for i, a in enumerate(n.args):
                find(a, (*path, i))

    find(node, ())
    if not calls:
        raise ValueError("nothing to swap")
    path = calls[int(rng.integers(len(calls)))]
    target: A.Node = node
    for i in path:
        assert isinstance(target, A.Call)
        target = target.args[i]
    assert isinstance(target, A.Call)
    group = next(g for g in SWAP_GROUPS if target.name in g)
    other = str(rng.choice([g for g in group if g != target.name]))
    return _replace(node, path, A.Call(other, target.args))


def mutate_expression(expr: str, rng: np.random.Generator) -> str:
    """One small change: a neighbouring window, a sibling operator, or a wrapper."""
    tree = parse(expr)
    for _ in range(30):
        choice = int(rng.integers(3))
        try:
            if choice == 0:
                wins = _windows_in(tree)
                path, value = wins[int(rng.integers(len(wins)))]
                i = WINDOWS.index(value)
                j = min(max(i + int(rng.choice([-1, 1])), 0), len(WINDOWS) - 1)
                cand = _replace(tree, path, A.Num(float(WINDOWS[j])))
            elif choice == 1:
                cand = _swap_ops(tree, rng)
            else:
                wrap = str(rng.choice(WRAPS))
                args = (
                    (tree, A.Num(float(rng.choice([5, 13, 26])))) if wrap == "TsZScore" else (tree,)
                )
                cand = A.Call(wrap, args)
            cand = canonicalise(cand)
            check(cand, "float")
        except (ValueError, IndexError, DSLError):
            continue
        text = to_string(cand)
        if text != to_string(canonicalise(tree)):
            return text
    raise ValueError("no valid mutation found")


class TemplateProposer:
    def __init__(self, seen: set[str] | None = None) -> None:
        self.seen = set(seen or ())

    def propose(
        self, directive: BatchDirective, goal: ResearchGoal, store: ResearchStore, rng_seed: int
    ) -> ProposalOutcome:  # fmt: skip
        rng = np.random.default_rng(rng_seed)
        pool = store.pool()
        seen = self.seen | store.proposed_hashes()
        draws: list[Draw] = []
        families = list(dict.fromkeys([*goal.families, *directive.explore_families])) or [
            "momentum"
        ]
        for _ in range(directive.n_new):
            draws.append(self._new(rng, goal, families, len(draws) % 2 == 0, seen))
        for _ in range(directive.n_mutate):
            draws.append(
                self._derive(rng, pool, "mutate", seen)
                or self._new(rng, goal, families, True, seen)
            )
        for _ in range(directive.n_refine):
            draws.append(
                self._derive(rng, pool, "refine", seen)
                or self._new(rng, goal, families, True, seen)
            )
        for _ in range(directive.n_combine):
            draws.append(
                self._combine(rng, pool, seen) or self._new(rng, goal, families, True, seen)
            )
        self.seen |= seen  # later batches from this proposer never repeat these
        return ProposalOutcome(
            raw=RawProposalBatch(
                rationale="Template batch.", proposals=[self._item(d) for d in draws]
            )
        )

    # -- new ---------------------------------------------------------------------------------
    def _new(self, rng: np.random.Generator, goal: ResearchGoal, families: list[str], use_template: bool, seen: set[str]) -> Draw:  # fmt: skip
        horizon: Horizon = goal.horizons[int(rng.integers(len(goal.horizons)))]
        for _ in range(60):
            family = families[int(rng.integers(len(families)))]
            if use_template and family in TEMPLATES:
                tpl, sign, why = TEMPLATES[family][int(rng.integers(len(TEMPLATES[family])))]
                expr = tpl.format(
                    n=int(rng.choice([2, 3, 5, 8, 13, 26])),
                    m=int(rng.choice([5, 13, 26, 65])),
                    k=int(rng.choice([2, 3, 5])),
                )
                text = f"Template for {family}: {why}."
            else:
                expr = random_expression(rng, 4)
                sign = str(rng.choice(["positive", "negative"]))
                text = f"Random expression in {family}; its sign is a guess to be tested."
            try:
                compiled = compile_expr(expr)
            except DSLError:
                continue
            if compiled.expr_hash not in seen:
                seen.add(compiled.expr_hash)
                return Draw("new", [], family, horizon, compiled.expression, None, sign, text)
        return Draw(
            "new",
            [],
            families[0],
            horizon,
            random_expression(rng, 4),
            None,
            "positive",
            "Fallback random expression.",
        )

    # -- mutate / refine -----------------------------------------------------------------------
    def _derive(self, rng: np.random.Generator, pool: list[PoolRow], kind: str, seen: set[str]) -> Draw | None:  # fmt: skip
        if not pool:
            return None
        for _ in range(40):
            parent = pool[int(rng.integers(len(pool)))]
            sign = "positive" if parent.sign_used > 0 else "negative"
            if kind == "mutate":
                try:
                    expr = mutate_expression(parent.canonical_expr, rng)
                    compiled = compile_expr(expr)
                except (ValueError, DSLError):
                    continue
                if compiled.expr_hash in seen:
                    continue
                seen.add(compiled.expr_hash)
                return Draw(
                    "mutate",
                    [parent.factor_id],
                    parent.family,
                    parent.horizon,
                    compiled.expression,
                    None,
                    sign,
                    f"Mutation of {parent.factor_id}: one small change to its expression.",
                )
            cond = CONDITIONS[int(rng.integers(len(CONDITIONS)))]
            try:
                compiled = compile_expr(parent.canonical_expr, condition=cond)
            except DSLError:
                continue
            key = compiled.full_hash
            if key in seen:
                continue
            seen.add(key)
            return Draw(
                "refine",
                [parent.factor_id],
                parent.family,
                parent.horizon,
                parent.canonical_expr,
                compiled.condition,
                sign,
                f"Refinement of {parent.factor_id}: test where its effect is strongest.",
            )
        return None

    # -- combine -------------------------------------------------------------------------------
    def _combine(self, rng: np.random.Generator, pool: list[PoolRow], seen: set[str]) -> Draw | None:  # fmt: skip
        if len(pool) < 2:
            return None
        for _ in range(40):
            i, j = (int(x) for x in rng.choice(len(pool), 2, replace=False))
            a, b = pool[i], pool[j]
            if a.family == b.family and len({p.family for p in pool}) > 1:
                continue  # different families stand in for different correlation clusters
            try:
                compiled = compile_expr(self._combine_text(a, b))
            except DSLError:
                continue
            if compiled.expr_hash in seen:
                continue
            seen.add(compiled.expr_hash)
            return Draw(
                "combine",
                [a.factor_id, b.factor_id],
                a.family,
                a.horizon,
                compiled.expression,
                None,
                "positive",
                f"Combination of {a.factor_id} and {b.factor_id} from different families.",
            )
        return None

    @staticmethod
    def _combine_text(a: PoolRow, b: PoolRow) -> str:
        def leg(p: PoolRow) -> str:
            core = f"Rank({p.canonical_expr})"
            return core if p.sign_used > 0 else f"Neg({core})"

        return f"Add({leg(a)}, {leg(b)})"

    @staticmethod
    def _item(d: Draw) -> dict[str, object]:
        return {
            "type": d.type,
            "parents": d.parents,
            "family": d.family,
            "horizon": d.horizon,
            "hypothesis": f"{d.why[0].upper()}{d.why[1:]}"[:400].ljust(20, "."),
            "mechanism": f"Mechanism for {d.family}: {d.why}"[:400],
            "expected_sign": d.sign,
            "expression": d.expression,
            "condition": d.condition,
            "refutations": ["cost_killed", "subsumed_by_baseline"],
            "novelty": f"{d.type} proposal from the template proposer.",
        }
