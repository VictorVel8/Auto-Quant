"""Stage 4: canonical tree -> a list of steps, each materialising one column.

The plan compiles an expression into one polars expression. Nested window expressions turned out
to return silent nulls in the installed polars (an `.over("tix")` result fed into an
`.over("tsi")` rank), so every operator is its own step instead. Identical subtrees share a step,
and columns are dropped once no later step reads them.
"""

from dataclasses import dataclass

import polars as pl

from alpha_factory.contracts.evaluation import ReasonCode
from alpha_factory.dsl import ast as A
from alpha_factory.dsl.canonical import canonicalise, to_string, windows_used
from alpha_factory.dsl.checker import check
from alpha_factory.dsl.errors import DSLError
from alpha_factory.dsl.parser import parse
from alpha_factory.dsl.registry import FIELD_COLUMN, REGISTRY
from alpha_factory.hashing import sha256_canonical

KEY_COLUMNS = ("tix", "tsi", "sidx", "session_date", "bar", "ts", "is_member")


@dataclass(frozen=True)
class Step:
    name: str
    expr: pl.Expr
    inputs: frozenset[str]


@dataclass(frozen=True)
class CompiledFactor:
    expression: str  # canonical form
    condition: str | None  # canonical form
    expr_hash: str  # of the expression alone: the dedupe key
    full_hash: str  # expression and condition
    ast_depth: int
    n_nodes: int
    windows_used: tuple[int, ...]
    steps: tuple[Step, ...]
    score_column: str
    condition_column: str | None

    def run(self, panel: pl.DataFrame) -> pl.DataFrame:
        """Evaluate on a panel. Returns the key columns, `score` (null outside the universe) and,
        if a condition was given, `cond`."""
        df = panel
        last_use: dict[str, int] = {}
        for i, st in enumerate(self.steps):
            for c in st.inputs:
                last_use[c] = i
        keep = {self.score_column, self.condition_column, *KEY_COLUMNS}
        for i, st in enumerate(self.steps):
            df = df.with_columns(st.expr.alias(st.name))
            dead = [
                c
                for c, last in last_use.items()
                if last == i and c.startswith("_s") and c not in keep and c in df.columns
            ]
            if dead:
                df = df.drop(dead)
        out = [pl.col(c) for c in KEY_COLUMNS]
        out.append(pl.when(pl.col("is_member")).then(pl.col(self.score_column)).alias("score"))
        if self.condition_column is not None:
            out.append(
                pl.when(pl.col("is_member")).then(pl.col(self.condition_column)).alias("cond")
            )
        return df.select(out)


def _clean(e: pl.Expr) -> pl.Expr:
    """Infinities and NaN become null after every float operator."""
    return pl.when(e.is_finite()).then(e)


class _Builder:
    def __init__(self) -> None:
        self.steps: list[Step] = []
        self.by_key: dict[str, str] = {}

    def column(self, node: A.Node) -> str | None:
        """Column holding a node's value, or None for a literal."""
        match node:
            case A.Num():
                return None
            case A.Field(name):
                return FIELD_COLUMN.get(name, name)
            case A.Call(name, args):
                key = to_string(node)
                if key in self.by_key:
                    return self.by_key[key]
                spec = REGISTRY[name]
                call_args: list[pl.Expr | float | int] = []
                inputs: set[str] = set()
                for kind, arg in zip(spec.args, args, strict=True):
                    if kind in ("float", "bool"):
                        col = self.column(arg)
                        if col is None:
                            # Full-length column: some polars rolling kernels reject scalars.
                            value = arg.value if isinstance(arg, A.Num) else 0.0
                            call_args.append(pl.repeat(value, pl.len(), dtype=pl.Float64))
                        else:
                            call_args.append(pl.col(col))
                            inputs.add(col)
                    elif kind == "window":
                        call_args.append(int(arg.value) if isinstance(arg, A.Num) else 0)
                    else:
                        call_args.append(float(arg.value) if isinstance(arg, A.Num) else 0.0)
                expr = spec.impl(*call_args)
                if spec.returns == "float":
                    expr = _clean(expr)
                step = f"_s{len(self.steps)}"
                self.steps.append(Step(step, expr, frozenset(inputs)))
                self.by_key[key] = step
                return step


def compile_expr(expr: str, *, condition: str | None = None) -> CompiledFactor:
    """Parse, check, canonicalise and compile. Raises DSLError carrying a ReasonCode."""
    tree = parse(expr)
    check(tree, "float")
    cond_tree = parse(condition) if condition else None
    if cond_tree is not None:
        check(cond_tree, "bool")
    canon = canonicalise(tree)
    cond_canon = canonicalise(cond_tree) if cond_tree is not None else None
    # Folding can create a constant such as 10.5. The canonical text must itself be a valid
    # expression, because it is what gets stored and recompiled elsewhere.
    if not isinstance(canon, A.Num):
        check(canon, "float")
    if cond_canon is not None:
        check(cond_canon, "bool")
    if isinstance(canon, A.Num):
        raise DSLError(ReasonCode.INVALID_PARSE, "the expression folds to a constant")

    b = _Builder()
    score = b.column(canon)
    cond = b.column(cond_canon) if cond_canon is not None else None
    assert score is not None
    text = to_string(canon)
    cond_text = to_string(cond_canon) if cond_canon is not None else None
    return CompiledFactor(
        expression=text,
        condition=cond_text,
        expr_hash=sha256_canonical({"expr": text}),
        full_hash=sha256_canonical({"expr": text, "condition": cond_text}),
        ast_depth=A.depth(canon),
        n_nodes=A.n_nodes(canon),
        windows_used=windows_used(canon),
        steps=tuple(b.steps),
        score_column=score,
        condition_column=cond,
    )
