"""Stage 2: names, argument kinds, types and size limits. Raises DSLError, returns nothing."""

from alpha_factory.contracts.evaluation import ReasonCode
from alpha_factory.dsl.ast import Call, Field, Node, Num, depth, n_nodes
from alpha_factory.dsl.errors import DSLError
from alpha_factory.dsl.registry import (
    FIELDS,
    MIN_WINDOW,
    REGISTRY,
    WINDOW_ONE_OK,
    WINDOWS,
)

MAX_DEPTH = 6
MAX_NODES = 30
MAX_CONSTANTS = 3


def _two_significant_digits(v: float) -> bool:
    return v == 0 or abs(float(f"{v:.2g}") - v) <= 1e-12 * max(1.0, abs(v))


def _fail(code: ReasonCode, msg: str) -> DSLError:
    return DSLError(code, msg)


def _walk(node: Node, want: str, consts: list[float]) -> None:
    """Check that `node` evaluates to a `want` ("float" or "bool"); collect float literals."""
    match node:
        case Num(v):
            if want != "float":
                raise _fail(ReasonCode.INVALID_PARSE, "a number cannot be used as a bool")
            if not _two_significant_digits(v):
                raise _fail(
                    ReasonCode.TOO_COMPLEX, f"constant {v} has more than 2 significant digits"
                )
            consts.append(v)
        case Field(name):
            if name not in FIELDS:
                raise _fail(ReasonCode.INVALID_PARSE, f"unknown field ${name}")
            if want != "float":
                raise _fail(ReasonCode.INVALID_PARSE, f"${name} is a float, expected a bool")
        case Call(name, args):
            spec = REGISTRY.get(name)
            if spec is None:
                raise _fail(ReasonCode.INVALID_PARSE, f"unknown operator {name}")
            if spec.returns != want:
                raise _fail(
                    ReasonCode.INVALID_PARSE, f"{name} returns {spec.returns}, expected {want}"
                )
            if len(args) != len(spec.args):
                raise _fail(
                    ReasonCode.INVALID_PARSE,
                    f"{name} takes {len(spec.args)} arguments, got {len(args)}",
                )
            for i, (kind, arg) in enumerate(zip(spec.args, args, strict=True)):
                if kind in ("float", "bool"):
                    _walk(arg, kind, consts)
                elif kind == "window":
                    _check_window(name, arg)
                else:
                    allowed = spec.choices[i] if i < len(spec.choices) else None
                    _check_const(name, arg, allowed)


def _check_window(op: str, arg: Node) -> None:
    if not isinstance(arg, Num) or arg.value != int(arg.value):
        raise _fail(ReasonCode.INVALID_PARSE, f"{op}: the window must be an integer literal")
    n = int(arg.value)
    if n not in WINDOWS:
        raise _fail(ReasonCode.INVALID_PARSE, f"{op}: window {n} is not one of {WINDOWS}")
    if n == 1 and op not in WINDOW_ONE_OK:
        raise _fail(ReasonCode.INVALID_PARSE, f"{op}: window 1 is only allowed for Ref, Delta, Pct")
    if n < MIN_WINDOW.get(op, 1 if op in WINDOW_ONE_OK else 2):
        raise _fail(ReasonCode.INVALID_PARSE, f"{op}: window {n} is too short")


def _check_const(op: str, arg: Node, allowed: tuple[float, ...] | None) -> None:
    if not isinstance(arg, Num):
        raise _fail(ReasonCode.INVALID_PARSE, f"{op}: this argument must be a numeric literal")
    if allowed is not None and arg.value not in allowed:
        raise _fail(ReasonCode.INVALID_PARSE, f"{op}: constant must be one of {allowed}")
    if not _two_significant_digits(arg.value):
        raise _fail(
            ReasonCode.TOO_COMPLEX, f"constant {arg.value} has more than 2 significant digits"
        )


def _has_field(node: Node) -> bool:
    if isinstance(node, Field):
        return True
    return isinstance(node, Call) and any(_has_field(a) for a in node.args)


def check(node: Node, want: str = "float") -> None:
    """Validate a tree. `want` is "float" for a factor and "bool" for a condition."""
    if depth(node) > MAX_DEPTH:
        raise _fail(ReasonCode.TOO_COMPLEX, f"depth {depth(node)} exceeds {MAX_DEPTH}")
    if n_nodes(node) > MAX_NODES:
        raise _fail(ReasonCode.TOO_COMPLEX, f"{n_nodes(node)} nodes exceed {MAX_NODES}")
    consts: list[float] = []
    _walk(node, want, consts)
    if len(consts) > MAX_CONSTANTS:
        raise _fail(ReasonCode.TOO_COMPLEX, f"{len(consts)} constants exceed {MAX_CONSTANTS}")
    if not _has_field(node):
        raise _fail(ReasonCode.INVALID_PARSE, "the expression uses no field")
