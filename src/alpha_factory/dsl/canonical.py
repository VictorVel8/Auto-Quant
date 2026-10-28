"""Stage 3: canonical form. Constants fold, commutative arguments sort, one fixed print format."""

from alpha_factory.dsl.ast import Call, Field, Node, Num
from alpha_factory.dsl.registry import REGISTRY


def fmt_number(v: float) -> str:
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else format(v, ".12g")


def to_string(node: Node) -> str:
    match node:
        case Num(v):
            return fmt_number(v)
        case Field(name):
            return f"${name}"
        case Call(name, args):
            return f"{name}({', '.join(to_string(a) for a in args)})"


def _fold(name: str, vals: list[float]) -> float | None:
    match name, vals:
        case "Neg", [a]:
            return -a
        case "Add", [a, b]:
            return a + b
        case "Sub", [a, b]:
            return a - b
        case "Mul", [a, b]:
            return a * b
        case "Div", [a, b] if abs(b) >= 1e-12:
            return a / b
        case _:
            return None
    return None


def canonicalise(node: Node) -> Node:
    """Fold constants bottom-up, then sort the arguments of commutative operators."""
    if not isinstance(node, Call):
        return node
    args = tuple(canonicalise(a) for a in node.args)
    if all(isinstance(a, Num) for a in args):
        folded = _fold(node.name, [a.value for a in args if isinstance(a, Num)])
        if folded is not None:
            return Num(folded)
    spec = REGISTRY[node.name]
    if spec.commutative:
        slots = [i for i, k in enumerate(spec.args) if k in ("float", "bool")]
        ordered = sorted((args[i] for i in slots), key=to_string)
        args_list = list(args)
        for i, a in zip(slots, ordered, strict=True):
            args_list[i] = a
        args = tuple(args_list)
    return Call(node.name, args)


def windows_used(node: Node) -> tuple[int, ...]:
    """Distinct window literals, sorted."""
    found: set[int] = set()

    def walk(n: Node) -> None:
        if isinstance(n, Call):
            spec = REGISTRY[n.name]
            for kind, a in zip(spec.args, n.args, strict=True):
                if kind == "window" and isinstance(a, Num):
                    found.add(int(a.value))
                else:
                    walk(a)

    walk(node)
    return tuple(sorted(found))
