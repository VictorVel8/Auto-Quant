"""Expression tree. Infix operators are ordinary calls (Add, Sub, Mul, Div, Neg, Gt, ...)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Field:
    name: str  # without the "$"


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple["Node", ...]


Node = Num | Field | Call


def depth(node: Node) -> int:
    """Longest path from the root to a leaf, counting nodes (a lone field has depth 1)."""
    if isinstance(node, Call):
        return 1 + max((depth(a) for a in node.args), default=0)
    return 1


def n_nodes(node: Node) -> int:
    if isinstance(node, Call):
        return 1 + sum(n_nodes(a) for a in node.args)
    return 1
