"""Stage 1: text -> tree (Lark). Any grammar failure is INVALID_PARSE."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from lark import Lark, Token, Transformer
from lark.exceptions import LarkError

from alpha_factory.contracts.evaluation import ReasonCode
from alpha_factory.dsl.ast import Call, Field, Node, Num
from alpha_factory.dsl.errors import DSLError

MAX_CHARS = 1000  # far above the 300 the contracts allow; stops pathological inputs early
INFIX = {
    "+": "Add",
    "-": "Sub",
    "*": "Mul",
    "/": "Div",
    ">": "Gt",
    ">=": "Ge",
    "<": "Lt",
    "<=": "Le",
}


class _ToTree(Transformer[Token, Node]):
    def number(self, items: list[Any]) -> Node:
        return Num(float(items[0]))

    def field(self, items: list[Any]) -> Node:
        return Field(str(items[0])[1:])

    def call(self, items: list[Any]) -> Node:
        return Call(str(items[0]), tuple(items[1:]))

    def neg(self, items: list[Any]) -> Node:
        operand = items[1]
        if isinstance(operand, Num):  # "-1" is a literal, so it can be a window-free constant
            return Num(-operand.value)
        return Call("Neg", (operand,))

    def _fold(self, items: list[Any]) -> Node:
        node = items[0]
        for op, rhs in zip(items[1::2], items[2::2], strict=True):
            node = Call(INFIX[str(op)], (node, rhs))
        return node

    sum = product = _fold

    def expr(self, items: list[Any]) -> Node:
        return Call(INFIX[str(items[1])], (items[0], items[2]))


@lru_cache(maxsize=1)
def _parser() -> Lark:
    grammar = (Path(__file__).parent / "grammar.lark").read_text()
    return Lark(grammar, parser="lalr", maybe_placeholders=False)


def parse(text: str) -> Node:
    """Parse an expression string into a tree."""
    if len(text) > MAX_CHARS:
        raise DSLError(ReasonCode.TOO_COMPLEX, f"expression longer than {MAX_CHARS} characters")
    try:
        tree = _parser().parse(text)  # pyright: ignore[reportUnknownMemberType]
        return _ToTree().transform(tree)
    except LarkError as e:
        raise DSLError(ReasonCode.INVALID_PARSE, f"syntax error: {str(e).splitlines()[0]}") from e
    except RecursionError as e:
        raise DSLError(ReasonCode.TOO_COMPLEX, "expression nests too deeply") from e
