import numpy as np
import pytest

from alpha_factory.contracts.evaluation import ReasonCode
from alpha_factory.dsl import DSLError, compile_expr, random_expression
from alpha_factory.dsl.registry import REGISTRY, WINDOWS, render_docs


def reason(expr: str, **kw: str) -> ReasonCode:
    with pytest.raises(DSLError) as e:
        compile_expr(expr, **kw)
    return e.value.reason


@pytest.mark.parametrize(
    ("text", "canonical"),
    [
        ("-Pct($close, 2)", "Neg(Pct($close, 2))"),
        ("$volume / Mean($volume, 130)", "Div($volume, Mean($volume, 130))"),
        ("$dayopen / $prevclose - 1", "Sub(Div($dayopen, $prevclose), 1)"),
        ("$close * $open", "Mul($close, $open)"),
        ("$open * $close", "Mul($close, $open)"),  # commutative arguments sort
        ("Corr($volume, $close, 13)", "Corr($close, $volume, 13)"),  # window stays last
        ("$close * (2 + 3)", "Mul($close, 5)"),  # constants fold
        ("Clip($ret, -1, 1)", "Clip($ret, -1, 1)"),  # "-1" is a literal
        ("  Rank(\n$ret )", "Rank($ret)"),  # whitespace
        ("If($close > $open, $ret, -$ret)", "If(Gt($close, $open), $ret, Neg($ret))"),
    ],
)
def test_canonical_form(text: str, canonical: str) -> None:
    assert compile_expr(text).expression == canonical


def test_equivalent_spellings_share_one_hash_and_different_expressions_do_not() -> None:
    a, b = compile_expr("$open + $close"), compile_expr("Add($close,$open)")
    assert a.expr_hash == b.expr_hash
    assert a.expr_hash != compile_expr("$open - $close").expr_hash


def test_condition_is_part_of_the_full_hash_only() -> None:
    a = compile_expr("$ret")
    b = compile_expr("$ret", condition="$volume > $trades")
    assert a.expr_hash == b.expr_hash and a.full_hash != b.full_hash
    assert b.condition == "Gt($trades, $volume)".replace(
        "Gt($trades, $volume)", "Gt($volume, $trades)"
    )


@pytest.mark.parametrize(
    "bad",
    [
        "$close +",  # grammar
        "Pct($close, 2",  # unbalanced
        "pct($close, 2)",  # lowercase name
        "Nope($close)",  # unknown operator
        "$nope",  # unknown field
        "Pct($close)",  # arity
        "Pct($close, 4)",  # 4 is not an allowed window
        "Std($close, 1)",  # window 1 only for Ref, Delta, Pct
        "Pct($close, $open)",  # window must be a literal
        "Winsor($ret, 0.1)",  # q must be 0.01 or 0.05
        "SignedPower($ret, 3)",  # p must be 0.5 or 2
        "If($close, $ret, $open)",  # condition must be bool
        "$close > $open",  # a factor must be float
        "Rank($close > $open)",  # float expected
        "2 + 3",  # no field
    ],
)
def test_invalid_parse(bad: str) -> None:
    assert reason(bad) == ReasonCode.INVALID_PARSE


@pytest.mark.parametrize(
    "bad",
    [
        "Mean(Mean(Mean(Mean(Mean(Mean(Mean($close, 2), 2), 2), 2), 2), 2), 2)",  # depth 8
        "$close * 1.234",  # more than 2 significant digits
        "$close + 1 + 2 + 3 + 4",  # four constants
        " + ".join(["$close"] * 16),  # more than 30 nodes
        "$close + " * 200 + "$open",  # longer than 1000 characters
    ],
)
def test_too_complex(bad: str) -> None:
    assert reason(bad) == ReasonCode.TOO_COMPLEX


def test_a_condition_must_be_bool() -> None:
    assert reason("$ret", condition="$close") == ReasonCode.INVALID_PARSE
    assert compile_expr("$ret", condition="And($close > $open, $volume > $trades)").condition


def test_folded_constants_must_still_be_valid() -> None:
    # 10 + 0.5 folds to 10.5, which would not re-parse as a legal constant
    assert reason("$close * (10 + 0.5)") == ReasonCode.TOO_COMPLEX


def test_canonical_text_recompiles_to_itself() -> None:
    rng = np.random.default_rng(1)
    for _ in range(200):
        text = random_expression(rng, 4)
        again = compile_expr(text)
        assert again.expression == text
        assert compile_expr(again.expression).expr_hash == again.expr_hash


def test_random_expressions_are_valid_varied_and_bounded() -> None:
    rng = np.random.default_rng(5)
    drawn = [compile_expr(random_expression(rng, 4)) for _ in range(300)]
    assert all(c.ast_depth <= 4 for c in drawn)
    assert len({c.expr_hash for c in drawn}) > 250
    used = {tok for c in drawn for tok in REGISTRY if f"{tok}(" in c.expression}
    assert len(used) >= 25  # a broad slice of the operator set


def test_windows_used_and_metadata() -> None:
    c = compile_expr("Corr(Mean($close, 8), $volume, 26) - Pct($open, 1)")
    assert c.windows_used == (1, 8, 26)
    assert c.ast_depth == 4 and c.n_nodes == 10


def test_registry_is_consistent_and_documented() -> None:
    assert WINDOWS == (1, 2, 3, 5, 8, 13, 26, 39, 65, 130)
    docs = render_docs()
    for name, spec in REGISTRY.items():
        if spec.infix is None:
            assert f"{name}(" in docs
        assert spec.doc
    assert "$prevclose" in docs


def test_redundant_parentheses_are_harmless() -> None:
    assert compile_expr("((($close)))").expression == "$close"
