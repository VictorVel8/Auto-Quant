"""No operator and no composition reads the future (Hypothesis).

For a random valid expression on a random synthetic panel, pick a time t, replace every row after t
(all tickers) with noise, and recompute: every value at or before t must be identical, nulls
included. Runs for each operator alone and for 500 random compositions.
"""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.helpers.panels import bar_rows, bars_frame

from alpha_factory.dsl import compile_expr, random_expression
from alpha_factory.dsl.registry import REGISTRY
from alpha_factory.sandbox.causality import causality_violations, perturb_after
from alpha_factory.sandbox.panel import Panel, bar_end

N_TICKERS = 7


def _sessions(n: int) -> dict[date, int]:
    """n weekday sessions, 13 bars each except a half-day (7 bars) on the sixth."""
    out: dict[date, int] = {}
    d = date(2024, 3, 4)
    while len(out) < n:
        if d.weekday() < 5:
            out[d] = 7 if len(out) == 5 else 13
        d += timedelta(days=1)
    return out


SESSION_BARS = _sessions(14)  # 170 bars: enough history for a 130-bar window
TS = [bar_end(d, b) for d, n in SESSION_BARS.items() for b in range(n)]


def random_market(seed: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Bars with ~0.5% gaps, a universe that changes by session, and three sectors."""
    rng = np.random.default_rng(seed)
    tickers = [f"t{i}" for i in range(N_TICKERS)]
    rows: list[dict[str, object]] = []
    for t in tickers:
        price = float(rng.uniform(20, 120))
        for d, n in SESSION_BARS.items():
            closes: list[float | None] = []
            for _ in range(n):
                price *= float(np.exp(rng.normal(0, 0.01)))
                closes.append(None if rng.random() < 0.005 else round(price, 4))
            if closes[0] is None or (t == tickers[0] and d == date(2024, 3, 8)):
                closes[0] = round(price, 4) if closes[0] is None else closes[0]
            if t == tickers[0] and d == date(2024, 3, 8):
                closes[4] = None  # one guaranteed gap
            rows += bar_rows(t, d, closes, volume=float(rng.integers(500, 5000)))
    members = [
        (d, t, r + 1)
        for d in SESSION_BARS
        for r, t in enumerate(rng.permutation(tickers)[: int(rng.integers(4, N_TICKERS + 1))])
    ]
    universe = pl.DataFrame(
        {
            "session_date": [m[0] for m in members],
            "ticker_id": [m[1] for m in members],
            "rank": pl.Series([m[2] for m in members], dtype=pl.Int32),
            "adv_usd_20d": [1e6] * len(members),
            "raw_prev_close": [50.0] * len(members),
            "tradable": [True] * len(members),
        }
    )
    sectors = pl.DataFrame(
        {"ticker_id": tickers, "sector": [f"s{int(rng.integers(0, 3))}" for _ in tickers]}
    )
    return bars_frame(rows), universe, sectors


def violations(expr: str, seed: int, cut: int, condition: str | None = None) -> int:
    bars, universe, sectors = random_market(seed)
    factor = compile_expr(expr, condition=condition)

    def feature(panel: Panel) -> pl.DataFrame:
        return factor.run(panel.df).select("tix", "ts", "score")

    return causality_violations(feature, bars, universe, sectors, SESSION_BARS, TS[cut], seed=seed)


EXAMPLES = {
    "Ref": "Ref($close, 3)", "Delta": "Delta($close, 2)", "Pct": "Pct($close, 5)",
    "Mean": "Mean($close, 8)", "Sum": "Sum($volume, 5)", "Std": "Std($ret, 13)",
    "Min": "Min($low, 5)", "Max": "Max($high, 5)", "Skew": "Skew($ret, 13)",
    "TsRank": "TsRank($close, 13)", "TsZScore": "TsZScore($close, 13)",
    "Corr": "Corr($close, $volume, 13)", "Cov": "Cov($close, $volume, 13)",
    "Decay": "Decay($close, 8)", "EMA": "EMA($close, 8)", "Slope": "Slope($close, 13)",
    "SessMean": "SessMean($ret)", "SessSum": "SessSum($volume)",
    "Rank": "Rank($ret)", "ZScore": "ZScore($ret)", "Demean": "Demean($ret)",
    "Winsor": "Winsor($ret, 0.05)", "SectorNeutral": "SectorNeutral($ret)",
    "Abs": "Abs($ret)", "Sign": "Sign($ret)", "SLog": "SLog($volume)",
    "SignedPower": "SignedPower($ret, 0.5)", "Greater": "Greater($open, $close)",
    "Less": "Less($open, $close)", "Clip": "Clip($ret, -1, 1)",
    "If": "If($close > $open, $ret, -$ret)",
    "And": "If(And($close > $open, $volume > $trades), $ret, -$ret)",
    "Or": "If(Or($close > $open, $volume > $trades), $ret, -$ret)",
    "Not": "If(Not($close > $open), $ret, -$ret)",
    "Add": "$open + $close", "Sub": "$open - $close", "Mul": "$open * $close",
    "Div": "$open / $close", "Neg": "-$close",
    "Gt": "If($open > $close, $ret, -$ret)", "Ge": "If($open >= $close, $ret, -$ret)",
    "Lt": "If($open < $close, $ret, -$ret)", "Le": "If($open <= $close, $ret, -$ret)",
    "dayopen": "$dayopen / $prevclose - 1",
}  # fmt: skip


def test_the_examples_cover_every_operator() -> None:
    assert set(REGISTRY) <= set(EXAMPLES)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_operator_alone_is_causal(name: str) -> None:
    for seed, cut in [(1, 5), (2, 40), (3, 90), (4, 120), (5, 150), (6, len(TS) - 2)]:
        assert violations(EXAMPLES[name], seed, cut) == 0, (name, seed, cut)


@settings(max_examples=500, deadline=None, derandomize=True)
@given(
    expr_seed=st.integers(0, 2**32 - 1),
    depth=st.integers(2, 5),
    market_seed=st.integers(0, 10_000),
    cut=st.integers(0, len(TS) - 2),
)
def test_random_compositions_are_causal(
    expr_seed: int, depth: int, market_seed: int, cut: int
) -> None:
    expr = random_expression(np.random.default_rng(expr_seed), depth)
    assert violations(expr, market_seed, cut) == 0, expr


def test_a_condition_expression_is_causal_too() -> None:
    assert (
        violations("$ret", 7, 20, condition="And($close > $open, $volume > Mean($volume, 8))") == 0
    )


# --- the test must be able to fail ---------------------------------------------------------------
def leaky(shift: int):  # type: ignore[no-untyped-def]
    def feature(panel: Panel) -> pl.DataFrame:
        return panel.df.select(
            "tix", "ts", pl.col("close").shift(-shift).over("tix").alias("score")
        )

    return feature


@pytest.mark.parametrize("shift", [1, 3])
def test_a_feature_that_reads_the_future_is_caught(shift: int) -> None:
    bars, universe, sectors = random_market(11)
    n = causality_violations(leaky(shift), bars, universe, sectors, SESSION_BARS, TS[20], seed=11)
    assert n >= shift  # the rows just before the cutoff see the noise after it


def test_the_aqua_style_leak_full_day_volume_is_caught() -> None:
    def feature(panel: Panel) -> pl.DataFrame:
        day = pl.col("volume").sum().over(["tix", "sidx"])
        cum = pl.col("volume").fill_null(0).cum_sum().over(["tix", "sidx"])
        return panel.df.select("tix", "ts", (cum / day).alias("score"))

    bars, universe, sectors = random_market(12)
    assert causality_violations(feature, bars, universe, sectors, SESSION_BARS, TS[5], seed=12) > 0


def test_the_perturbation_really_changes_later_rows_and_spares_earlier_ones() -> None:
    bars, _, _ = random_market(13)
    noisy = perturb_after(bars, TS[20], np.random.default_rng(0))
    before = bars.filter(pl.col("ts") <= TS[20])
    assert noisy.filter(pl.col("ts") <= TS[20]).equals(before)
    later = bars.filter(pl.col("ts") > TS[20])
    assert not noisy.filter(pl.col("ts") > TS[20]).select("close").equals(later.select("close"))


def values_before_cutoff(expr: str, seed: int, cut: int) -> int:
    bars, universe, sectors = random_market(seed)
    from alpha_factory.sandbox.panel import build_panel

    panel = build_panel(bars, universe, sectors, SESSION_BARS)
    out = compile_expr(expr).run(panel.df).filter(pl.col("ts") <= TS[cut])
    return out["score"].drop_nulls().len()


def test_the_checks_are_not_vacuous_long_windows_do_produce_values() -> None:
    for expr in ("Mean($close, 130)", "TsRank($close, 65)", "Corr($close, $volume, 39)"):
        assert values_before_cutoff(expr, 3, 160) > 50, expr


def test_most_random_expressions_produce_values_to_compare() -> None:
    rng = np.random.default_rng(9)
    informative = sum(
        values_before_cutoff(random_expression(rng, 4), 5, 160) > 0 for _ in range(60)
    )
    assert informative >= 35
