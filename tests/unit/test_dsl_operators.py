"""Operator semantics against independent numpy/scipy references and hand-worked values."""

from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
from scipy import stats
from tests.helpers.panels import D1, D2, D3, make_panel

from alpha_factory.dsl import compile_expr

rng = np.random.default_rng(3)
WALK: list[float] = [float(v) for v in 100 + np.cumsum(rng.normal(0, 1, 39))]  # 3 sessions
GAP_AT = 20
WALK_GAP: list[float | None] = [None if i == GAP_AT else v for i, v in enumerate(WALK)]


def one_ticker(values: Sequence[float | None]) -> pl.DataFrame:
    sessions = {D1: values[:13], D2: values[13:26], D3: values[26:]}
    return make_panel({"a": sessions}).df


def score(expr: str, df: pl.DataFrame) -> list[float | None]:
    return compile_expr(expr).run(df).filter(pl.col("tix") == 0)["score"].to_list()


def close_of(values: Sequence[float | None]) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in values], dtype=float)


def reference(fn, x: np.ndarray, n: int) -> np.ndarray:  # type: ignore[no-untyped-def]
    out = np.full(len(x), np.nan)
    for t in range(n - 1, len(x)):
        w = x[t - n + 1 : t + 1]
        if not np.isnan(w).any():
            out[t] = fn(w)
    return out


def ema_ref(x: np.ndarray, n: int) -> np.ndarray:
    a = 2 / (n + 1)
    y = np.full(len(x), np.nan)
    y[0] = x[0]
    for t in range(1, len(x)):
        y[t] = a * x[t] + (1 - a) * y[t - 1]
    y[: n - 1] = np.nan
    return y


REFERENCES = {
    "Mean": lambda w: w.mean(),
    "Sum": lambda w: w.sum(),
    "Std": lambda w: w.std(ddof=1),
    "Min": lambda w: w.min(),
    "Max": lambda w: w.max(),
    "Skew": lambda w: stats.skew(w, bias=False),
    "TsRank": lambda w: ((w < w[-1]).sum() + 0.5 * (w == w[-1]).sum()) / len(w),
    "TsZScore": lambda w: (w[-1] - w.mean()) / w.std(ddof=1),
    "Decay": lambda w: np.dot(np.arange(1, len(w) + 1), w) / np.arange(1, len(w) + 1).sum(),
    "Slope": lambda w: np.polyfit(np.arange(len(w)), w, 1)[0],
}


@pytest.mark.parametrize("op", sorted(REFERENCES))
@pytest.mark.parametrize("n", [3, 8])
@pytest.mark.parametrize("gap", [False, True])
def test_time_series_operators_match_numpy(op: str, n: int, gap: bool) -> None:
    values = WALK_GAP if gap else WALK
    got = np.array(
        [np.nan if v is None else v for v in score(f"{op}($close, {n})", one_ticker(values))]
    )
    want = reference(REFERENCES[op], close_of(values), n)
    # the dataset's close is the walk itself (adjustment factor 1)
    assert np.allclose(got, want, rtol=1e-8, atol=1e-8, equal_nan=True), (op, n, gap)


@pytest.mark.parametrize("n", [3, 8])
def test_ema_matches_the_recursion_and_a_gap_stays_a_gap(n: int) -> None:
    got = np.array(
        [np.nan if v is None else v for v in score(f"EMA($close, {n})", one_ticker(WALK))]
    )
    assert np.allclose(got, ema_ref(close_of(WALK), n), equal_nan=True)
    gapped = score(f"EMA($close, {n})", one_ticker(WALK_GAP))
    assert gapped[GAP_AT] is None and gapped[GAP_AT + 1] is not None


@pytest.mark.parametrize("op", ["Corr", "Cov"])
def test_corr_and_cov_match_numpy(op: str) -> None:
    n = 8
    got = np.array(
        [np.nan if v is None else v for v in score(f"{op}($close, $high, {n})", one_ticker(WALK))]
    )
    close = close_of(WALK)
    # the helper opens a session at its first close, then at the previous close
    open_ = np.r_[close[0], close[:-1]]
    open_[[13, 26]] = close[[13, 26]]
    high = np.maximum(close, open_) + 0.01
    f = (
        (lambda a, b: np.corrcoef(a, b)[0, 1])
        if op == "Corr"
        else (lambda a, b: np.cov(a, b)[0, 1])
    )
    want = np.full(len(close), np.nan)
    for t in range(n - 1, len(close)):
        want[t] = f(close[t - n + 1 : t + 1], high[t - n + 1 : t + 1])
    assert np.allclose(got, want, rtol=1e-6, atol=1e-8, equal_nan=True)


def test_ref_delta_pct_and_a_gap_reaches_exactly_n_bars_back() -> None:
    df = one_ticker(WALK_GAP)
    close = close_of(WALK_GAP)
    ref = score("Ref($close, 2)", df)
    assert ref[:2] == [None, None] and ref[5] == pytest.approx(close[3])
    assert ref[GAP_AT + 2] is None  # looks back at the gap, not at the bar before it
    assert score("Delta($close, 1)", df)[5] == pytest.approx(close[5] - close[4])
    assert score("Pct($close, 3)", df)[7] == pytest.approx(close[7] / close[4] - 1)
    assert score("Delta($close, 1)", df)[GAP_AT] is None


def test_session_operators_restart_each_session() -> None:
    vol = one_ticker(WALK)  # volume is 1000 every bar
    s = score("SessSum($volume)", vol)
    assert s[:3] == [1000.0, 2000.0, 3000.0] and s[13] == 1000.0 and s[25] == 13000.0
    assert score("SessMean($volume)", vol)[14] == pytest.approx(1000.0)


def test_dayopen_prevclose_and_ret_fields() -> None:
    df = one_ticker(WALK)
    close = close_of(WALK)
    assert (
        score("$dayopen", df)[13:16] == [pytest.approx(close[13])] * 3
    )  # open of bar 0 = prev close
    assert score("$prevclose", df)[13] == pytest.approx(close[12])
    assert score("$prevclose", df)[0] is None
    assert score("$ret", df)[1] == pytest.approx(close[1] / close[0] - 1)


# --- cross-sectional ---------------------------------------------------------------------------
FOUR = {
    "a": {D1: [10.0] * 13},
    "b": {D1: [20.0] * 13},
    "c": {D1: [30.0] * 13},
    "d": {D1: [40.0] * 13},
}


def cs(
    expr: str, members: list[str] | None = None, sector_of: dict[str, str] | None = None
) -> dict[str, float | None]:
    panel = make_panel(FOUR, members, sector_of)
    out = compile_expr(expr).run(panel.df).filter(pl.col("bar") == 0).sort("tix")
    return dict(zip(panel.tickers, out["score"].to_list(), strict=True))


def test_rank_zscore_demean_over_members_only() -> None:
    assert cs("Rank($close)") == {"a": 0.25, "b": 0.5, "c": 0.75, "d": 1.0}
    z = np.array([10.0, 20, 30, 40])
    zs = (z - z.mean()) / z.std(ddof=1)
    assert [cs("ZScore($close)")[k] for k in "abcd"] == pytest.approx(list(zs))
    assert cs("Demean($close)") == {"a": -15.0, "b": -5.0, "c": 5.0, "d": 15.0}
    # d is not a member: it is null and does not take part in anyone's rank
    assert cs("Rank($close)", members=["a", "b", "c"]) == {
        "a": 1 / 3,
        "b": 2 / 3,
        "c": 1.0,
        "d": None,
    }
    assert cs("Demean($close)", members=["a", "b", "c"])["d"] is None


def test_winsor_and_sector_neutral() -> None:
    w = cs("Winsor($close, 0.05)")
    # linear-interpolated 5% / 95% quantiles of [10, 20, 30, 40] are 11.5 and 38.5
    assert w == {"a": pytest.approx(11.5), "b": 20.0, "c": 30.0, "d": pytest.approx(38.5)}
    sn = cs("SectorNeutral($close)", sector_of={"a": "x", "b": "x", "c": "y", "d": "y"})
    assert sn == {"a": -5.0, "b": 5.0, "c": -5.0, "d": 5.0}


# --- element-wise and safety ---------------------------------------------------------------------
def test_elementwise_operators() -> None:
    df = one_ticker(WALK)
    c = close_of(WALK)
    x = c[4] - c[0]
    assert score("Abs($close - $open)", df)[4] == pytest.approx(abs(c[4] - c[3]))
    assert score("SLog($close - $dayopen)", df)[4] == pytest.approx(
        np.sign(c[4] - c[0]) * np.log1p(abs(x))
    )
    assert score("SignedPower($close - $dayopen, 2)", df)[4] == pytest.approx(np.sign(x) * x**2)
    assert score("Greater($close, 100)", df)[4] == pytest.approx(max(c[4], 100))
    assert score("Less($close, 100)", df)[4] == pytest.approx(min(c[4], 100))
    assert score("Clip($close - $dayopen, -1, 1)", df)[4] == pytest.approx(np.clip(x, -1, 1))
    up = c[4] > c[3]
    high4 = max(c[4], c[3]) + 0.01
    low4 = min(c[4], c[3]) - 0.01
    assert score("If($close > $open, $high, $low)", df)[4] == pytest.approx(high4 if up else low4)
    both = score("If(And($close > $low, Not($close > $high)), $close, $volume)", df)[4]
    assert both == pytest.approx(c[4])


def test_division_by_near_zero_is_null_and_infinities_never_escape() -> None:
    df = one_ticker(WALK)
    assert all(v is None for v in score("$close / ($close - $close)", df))
    assert all(
        v is None for v in score("$close / (0.000000000000001 * $close)", df)
    )  # denominator ~1e-13
    out = score("Pct($close - $close, 1)", df)  # 0 / 0 style
    assert all(v is None for v in out)
    assert all(v is None or np.isfinite(v) for v in score("SLog($volume) / ($close - $close)", df))


def test_an_unconditional_nan_input_does_not_leak_into_other_rows() -> None:
    df = one_ticker(WALK_GAP)
    mean = score("Mean($close, 3)", df)
    assert mean[GAP_AT] is None and mean[GAP_AT + 2] is None and mean[GAP_AT + 3] is not None
    assert mean[GAP_AT - 1] is not None


def test_condition_runs_alongside_the_score() -> None:
    panel = make_panel(FOUR)
    out = (
        compile_expr("$close", condition="$close > 25")
        .run(panel.df)
        .filter(pl.col("bar") == 0)
        .sort("tix")
    )
    assert out["cond"].to_list() == [False, False, True, True]
