from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest
from scipy import stats

from alpha_factory.contracts.sandbox import DateRange, Fold
from alpha_factory.eval.stats import (
    by_year,
    daily_ic,
    fold_stats,
    icir,
    intraday_ics,
    newey_west_tstat,
)

T0 = datetime(2024, 3, 4, 15, 0, tzinfo=timezone.utc)  # noqa: UP017


def frames(
    per_ts: Sequence[tuple[Sequence[float | None], Sequence[float | None]]],
    day_of: list[int] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    rows_s, rows_l = [], []
    for i, (s, y) in enumerate(per_ts):
        ts = T0 + timedelta(minutes=30 * i)
        d = date(2024, 3, 4) + timedelta(days=(day_of[i] if day_of else 0))
        for tix, (sv, yv) in enumerate(zip(s, y, strict=True)):
            rows_s.append({"tix": tix, "ts": ts, "session_date": d, "score": sv})
            rows_l.append({"tix": tix, "ts": ts, "y_h1": yv})
    return pl.DataFrame(rows_s), pl.DataFrame(rows_l)


def test_spearman_golden_values() -> None:
    # 1 - 6 * sum(d^2) / (n (n^2 - 1)) = 1 - 6 * 2 / 60 = 0.8
    s, y = frames([([1, 2, 3, 4], [1, 3, 2, 4])])
    out = intraday_ics(s, y, "y_h1", min_names=4)
    assert out["ic"][0] == pytest.approx(0.8) and out["n"][0] == 4


def test_ties_use_average_ranks_like_scipy() -> None:
    a, b = [1.0, 1.0, 2.0, 3.0, 3.0], [5.0, 1.0, 4.0, 2.0, 3.0]
    s, y = frames([(a, b)])
    assert intraday_ics(s, y, "y_h1", 5)["ic"][0] == pytest.approx(stats.spearmanr(a, b)[0])


def test_names_with_a_missing_score_or_label_are_dropped_and_small_sections_skipped() -> None:
    s, y = frames(
        [([1, 2, 3, 4, None], [1, 3, 2, 4, 9]), ([1, 2, 3, 4, 5], [1, 3, None, None, None])]
    )
    out = intraday_ics(s, y, "y_h1", min_names=4)
    assert out.height == 1 and out["n"][0] == 4 and out["ic"][0] == pytest.approx(0.8)


def test_constant_scores_give_no_ic() -> None:
    s, y = frames([([2, 2, 2, 2], [1, 2, 3, 4])])
    assert intraday_ics(s, y, "y_h1", 4).is_empty()


def test_daily_ic_averages_bars_within_a_session_and_applies_the_sign() -> None:
    s, y = frames(
        [([1, 2, 3, 4], [1, 2, 3, 4]), ([1, 2, 3, 4], [4, 3, 2, 1]), ([1, 2, 3, 4], [1, 3, 2, 4])],
        day_of=[0, 0, 1],
    )
    ics = intraday_ics(s, y, "y_h1", 4)
    d = daily_ic(ics, sign=1).sort("session_date")
    assert d["ic"].to_list() == pytest.approx([0.0, 0.8])  # (1 + -1) / 2 and 0.8
    assert daily_ic(ics, sign=-1).sort("session_date")["ic"].to_list() == pytest.approx([0.0, -0.8])


def nw_reference(x: np.ndarray, lag: int) -> float:
    """Direct, loop-based Newey-West with Bartlett weights."""
    d = len(x)
    m = x.mean()
    g = [sum((x[t] - m) * (x[t - k] - m) for t in range(k, d)) / d for k in range(lag + 1)]
    omega = g[0] + 2 * sum((1 - k / (lag + 1)) * g[k] for k in range(1, lag + 1))
    return m / np.sqrt(omega / d)


def test_newey_west_matches_an_independent_loop_implementation() -> None:
    rng = np.random.default_rng(0)
    x = 0.02 + np.convolve(rng.normal(0, 0.05, 220), [1, 0.6, 0.3], mode="same")  # autocorrelated
    assert newey_west_tstat(x, 5) == pytest.approx(nw_reference(x, 5))
    assert newey_west_tstat(x, 0) == pytest.approx(x.mean() / np.sqrt(x.var() / len(x)))
    # positive autocorrelation makes the effective sample smaller, so the t-stat falls
    assert abs(newey_west_tstat(x, 5)) < abs(newey_west_tstat(x, 0))


def test_degenerate_series() -> None:
    assert newey_west_tstat(np.array([0.1, 0.1, 0.1]), 5) == 0.0
    assert newey_west_tstat(np.ones(30), 5) == 0.0
    assert icir(np.array([0.5])) == 0.0


def test_icir_is_mean_over_standard_deviation() -> None:
    x = np.array([0.01, 0.03, 0.02, 0.04])
    assert icir(x) == pytest.approx(x.mean() / x.std(ddof=1))


def test_fold_and_year_statistics() -> None:
    days = [date(2022, 1, 3) + timedelta(days=i) for i in range(0, 400, 2)]
    daily = pl.DataFrame(
        {"session_date": days, "ic": [0.01 if d.year == 2022 else -0.02 for d in days]}
    )
    folds = (
        Fold(fold_id="A", train=None, val=DateRange(start=date(2022, 1, 1), end=date(2022, 6, 30))),
        Fold(fold_id="B", train=None, val=DateRange(start=date(2023, 1, 1), end=date(2023, 1, 31))),
    )
    a, b = fold_stats(daily, folds)
    assert a.ic_mean == pytest.approx(0.01) and a.n_days == 90
    assert b.ic_mean == pytest.approx(-0.02) and b.n_days == 15
    assert by_year(daily) == pytest.approx({2022: 0.01, 2023: -0.02})


def test_inverse_normal_matches_scipy() -> None:
    from alpha_factory.eval.stats import norm_ppf

    p = np.concatenate([np.linspace(1e-6, 1 - 1e-6, 2001), [0.02425, 0.97575, 0.5]])
    assert np.allclose(norm_ppf(p), stats.norm.ppf(p), rtol=1e-8, atol=1e-8)
