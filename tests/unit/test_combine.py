from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from tests.helpers.panels import make_panel

from alpha_factory.contracts import DateRange
from alpha_factory.eval.combine import rank_average, rank_features, ridge_fit, walk_forward_ridge


def frame(scores: dict[str, list[float | None]]) -> pl.DataFrame:
    """One timestamp, one row per ticker, built from a panel so the keys are real."""
    panel = make_panel({t: {date(2024, 3, 4): [100.0] * 13} for t in scores})
    df = panel.df.filter(pl.col("bar") == 0).sort("tix")
    return df.select("tix", "ts", "session_date", "tsi").with_columns(
        pl.Series("score", list(scores.values())[0])
    )


def two_factor_frames(a: list[float | None], b: list[float | None]) -> dict[str, pl.DataFrame]:
    tickers = {f"t{i}": 0 for i in range(len(a))}
    base = frame({t: [1.0] * len(a) for t in tickers})
    return {
        "a": base.with_columns(pl.Series("score", a)),
        "b": base.with_columns(pl.Series("score", b)),
    }


def test_ranks_are_centred_cross_sectional_percentiles() -> None:
    f = rank_features(two_factor_frames([10.0, 20.0, 30.0, 40.0], [4.0, 3.0, 2.0, 1.0]))
    assert f.sort("tix")["a"].to_list() == pytest.approx([-0.25, 0.0, 0.25, 0.5])
    assert f.sort("tix")["b"].to_list() == pytest.approx([0.5, 0.25, 0.0, -0.25])


def test_rank_average_applies_signs_and_averages() -> None:
    frames = two_factor_frames([10.0, 20.0, 30.0, 40.0], [4.0, 3.0, 2.0, 1.0])
    # b ranks are the reverse of a's; the average of a and b cancels to a constant
    cancel = rank_average(frames, {"a": 1, "b": 1}).sort("tix")["score"].to_list()
    assert cancel == pytest.approx([0.125, 0.125, 0.125, 0.125])
    # with b's sign flipped: mean([-.25, 0, .25, .5], [-.5, -.25, 0, .25])
    flipped = rank_average(frames, {"a": 1, "b": -1}).sort("tix")["score"].to_list()
    assert flipped == pytest.approx([-0.375, -0.125, 0.125, 0.375])


def test_rank_average_needs_at_least_half_the_factors() -> None:
    frames = two_factor_frames([1.0, 2.0, 3.0, 4.0], [None, 1.0, None, None])
    two = rank_average(frames, {"a": 1, "b": 1}).sort("tix")["score"].to_list()
    assert all(v is not None for v in two)  # one of two factors is enough (ceil(2/2) = 1)
    three = {**frames, "c": frames["b"]}
    out = rank_average(three, {"a": 1, "b": 1, "c": 1}).sort("tix")["score"].to_list()
    assert out[0] is None and out[1] is not None  # ticker 0 has 1 of 3; ticker 1 has all 3


def test_ridge_recovers_the_weights_of_a_linear_target() -> None:
    rng = np.random.default_rng(0)
    n = 4000
    ts = [date(2024, 1, 1) + timedelta(days=i % 40) for i in range(n)]
    f1, f2 = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.5, 0.5, n)
    y = 0.05 * f1 - 0.02 * f2 + rng.normal(0, 0.001, n)
    base = pl.DataFrame({"tix": np.arange(n, dtype=np.uint32), "ts": np.arange(n), "session_date": ts, "tsi": np.arange(n, dtype=np.uint32), "f1": f1, "f2": f2})  # fmt: skip
    labels = pl.DataFrame({"tix": np.arange(n, dtype=np.uint32), "ts": np.arange(n), "d_h2": y})
    model = ridge_fit(base, labels, ["f1", "f2"], "d_h2", alpha=1e-6)
    assert model.weights == pytest.approx([0.05, -0.02], abs=2e-3)
    scored = model.predict(base).sort("tix")
    assert np.corrcoef(scored["score"].to_numpy(), y)[0, 1] > 0.99
    shrunk = ridge_fit(base, labels, ["f1", "f2"], "d_h2", alpha=1e6)  # heavy shrinkage
    assert np.abs(shrunk.weights).max() < np.abs(model.weights).max()


def test_walk_forward_scores_only_later_folds_with_an_embargo() -> None:
    rng = np.random.default_rng(1)
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(60)]
    rows = [(d, j) for d in days for j in range(20)]
    n = len(rows)
    f = rng.uniform(-0.5, 0.5, n)
    base = pl.DataFrame({"tix": np.array([r[1] for r in rows], dtype=np.uint32), "ts": np.arange(n), "session_date": [r[0] for r in rows], "tsi": np.arange(n, dtype=np.uint32), "f": f})  # fmt: skip
    labels = pl.DataFrame({"tix": base["tix"], "ts": base["ts"], "d_h2": 0.03 * f})
    folds = [DateRange(start=days[0], end=days[19]), DateRange(start=days[20], end=days[39]), DateRange(start=days[40], end=days[59])]  # fmt: skip
    out = walk_forward_ridge(
        base, labels, ["f"], "d_h2", folds, days, alpha=1e-6, purge_sessions=1, embargo_sessions=5
    )
    assert out["session_date"].min() == days[20] and out["session_date"].max() == days[59]
    assert out.height == 40 * 20  # fold 1 has no training data, so it has no out-of-sample score
    assert np.corrcoef(out.sort("ts")["score"].to_numpy(), f[20 * 20 :])[0, 1] > 0.99
