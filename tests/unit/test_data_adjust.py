from datetime import date, timedelta

import polars as pl
import pytest

from alpha_factory.data.adjust import apply_adjustment, compute_adj_factors

D = [date(2024, 3, 4) + timedelta(days=i) for i in range(6)]  # d1..d6 (consecutive for simplicity)


def daily(closes: list[float], ticker: str = "A") -> pl.DataFrame:
    return pl.DataFrame(
        {"ticker_id": ticker, "session_date": D[: len(closes)], "raw_close": closes}
    )


SPLITS = pl.DataFrame(
    {"ticker_id": ["A"], "execution_date": [D[2]], "ratio": [2.0]},
    schema={"ticker_id": pl.String, "execution_date": pl.Date, "ratio": pl.Float64},
)
DIVS = pl.DataFrame(
    {"ticker_id": ["A"], "ex_date": [D[4]], "cash_amount": [2.0]},
    schema={"ticker_id": pl.String, "ex_date": pl.Date, "cash_amount": pl.Float64},
)
CLOSES = [100.0, 100.0, 50.0, 50.0, 48.0, 48.0]  # 2:1 split on d3, $2 dividend on d5


def test_golden_factors() -> None:
    f = compute_adj_factors(daily(CLOSES), SPLITS, DIVS, D[0])
    assert f["split_factor"].to_list() == [1, 1, 2, 2, 2, 2]
    expected_div = 50 / 48  # C(e-1) / (C(e-1) - cash)
    assert f["div_factor"].to_list() == pytest.approx([1, 1, 1, 1, expected_div, expected_div])
    assert f["adj_factor"].to_list() == pytest.approx(
        [1, 1, 2, 2, 2 * expected_div, 2 * expected_div]
    )


def test_adjusted_closes_are_continuous_across_split_and_ex_date() -> None:
    f = compute_adj_factors(daily(CLOSES), SPLITS, DIVS, D[0])
    adj = [c * a for c, a in zip(CLOSES, f["adj_factor"].to_list(), strict=True)]
    assert adj == pytest.approx([100, 100, 100, 100, 100, 100])


def test_actions_on_or_before_the_first_session_are_ignored() -> None:
    early = pl.DataFrame(
        {"ticker_id": ["A", "A"], "execution_date": [D[0], D[2]], "ratio": [3.0, 2.0]}
    )
    f = compute_adj_factors(daily(CLOSES), early, DIVS.clear(), D[0])
    assert f["split_factor"].to_list() == [1, 1, 2, 2, 2, 2]


def test_reverse_split_factor() -> None:
    sp = pl.DataFrame({"ticker_id": ["A"], "execution_date": [D[1]], "ratio": [0.1]})
    f = compute_adj_factors(daily([10, 100, 100]), sp, DIVS.clear(), D[0])
    assert f["split_factor"].to_list() == pytest.approx([1, 0.1, 0.1])


def test_factors_do_not_depend_on_the_future() -> None:
    """Truncating the data (and dropping later actions) leaves earlier factors unchanged."""
    full = compute_adj_factors(daily(CLOSES), SPLITS, DIVS, D[0])
    for k in range(1, 7):
        cut = compute_adj_factors(
            daily(CLOSES[:k]),
            SPLITS.filter(pl.col("execution_date") <= D[k - 1]),
            DIVS.filter(pl.col("ex_date") <= D[k - 1]),
            D[0],
        )
        assert cut["adj_factor"].to_list() == pytest.approx(full["adj_factor"].to_list()[:k])


def test_instruments_are_independent() -> None:
    both = pl.concat([daily(CLOSES, "A"), daily([20.0] * 6, "B")])
    f = compute_adj_factors(both, SPLITS, DIVS, D[0])
    assert f.filter(pl.col("ticker_id") == "B")["adj_factor"].to_list() == [1.0] * 6


def test_dividend_without_prior_close_is_ignored() -> None:
    dv = pl.DataFrame({"ticker_id": ["Z"], "ex_date": [D[3]], "cash_amount": [1.0]})
    f = compute_adj_factors(daily(CLOSES), SPLITS, dv, D[0])
    assert f["div_factor"].to_list() == [1.0] * 6


def test_apply_adjustment_columns() -> None:
    from datetime import UTC, datetime

    raw = pl.DataFrame(
        {
            "session_date": [D[2]],
            "ts": [datetime(2024, 3, 6, 15, 0, tzinfo=UTC)],
            "bar": [0],
            "ticker_id": ["A"],
            "ticker": ["A"],
            "open": [50.0],
            "high": [51.0],
            "low": [49.0],
            "close": [50.0],
            "volume": [1000.0],
            "trades": [10],
            "vwap": [50.0],
            "n_minutes": [30],
        },
        schema_overrides={"bar": pl.Int8, "n_minutes": pl.Int16, "ts": pl.Datetime("us", "UTC")},
    )
    factors = pl.DataFrame(
        {
            "ticker_id": ["A"],
            "session_date": [D[2]],
            "split_factor": [2.0],
            "div_factor": [1.0],
            "adj_factor": [2.0],
        }
    )
    row = apply_adjustment(raw, factors).row(0, named=True)
    assert (row["open"], row["high"], row["low"], row["close"], row["vwap"]) == (
        100,
        102,
        98,
        100,
        100,
    )
    assert row["raw_close"] == 50.0
    assert row["volume"] == 500.0  # raw / split_factor
    assert row["adj_factor"] == 2.0
    with pytest.raises(ValueError, match="without an adjustment factor"):
        apply_adjustment(raw, factors.clear())


def test_dividends_on_the_same_ex_date_are_one_price_drop() -> None:
    two = pl.DataFrame(
        {"ticker_id": ["A", "A"], "ex_date": [D[4], D[4]], "cash_amount": [1.5, 0.5]}
    )
    f = compute_adj_factors(daily(CLOSES), SPLITS, two, D[0])
    # C(e-1) = 50 and total cash 2.0, not two compounded factors (50/48.5 * 50/49.5)
    assert f["div_factor"].to_list()[-1] == pytest.approx(50 / 48)
