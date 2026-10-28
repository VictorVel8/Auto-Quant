import polars as pl
import pytest
from tests.helpers.panels import D1, D2, make_panel, small_sandbox

from alpha_factory.sandbox.labels import compute_labels

SB = small_sandbox()
FULL = [100.0 + b for b in range(13)]  # close_b = 100 + b, open_b = close_{b-1}
HALF = [200.0 + 2 * b for b in range(7)]


def lab(panel_prices: dict, ticker: str = "a", session=D1, members=None) -> pl.DataFrame:  # type: ignore[type-arg]
    panel = make_panel(panel_prices, members)
    labels = compute_labels(panel, SB)
    tix = panel.tickers.index(ticker)
    ts = panel.df.filter((pl.col("tix") == tix) & (pl.col("session_date") == session)).select(
        "bar", "ts"
    )
    return labels.filter(pl.col("tix") == tix).join(ts, on="ts").sort("bar")


def value(out: pl.DataFrame, bar: int, col: str) -> float | None:
    """The label at a bar, or None when it is null or the row was dropped (no label at all)."""
    row = out.filter(pl.col("bar") == bar)
    return None if row.is_empty() else row[col][0]


def test_golden_h1_value_and_signal_bar_counts_on_a_normal_day() -> None:
    out = lab({"a": {D1: FULL}, "b": {D1: FULL}})
    # signal bar t=0: enter at open of bar 1 (= close_0 = 100), exit at close of bar 2 (= 102)
    assert out["y_h1"][0] == pytest.approx(102 / 100 - 1)
    assert out.filter(pl.col("y_h1").is_not_null())["bar"].to_list() == list(range(0, 11))
    assert out.filter(pl.col("y_h2").is_not_null())["bar"].to_list() == list(range(0, 9))
    assert out.filter(pl.col("y_h4").is_not_null())["bar"].to_list() == list(range(0, 5))
    # h4 at bar 4: open of bar 5 = close_4 = 104, close of bar 12 = 112
    assert out.filter(pl.col("bar") == 4)["y_h4"][0] == pytest.approx(112 / 104 - 1)


def test_half_days_have_shorter_horizons_and_no_h4() -> None:
    out = lab({"a": {D1: HALF}, "b": {D1: HALF}})
    assert out.filter(pl.col("y_h1").is_not_null())["bar"].to_list() == [0, 1, 2, 3, 4]
    assert out.filter(pl.col("y_h2").is_not_null())["bar"].to_list() == [0, 1, 2]
    assert out.filter(pl.col("y_h4").is_not_null()).height == 0


def test_a_missing_exit_or_entry_bar_gives_a_null_label_and_nothing_is_filled() -> None:
    no_exit = [*FULL[:2], None, *FULL[3:]]  # bar 2 missing: it is h1's exit bar for t=0
    out = lab({"a": {D1: no_exit}, "b": {D1: FULL}})
    assert value(out, 0, "y_h1") is None  # exit bar missing
    assert value(out, 1, "y_h1") is None  # entry bar (bar 2's open) missing
    assert value(out, 3, "y_h1") == pytest.approx(105 / 103 - 1)  # fully observed again
    no_entry = [FULL[0], None, *FULL[2:]]  # bar 1 missing: entry for t=0
    out = lab({"a": {D1: no_entry}, "b": {D1: FULL}})
    assert value(out, 0, "y_h1") is None and value(out, 0, "y_h2") is None


def test_labels_stay_inside_the_session() -> None:
    out = lab({"a": {D1: FULL, D2: FULL}, "b": {D1: FULL, D2: FULL}})
    assert value(out, 11, "y_h1") is None  # t+2 would be tomorrow
    assert value(out, 10, "y_h1") is not None


def test_demeaned_labels_average_zero_over_members_and_non_members_get_none() -> None:
    prices = {
        "a": {D1: FULL},
        "b": {D1: [100.0 + 2 * b for b in range(13)]},
        "c": {D1: [100.0 - b for b in range(13)]},
    }
    panel = make_panel(prices, members=["a", "b"])
    labels = compute_labels(panel, SB)
    assert panel.tickers.index("c") not in set(labels["tix"].unique())
    means = labels.group_by("ts").agg(pl.col("d_h1").mean().abs().max().alias("m"))
    assert means["m"].max() == pytest.approx(0.0, abs=1e-12)
    raw = labels.filter(pl.col("ts") == labels["ts"].min()).sort("tix")
    assert raw["y_h1"].to_list() == pytest.approx([102 / 100 - 1, 104 / 100 - 1])


def test_panel_is_dense_with_null_rows_for_gaps_and_correct_derived_fields() -> None:
    gap = [*FULL[:4], None, *FULL[5:]]
    p = make_panel({"a": {D1: gap, D2: FULL}})
    a = p.df.filter(pl.col("tix") == 0)
    assert a.height == 26  # two sessions x 13 bars, the gap kept as a null row
    assert a.filter((pl.col("session_date") == D1) & (pl.col("bar") == 4))["close"][0] is None
    assert a["tsi"].diff().drop_nulls().unique().to_list() == [1]
    day2 = a.filter(pl.col("session_date") == D2)
    assert day2["prevclose"][0] == 112.0 and day2["dayopen"].to_list() == [100.0] * 13
    # ret needs the previous bar: null right after the gap, normal elsewhere
    d1 = a.filter(pl.col("session_date") == D1).sort("bar")
    assert d1["ret"][5] is None and d1["ret"][3] == pytest.approx(103 / 102 - 1)
    assert a.filter(pl.col("session_date") == D1)["prevclose"].null_count() == 13
