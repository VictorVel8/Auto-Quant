from datetime import UTC, date, datetime
from io import StringIO
from typing import Any
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from alpha_factory.data.bars import aggregate_session, symbols_for_session
from alpha_factory.data.massive import MINUTE_SCHEMA

NY = ZoneInfo("America/New_York")
MAPPING = pl.DataFrame(
    {"ticker": ["AAA", "ETF"], "ticker_id": ["id_a", "id_e"], "type": ["CS", "ETF"]}
)


def ns(d: date, hh: int, mm: int) -> int:
    return int(datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY).timestamp()) * 10**9


def minutes(d: date, rows: list[tuple[Any, ...]]) -> pl.LazyFrame:
    """rows: ticker, hh, mm, open, high, low, close, volume"""
    lines = ["ticker,volume,open,close,high,low,window_start,transactions"]
    for t, hh, mm, o, h, lo, c, v in rows:
        lines.append(f"{t},{v},{o},{c},{h},{lo},{ns(d, hh, mm)},3")
    return pl.read_csv(StringIO("\n".join(lines)), schema_overrides=MINUTE_SCHEMA).lazy()


DAY = date(2024, 11, 4)  # EST: UTC-5
ROWS = [
    ("AAA", 4, 0, 9.0, 9.0, 9.0, 9.0, 50.0),  # pre-market: dropped
    ("AAA", 9, 30, 10.0, 11.0, 9.0, 10.5, 100.0),
    ("AAA", 9, 31, 10.5, 12.0, 10.0, 11.0, 300.0),
    ("AAA", 9, 59, 11.0, 11.5, 10.8, 11.2, 100.0),
    ("AAA", 10, 0, 11.2, 11.3, 11.0, 11.1, 200.0),  # first minute of bar 1
    ("AAA", 10, 1, 11.1, 11.2, 11.0, 11.0, 0.0),  # zero volume: dropped
    ("AAA", 16, 0, 11.0, 11.0, 11.0, 11.0, 80.0),  # after close: dropped
    ("ETF", 9, 30, 100.0, 100.0, 100.0, 100.0, 10.0),  # mapped, wrong type: excluded
    ("ZZZ", 9, 30, 5.0, 5.0, 5.0, 5.0, 40.0),  # unmapped
]


def test_golden_bar_values() -> None:
    bars, _ = aggregate_session(minutes(DAY, ROWS), DAY, 13, MAPPING, ("CS",))
    assert bars["ticker_id"].unique().to_list() == ["id_a"]
    b0, b1 = bars.sort("bar").to_dicts()
    assert b0["bar"] == 0 and b1["bar"] == 1
    assert (b0["open"], b0["high"], b0["low"], b0["close"]) == (10.0, 12.0, 9.0, 11.2)
    assert b0["volume"] == 500.0 and b0["trades"] == 9 and b0["n_minutes"] == 3
    typical = [(11 + 9 + 10.5) / 3, (12 + 10 + 11) / 3, (11.5 + 10.8 + 11.2) / 3]
    expect = sum(t * v for t, v in zip(typical, [100, 300, 100], strict=True)) / 500
    assert b0["vwap"] == pytest.approx(expect)
    assert b1["n_minutes"] == 1 and b1["volume"] == 200.0


def test_bars_are_labelled_by_their_end_time_in_utc() -> None:
    bars, _ = aggregate_session(minutes(DAY, ROWS), DAY, 13, MAPPING, ("CS",))
    b0, b1 = bars.sort("bar").to_dicts()
    assert b0["ts"] == datetime(2024, 11, 4, 15, 0, tzinfo=UTC)  # 10:00 EST
    assert b1["ts"] == datetime(2024, 11, 4, 15, 30, tzinfo=UTC)


def test_summer_time_offsets() -> None:
    d = date(2024, 7, 1)  # EDT: UTC-4
    bars, _ = aggregate_session(
        minutes(d, [("AAA", 9, 30, 1, 1, 1, 1, 10)]), d, 13, MAPPING, ("CS",)
    )
    assert bars["ts"][0] == datetime(2024, 7, 1, 14, 0, tzinfo=UTC)


def test_dollar_volume_stats() -> None:
    _, stats = aggregate_session(minutes(DAY, ROWS), DAY, 13, MAPPING, ("CS",))
    assert stats.cs_dollar_volume == pytest.approx(11.2 * 500 + 11.1 * 200)
    assert stats.unmapped_dollar_volume == pytest.approx(
        5.0 * 40
    )  # the ETF is mapped, not unmapped


def test_half_day_has_seven_bars() -> None:
    d = date(2024, 11, 29)
    rows = [
        ("AAA", 12, 59, 1, 1, 1, 1, 10),
        ("AAA", 13, 0, 2, 2, 2, 2, 10),
        ("AAA", 9, 30, 1, 1, 1, 1, 10),
    ]
    bars, _ = aggregate_session(minutes(d, rows), d, 7, MAPPING, ("CS",))
    assert bars.sort("bar")["bar"].to_list() == [0, 6]
    assert bars.sort("bar")["ts"][-1] == datetime(2024, 11, 29, 18, 0, tzinfo=UTC)  # 13:00 EST


def test_other_dates_in_a_file_are_ignored() -> None:
    rows = [("AAA", 9, 30, 1, 1, 1, 1, 10)]
    other = date(2024, 11, 5)
    bars, _ = aggregate_session(minutes(other, rows), DAY, 13, MAPPING, ("CS",))
    assert bars.is_empty()


def test_symbol_periods_resolve_renames_and_reused_tickers() -> None:
    symbols = pl.DataFrame(
        {
            "ticker_id": ["old", "new", "late"],
            "ticker": ["TKR", "TKR", "TKR2"],
            "valid_from": [date(2010, 1, 1), date(2024, 6, 1), None],
            "valid_to": [date(2024, 5, 31), None, date(2020, 1, 1)],
        }
    )
    instruments = pl.DataFrame({"ticker_id": ["old", "new", "late"], "type": ["CS"] * 3})
    for d, expected in ((date(2024, 1, 2), "old"), (date(2024, 7, 1), "new")):
        m = symbols_for_session(symbols, instruments, d).filter(pl.col("ticker") == "TKR")
        assert m["ticker_id"].to_list() == [expected]
    assert (
        symbols_for_session(symbols, instruments, date(2024, 7, 1))
        .filter(pl.col("ticker") == "TKR2")
        .is_empty()
    )


def test_fractional_volumes_sum_exactly() -> None:
    rows = [("AAA", 9, 30 + i % 29, 1, 1, 1, 1, 0.1) for i in range(29)]  # 29 x 0.1 shares
    rows += [("AAA", 9, 59, 1, 1, 1, 1, 1220415.586502)]
    bars, _ = aggregate_session(minutes(DAY, rows), DAY, 13, MAPPING, ("CS",))
    assert abs(bars["volume"][0] - 1220418.486502) < 1e-9


def test_two_symbols_sharing_an_instrument_keep_only_the_busier_one() -> None:
    mapping = pl.DataFrame(
        {
            "ticker": ["BIG", "TINY", "SOLO"],
            "ticker_id": ["id_x", "id_x", "id_s"],
            "type": ["CS"] * 3,
        }
    )
    rows = [
        ("BIG", 9, 30, 10, 10, 10, 10, 1000), ("BIG", 9, 31, 10, 10, 10, 10, 1000),
        ("TINY", 9, 30, 50, 50, 50, 50, 10), ("SOLO", 9, 30, 7, 7, 7, 7, 5),
    ]  # fmt: skip
    bars, stats = aggregate_session(minutes(DAY, rows), DAY, 13, mapping, ("CS",))
    assert sorted(bars["ticker"].to_list()) == ["BIG", "SOLO"]
    assert bars.group_by("ticker_id", "bar").len()["len"].max() == 1  # no duplicate (id, bar)
    assert stats.cs_dollar_volume == pytest.approx(
        10 * 2000 + 50 * 10 + 7 * 5
    )  # volume is still counted
