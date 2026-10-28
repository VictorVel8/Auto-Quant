import json
from datetime import date
from pathlib import Path

import polars as pl

from alpha_factory.data.layout import DataLayout
from alpha_factory.data.reference import build_ref_tables

FETCH = "2026-10-28.json"
TICKERS = [
    {"ticker": "AAA", "type": "CS", "share_class_figi": "FIGI_A"},
    {"ticker": "OLD", "type": "CS", "delisted_utc": "2023-02-14T00:00:00Z"},
]


def write(root: Path, endpoint: str, results: object) -> None:
    d = root / "raw" / "ref" / endpoint
    d.mkdir(parents=True, exist_ok=True)
    (d / FETCH).write_text(json.dumps({"results": results}))


def make(root: Path, dividends: list[dict[str, object]]) -> DataLayout:
    write(root, "tickers_active", TICKERS[:1])
    write(root, "tickers_inactive", TICKERS[1:])
    write(root, "splits", [])
    write(root, "dividends", dividends)
    write(
        root,
        "ticker_details",
        {
            "AAA": {"results": {"list_date": "2010-01-04", "sic_code": "3674"}},
            "OLD": {"results": None},
        },
    )
    return DataLayout(root)


DIVS = [
    {"ticker": "AAA", "ex_dividend_date": "2026-09-10", "cash_amount": 0.5, "currency": "USD"},
    {"ticker": "AAA", "ex_dividend_date": "2026-09-10", "cash_amount": 2.0, "currency": "USD"},
    {"ticker": "AAA", "ex_dividend_date": "2026-09-01", "cash_amount": 0.3, "currency": "USD"},
]


def test_reference_tables_do_not_depend_on_response_order(tmp_path: Path) -> None:
    a = make(tmp_path / "a", DIVS)
    b = make(tmp_path / "b", list(reversed(DIVS)))
    build_ref_tables(a)
    build_ref_tables(b)
    for name in ("instruments", "symbols", "splits", "dividends", "sic"):
        assert pl.read_parquet(a.ref_table(name)).equals(pl.read_parquet(b.ref_table(name))), name


def test_details_supply_list_date_and_sic_but_delisted_names_have_none(tmp_path: Path) -> None:
    layout = make(tmp_path, DIVS)
    build_ref_tables(layout)
    inst = pl.read_parquet(layout.ref_table("instruments")).sort("ticker_id")
    aaa = inst.filter(pl.col("ticker_id") == "FIGI_A").row(0, named=True)
    old = inst.filter(pl.col("ticker_id") == "TKR:OLD:na").row(0, named=True)
    assert aaa["list_date"] == date(2010, 1, 4)
    assert old["list_date"] is None and old["delisted_date"] == date(2023, 2, 14)
    sic = pl.read_parquet(layout.ref_table("sic"))
    assert sic["ticker_id"].to_list() == ["FIGI_A"] and sic["sic_code"].to_list() == ["3674"]
