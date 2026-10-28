from datetime import date

from alpha_factory.data import massive

TICKER_ROWS = [
    {
        "ticker": "NVDA",
        "name": "NVIDIA",
        "type": "CS",
        "primary_exchange": "XNAS",
        "composite_figi": "BBG000BBJQV0",
        "share_class_figi": "BBG001S5TZJ6",
        "list_date": "1999-01-22",
    },
    {
        "ticker": "OLD",
        "type": "CS",
        "list_date": "2010-05-03",
        "delisted_utc": "2023-02-14T00:00:00Z",
    },
    {"ticker": "", "type": "CS"},
]


def test_ticker_id_uses_share_class_figi_or_falls_back() -> None:
    a, b = massive.parse_tickers(TICKER_ROWS, {"OLD": date(2010, 5, 3)})
    assert a["ticker_id"] == "BBG001S5TZJ6"
    assert b["ticker_id"] == "TKR:OLD:2010-05-03"
    assert massive.parse_tickers(TICKER_ROWS)[1]["ticker_id"] == "TKR:OLD:na"
    assert massive.make_ticker_id(None, "X", None) == "TKR:X:na"


def test_ticker_dates_and_delisting() -> None:
    a, b = massive.parse_tickers(TICKER_ROWS, {"NVDA": date(1999, 1, 22)})
    assert a["list_date"] == date(1999, 1, 22) and a["delisted_date"] is None
    assert b["delisted_date"] == date(2023, 2, 14)


def test_split_ratio_is_shares_after_over_before() -> None:
    rows = [
        {
            "ticker": "NVDA",
            "execution_date": "2024-06-10",
            "split_from": 1,
            "split_to": 10,
            "adjustment_type": "forward_split",
        },
        {
            "ticker": "XYZ",
            "execution_date": "2023-01-03",
            "split_from": 10,
            "split_to": 1,
            "adjustment_type": "reverse_split",
        },
        {"ticker": "BAD", "execution_date": "2023-01-03"},
    ]
    nvda, xyz = massive.parse_splits(rows)
    assert nvda["ratio"] == 10.0
    assert xyz["ratio"] == 0.1
    assert nvda["execution_date"] == date(2024, 6, 10)


def test_dividends_and_sic() -> None:
    divs = massive.parse_dividends(
        [
            {
                "ticker": "A",
                "ex_dividend_date": "2024-02-01",
                "cash_amount": 0.25,
                "currency": "USD",
            },
            {"ticker": "B"},
            {
                "ticker": "C",
                "ex_dividend_date": "2024-02-01",
                "cash_amount": 1.0,
                "currency": "EUR",
            },
        ]
    )
    assert divs == [{"ticker": "A", "ex_date": date(2024, 2, 1), "cash_amount": 0.25}]
    sic = massive.parse_sic(
        {
            "A": {"results": {"sic_code": "3674"}},
            "B": {"results": {}},
            "C": {"results": {"sic_code": 100}},
        }
    )
    assert sic == {"A": "3674", "C": "0100"}


def test_list_dates_come_from_details_and_skip_delisted_markers() -> None:
    dates = massive.parse_list_dates(
        {
            "A": {"results": {"list_date": "1999-01-22"}},
            "B": {"results": None},
            "C": {"results": {}},
        }
    )
    assert dates == {"A": date(1999, 1, 22)}


def test_minute_key() -> None:
    assert (
        massive.minute_key(date(2026, 10, 23))
        == "us_stocks_sip/minute_aggs_v1/2026/10/2026-10-23.csv.gz"
    )
