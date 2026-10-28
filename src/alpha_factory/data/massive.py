"""Massive adapter: the only file that knows the vendor's field names.

Verified against the live service on 2026-10-28:
- Flat files: bucket `flatfiles`, key `us_stocks_sip/minute_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz`
  exist back to 2003; columns are exactly MINUTE_COLUMNS, `window_start` is Unix ns, minutes
  cover 04:00-20:00 ET, volume is an integer, there are no nulls (checked on 2024-06-10).

NOT yet verified (no REST API key was available): the REST base URL, the auth header, the
endpoint paths and every field name in the tables below (TICKER_FIELDS, SPLIT_FIELDS,
DIVIDEND_FIELDS, DETAIL_SIC_FIELD). They follow the plan's description. Probe each endpoint once
and fix the mapping here; nothing else in the package depends on vendor names.
"""

from datetime import date, datetime
from typing import Any

import polars as pl

S3_KEY_TEMPLATE = "us_stocks_sip/minute_aggs_v1/{y:04d}/{m:02d}/{date}.csv.gz"
REST_BASE_URL = "https://api.massive.com"

# Endpoint cache names -> (path, query params). Responses are cached under raw/ref/<name>/.
ENDPOINTS: dict[str, tuple[str, dict[str, str]]] = {
    "tickers_active": (
        "/v3/reference/tickers",
        {"market": "stocks", "active": "true", "limit": "1000"},
    ),
    "tickers_inactive": (
        "/v3/reference/tickers",
        {"market": "stocks", "active": "false", "limit": "1000"},
    ),
    "splits": ("/stocks/v1/splits", {"limit": "1000"}),
    "dividends": ("/stocks/v1/dividends", {"limit": "1000"}),
}
DETAILS_ENDPOINT = "ticker_details"  # GET /v3/reference/tickers/{ticker}, one cached file of many

# internal column -> vendor field
TICKER_FIELDS = {
    "ticker": "ticker",
    "name": "name",
    "type": "type",
    "primary_exchange": "primary_exchange",
    "composite_figi": "composite_figi",
    "share_class_figi": "share_class_figi",
    "list_date": "list_date",
    "delisted_date": "delisted_utc",
}
SPLIT_FIELDS = {
    "ticker": "ticker",
    "execution_date": "execution_date",
    "split_from": "split_from",
    "split_to": "split_to",
    "adjustment_type": "adjustment_type",
}
DIVIDEND_FIELDS = {"ticker": "ticker", "ex_date": "ex_dividend_date", "cash_amount": "cash_amount"}
DETAIL_SIC_FIELD = "sic_code"

MINUTE_COLUMNS = (
    "ticker",
    "volume",
    "open",
    "close",
    "high",
    "low",
    "window_start",
    "transactions",
)


def minute_key(session: date) -> str:
    return S3_KEY_TEMPLATE.format(y=session.year, m=session.month, date=session.isoformat())


def _date(value: Any) -> date | None:
    """Parse "2024-06-10" or an ISO timestamp into a date."""
    if value in (None, ""):
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _get(row: dict[str, Any], fields: dict[str, str], key: str) -> Any:
    return row.get(fields[key])


def make_ticker_id(share_class_figi: str | None, ticker: str, list_date: date | None) -> str:
    """The share-class FIGI when present, else TKR:<symbol>:<list_date>."""
    if share_class_figi:
        return share_class_figi
    return f"TKR:{ticker}:{list_date.isoformat() if list_date else 'na'}"


def parse_tickers(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        ticker = _get(r, TICKER_FIELDS, "ticker")
        if not ticker:
            continue
        list_date = _date(_get(r, TICKER_FIELDS, "list_date"))
        figi = _get(r, TICKER_FIELDS, "share_class_figi") or None
        out.append(
            {
                "ticker_id": make_ticker_id(figi, ticker, list_date),
                "ticker": ticker,
                "name": _get(r, TICKER_FIELDS, "name") or "",
                "type": _get(r, TICKER_FIELDS, "type") or "",
                "primary_exchange": _get(r, TICKER_FIELDS, "primary_exchange") or "",
                "composite_figi": _get(r, TICKER_FIELDS, "composite_figi") or None,
                "share_class_figi": figi,
                "list_date": list_date,
                "delisted_date": _date(_get(r, TICKER_FIELDS, "delisted_date")),
            }
        )
    return out


def parse_splits(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """ratio = shares after / shares before = split_to / split_from."""
    out: list[dict[str, Any]] = []
    for r in rows:
        frm, to = _get(r, SPLIT_FIELDS, "split_from"), _get(r, SPLIT_FIELDS, "split_to")
        d = _date(_get(r, SPLIT_FIELDS, "execution_date"))
        if not (frm and to and d and _get(r, SPLIT_FIELDS, "ticker")):
            continue
        out.append(
            {
                "ticker": r[SPLIT_FIELDS["ticker"]],
                "execution_date": d,
                "ratio": float(to) / float(frm),
                "adjustment_type": str(_get(r, SPLIT_FIELDS, "adjustment_type") or ""),
            }
        )
    return out


def parse_dividends(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rows:
        d, cash = (
            _date(_get(r, DIVIDEND_FIELDS, "ex_date")),
            _get(r, DIVIDEND_FIELDS, "cash_amount"),
        )
        if d and cash is not None and _get(r, DIVIDEND_FIELDS, "ticker"):
            out.append(
                {"ticker": r[DIVIDEND_FIELDS["ticker"]], "ex_date": d, "cash_amount": float(cash)}
            )
    return out


def parse_sic(details: dict[str, dict[str, Any]]) -> dict[str, str]:
    """ticker -> zero-padded 4-digit SIC string, for tickers that have one."""
    out: dict[str, str] = {}
    for ticker, d in details.items():
        code = (d.get("results") or d).get(DETAIL_SIC_FIELD)
        if code not in (None, ""):
            out[ticker] = f"{int(code):04d}"
    return out


MINUTE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ticker": pl.String,
    "volume": pl.Float64,
    "open": pl.Float64,
    "close": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "window_start": pl.Int64,
    "transactions": pl.Int64,
}
