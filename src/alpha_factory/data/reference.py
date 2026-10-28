"""Reference tables: fetch and cache REST responses, then normalise them into ref/*.parquet."""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from alpha_factory.data import massive
from alpha_factory.data.clients import RestClient, write_cached_json
from alpha_factory.data.layout import DataLayout


def latest_cached(layout: DataLayout, endpoint: str) -> dict[str, Any] | None:
    """The newest cached response for an endpoint (file names are fetch dates)."""
    files = sorted(layout.raw_ref_dir(endpoint).glob("*.json"))
    return json.loads(files[-1].read_text()) if files else None


def fetch_reference(
    client: RestClient,
    layout: DataLayout,
    fetch_date: date,
    start: date,
    types: tuple[str, ...],
    concurrency: int = 8,
) -> None:
    """Fetch every reference endpoint and cache the raw JSON. Details are fetched incrementally."""
    for name, (path, params) in massive.ENDPOINTS.items():
        extra = {"execution_date.gte": start.isoformat()} if name == "splits" else {}
        if name == "dividends":
            extra = {"ex_dividend_date.gte": start.isoformat()}
        rows = client.get_all(path, {**params, **extra})
        write_cached_json(layout.raw_ref_dir(name), fetch_date.isoformat(), {"results": rows})

    active = massive.parse_tickers(
        (latest_cached(layout, "tickers_active") or {"results": []})["results"]
    )
    wanted = sorted({r["ticker"] for r in active if r["type"] in types})
    cached: dict[str, Any] = (latest_cached(layout, massive.DETAILS_ENDPOINT) or {"results": {}})[
        "results"
    ]
    todo = [t for t in wanted if t not in cached]

    def one(t: str) -> tuple[str, dict[str, Any]]:
        # Delisted names 404; keep a marker so reruns do not ask again.
        return t, client.get_json_or_none(f"/v3/reference/tickers/{t}") or {"results": None}

    with ThreadPoolExecutor(concurrency) as ex:
        for t, payload in ex.map(one, todo):
            cached[t] = payload
    write_cached_json(
        layout.raw_ref_dir(massive.DETAILS_ENDPOINT), fetch_date.isoformat(), {"results": cached}
    )


def build_ref_tables(layout: DataLayout) -> dict[str, int]:
    """Normalise the newest cached responses into ref/*.parquet (no network). Returns row counts."""
    raw: dict[str, list[dict[str, Any]]] = {
        n: (latest_cached(layout, n) or {"results": []})["results"] for n in massive.ENDPOINTS
    }
    details = (latest_cached(layout, massive.DETAILS_ENDPOINT) or {"results": {}})["results"]
    tickers = massive.parse_tickers(
        raw["tickers_active"] + raw["tickers_inactive"], massive.parse_list_dates(details)
    )

    instruments = pl.DataFrame(
        tickers,
        schema={
            "ticker_id": pl.String,
            "ticker": pl.String,
            "name": pl.String,
            "type": pl.String,
            "primary_exchange": pl.String,
            "composite_figi": pl.String,
            "share_class_figi": pl.String,
            "list_date": pl.Date,
            "delisted_date": pl.Date,
        },
    )
    symbols = (
        instruments.select(
            "ticker_id",
            "ticker",
            pl.col("list_date").alias("valid_from"),
            pl.col("delisted_date").alias("valid_to"),
        )
        .unique()
        .sort("ticker", "valid_from", "ticker_id", nulls_last=False)
    )
    # One row per instrument: an active record wins over a stale delisted one.
    instruments = (
        instruments.sort(
            "ticker_id",
            "delisted_date",
            "ticker",
            descending=[False, True, False],
            nulls_last=False,
        )
        .unique("ticker_id", keep="first", maintain_order=True)
        .drop("ticker")
        .sort("ticker_id")
    )

    # Prefer the instrument active on the split date; tickers map through symbols by symbol period.
    def by_symbol(
        rows: list[dict[str, Any]], date_col: str, schema: dict[str, Any]
    ) -> pl.DataFrame:
        df = pl.DataFrame(rows, schema=schema)
        joined = df.join(symbols, on="ticker", how="inner").filter(
            (pl.col("valid_from").is_null() | (pl.col(date_col) >= pl.col("valid_from")))
            & (pl.col("valid_to").is_null() | (pl.col(date_col) <= pl.col("valid_to")))
        )
        return (
            joined.select("ticker_id", *[c for c in schema if c != "ticker"])
            .unique()
            # Sort on every column: rows tied on (ticker, date), e.g. a regular and a special
            # dividend on one ex-date, must not come out in arbitrary order.
            .sort("ticker_id", date_col, *[c for c in schema if c not in ("ticker", date_col)])
        )

    splits = by_symbol(
        massive.parse_splits(raw["splits"]),
        "execution_date",
        {
            "ticker": pl.String,
            "execution_date": pl.Date,
            "ratio": pl.Float64,
            "adjustment_type": pl.String,
        },
    )
    dividends = by_symbol(
        massive.parse_dividends(raw["dividends"]),
        "ex_date",
        {"ticker": pl.String, "ex_date": pl.Date, "cash_amount": pl.Float64},
    )
    sic_by_ticker = massive.parse_sic(details)
    sic = (
        pl.DataFrame(
            {"ticker": list(sic_by_ticker), "sic_code": list(sic_by_ticker.values())},
            schema={"ticker": pl.String, "sic_code": pl.String},
        )
        .join(symbols.select("ticker", "ticker_id"), on="ticker", how="inner")
        .select("ticker_id", "sic_code")
        .sort("ticker_id", "sic_code")
        .unique("ticker_id", keep="first", maintain_order=True)
        .sort("ticker_id")
    )

    tables = {
        "instruments": instruments,
        "symbols": symbols,
        "splits": splits,
        "dividends": dividends,
        "sic": sic,
    }
    for name, df in tables.items():
        path: Path = layout.ref_table(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path, compression="zstd", statistics=False)
    return {n: df.height for n, df in tables.items()}
