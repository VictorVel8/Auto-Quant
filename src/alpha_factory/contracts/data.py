"""Data manifest, QA report and the bar schema Plan 2 must produce exactly."""

from datetime import date, datetime
from typing import Literal

import polars as pl
from pydantic import computed_field

from alpha_factory.contracts.base import Record, Sha256

# Column order and dtypes of every bars file, synthetic or from Massive.
BAR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "session_date": pl.Date,  # trading session (New York date)
    "ts": pl.Datetime("us", "UTC"),  # bar end time
    "bar": pl.Int8,  # bar of the day, 0 to 12 (fewer on half-days)
    "ticker_id": pl.String,  # stable instrument id
    "ticker": pl.String,  # symbol as traded that day
    "open": pl.Float64,  # forward-adjusted
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "raw_close": pl.Float64,  # unadjusted close, for price filters and share counts
    "adj_factor": pl.Float64,  # cumulative forward adjustment as of this session
    "volume": pl.Float64,  # forward-adjusted share volume
    "trades": pl.Int64,
    "vwap": pl.Float64,  # approximated from minute bars
    "n_minutes": pl.Int16,  # minutes with trades in the bar
}


class FileEntry(Record):
    path: str  # relative to AF_DATA_DIR
    sha256: Sha256
    rows: int
    bytes: int


class DataManifest(Record):
    dataset_id: str
    dataset_hash: Sha256  # hash of sorted file hashes + build params
    kind: Literal["dev", "full"]  # dev physically excludes the sealed window
    source: Literal["massive", "synthetic"]
    start: date
    end: date
    n_sessions: int
    n_tickers: int
    bar_minutes: Literal[30]
    adjustment: Literal["forward"]
    params_hash: Sha256
    code_commit: str
    created_at: datetime
    files: tuple[FileEntry, ...]


class QACheck(Record):
    name: str
    passed: bool
    value: float
    threshold: float
    detail: str


class QAReport(Record):
    dataset_hash: Sha256
    checks: tuple[QACheck, ...]

    @computed_field
    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)
