"""Minute flat files -> raw (unadjusted) 30-min bars, one session at a time.

Bars are labelled by their end time. Bar k holds the minutes starting in
[09:30 + 30k, 09:30 + 30(k+1)) New York time. An empty bar has no row; nothing is forward-filled.
"""

import json
import math
from datetime import date, datetime, time
from typing import Any

import polars as pl
from pydantic import BaseModel

from alpha_factory.data.calendar import BAR_MINUTES, OPEN
from alpha_factory.data.config import DataConfig
from alpha_factory.data.layout import DataLayout
from alpha_factory.data.massive import MINUTE_SCHEMA

NY = "America/New_York"
OPEN_MINUTE = OPEN.hour * 60 + OPEN.minute
PV_SCALE = 1_000_000

RAW_BAR_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "session_date": pl.Date,
    "ts": pl.Datetime("us", "UTC"),
    "bar": pl.Int8,
    "ticker_id": pl.String,
    "ticker": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "trades": pl.Int64,
    "vwap": pl.Float64,
    "n_minutes": pl.Int16,
}


class SessionStats(BaseModel):
    session_date: date
    cs_dollar_volume: float  # mapped instruments of an accepted type
    unmapped_dollar_volume: float  # symbols with no symbol period at all


def symbols_for_session(
    symbols: pl.DataFrame, instruments: pl.DataFrame, session: date
) -> pl.DataFrame:
    """ticker -> (ticker_id, type) valid on a session. One row per ticker."""
    valid = symbols.filter(
        (pl.col("valid_from").is_null() | (pl.col("valid_from") <= session))
        & (pl.col("valid_to").is_null() | (pl.col("valid_to") >= session))
    )
    return (
        valid.join(instruments.select("ticker_id", "type"), on="ticker_id", how="left")
        .sort("valid_from", descending=True, nulls_last=True)
        .unique("ticker", keep="first", maintain_order=True)
        .select("ticker", "ticker_id", "type")
    )


def one_symbol_per_instrument(bars: pl.DataFrame) -> pl.DataFrame:
    """When several symbols map to one instrument in a session, keep the busiest one.

    The vendor shares one share-class FIGI between symbols that trade at the same time (BK and
    BNY, DOC and PEAK). Keeping both would interleave two unrelated price series under one id, so
    the symbol with the larger dollar volume that session wins and the other's bars are dropped.
    Renames do not trigger this: the old and new symbols never trade on the same day.
    """
    dv = bars.group_by("ticker_id", "ticker").agg(pl.col("_dv").sum().alias("_sym_dv"))
    winner = (
        dv.sort("_sym_dv", "ticker", descending=[True, False])
        .unique("ticker_id", keep="first", maintain_order=True)
        .select("ticker_id", "ticker")
    )
    return bars.join(winner, on=["ticker_id", "ticker"], how="inner")


def aggregate_session(
    raw: pl.LazyFrame,
    session: date,
    n_bars: int,
    mapping: pl.DataFrame,
    types: tuple[str, ...],
) -> tuple[pl.DataFrame, SessionStats]:
    """Aggregate one session's minutes into raw 30-min bars for accepted instruments."""
    local = (
        pl.col("window_start")
        .cast(pl.Datetime("ns"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone(NY)
    )
    rel = local.dt.hour().cast(pl.Int32) * 60 + local.dt.minute().cast(pl.Int32) - OPEN_MINUTE
    bars = (
        raw.with_columns(local.dt.date().alias("_d"), rel.alias("_rel"))
        .filter(
            (pl.col("_d") == session)
            & (pl.col("_rel") >= 0)
            & (pl.col("_rel") < n_bars * BAR_MINUTES)
            & (pl.col("volume") > 0)
        )
        .with_columns(
            (pl.col("_rel") // BAR_MINUTES).cast(pl.Int8).alias("bar"),
            # Volumes can be fractional (2026 files), so sum them as exact integer micro-shares.
            (pl.col("volume") * PV_SCALE).round().cast(pl.Int64).alias("_vol"),
            # Integer micro-dollars: integer sums are exact, so a parallel group_by cannot
            # change the result by reordering float additions.
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume") * PV_SCALE)
            .round()
            .cast(pl.Int64)
            .alias("_pv"),
        )
        .sort("window_start")
        .group_by("ticker", "bar", maintain_order=True)
        .agg(
            pl.col("open").first(),
            pl.col("high").max(),
            pl.col("low").min(),
            pl.col("close").last(),
            pl.col("_vol").sum(),
            pl.col("transactions").sum().alias("trades"),
            pl.col("_pv").sum(),
            pl.len().cast(pl.Int16).alias("n_minutes"),
        )
        .with_columns((pl.col("_vol") / PV_SCALE).alias("volume"))
        .with_columns(
            pl.when(pl.col("volume") > 0)
            .then(pl.col("_pv") / PV_SCALE / pl.col("volume"))
            .otherwise(pl.col("close"))
            .alias("vwap"),
            pl.lit(session).alias("session_date"),
        )
        .collect()
    )
    bars = bars.with_columns((pl.col("close") * pl.col("volume")).alias("_dv")).join(
        mapping, on="ticker", how="left"
    )
    cs = bars.filter(pl.col("type").is_in(list(types)))
    stats = SessionStats(
        session_date=session,
        cs_dollar_volume=math.fsum(cs["_dv"].to_list()),
        unmapped_dollar_volume=math.fsum(
            bars.filter(pl.col("ticker_id").is_null())["_dv"].to_list()
        ),
    )
    end_local = pl.lit(datetime.combine(session, time(9, 30))) + pl.duration(
        minutes=(pl.col("bar").cast(pl.Int64) + 1) * BAR_MINUTES
    )
    cs = one_symbol_per_instrument(cs)
    out = (
        cs.with_columns(
            end_local.dt.replace_time_zone(NY)
            .dt.convert_time_zone("UTC")
            .dt.cast_time_unit("us")
            .alias("ts")
        )
        .select([pl.col(c).cast(t) for c, t in RAW_BAR_SCHEMA.items()])
        .sort("ticker_id", "bar")
    )
    return out, stats


def _cache_key(raw_size: int, symbols_key: str, cfg: DataConfig) -> str:
    return f"v3|{raw_size}|{symbols_key}|{cfg.bars.minutes}|{','.join(cfg.types)}"


def ensure_session_cache(
    layout: DataLayout,
    session: date,
    n_bars: int,
    symbols: pl.DataFrame,
    instruments: pl.DataFrame,
    symbols_key: str,
    cfg: DataConfig,
) -> SessionStats | None:
    """Aggregate a session into the shared cache unless an up-to-date partition exists.

    Returns None when the raw file for the session is missing.
    """
    raw_path = layout.raw_minute(session)
    if not raw_path.exists():
        return None
    key = _cache_key(raw_path.stat().st_size, symbols_key, cfg)
    meta_path, part = layout.bars_cache_meta(session), layout.bars_cache(session)
    if meta_path.exists() and part.exists():
        meta: dict[str, Any] = json.loads(meta_path.read_text())
        if meta["key"] == key:
            return SessionStats.model_validate(meta["stats"])
    mapping = symbols_for_session(symbols, instruments, session)
    df, stats = aggregate_session(
        pl.scan_csv(raw_path, schema_overrides=MINUTE_SCHEMA), session, n_bars, mapping, cfg.types
    )
    part.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(part, compression="zstd", statistics=False)
    meta_path.write_text(json.dumps({"key": key, "stats": json.loads(stats.model_dump_json())}))
    return stats
