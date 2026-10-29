"""The dense panel every DSL expression runs on.

One row per (ticker, bar) over each ticker's whole life, sorted by (tix, tsi). A bar with no data
is a row of nulls, never a missing row, so `shift(n)` always means "n bars ago". Null is the
universal missing marker: rolling operators with `min_samples=n` return null across any gap.

Columns: tix, tsi (global bar index), sidx (session index), session_date, bar, last_bar, ts,
open/high/low/close/volume/trades/vwap, raw_close, adj_factor, is_member, rank, tradable, sector,
and the derived DSL fields ret, dayopen, prevclose, barnum.
"""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import polars as pl

from alpha_factory.data.calendar import BAR_MINUTES, OPEN
from alpha_factory.data.marketdata import MarketData

NY = ZoneInfo("America/New_York")

VALUE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trades",
    "vwap",
    "raw_close",
    "adj_factor",
)
DSL_FIELDS = (
    "open", "high", "low", "close", "volume", "trades", "vwap",
    "ret", "dayopen", "prevclose", "bar",
)  # fmt: skip


@dataclass(frozen=True)
class Panel:
    df: pl.DataFrame
    tickers: list[str]  # tix -> ticker_id

    def ticker_ids(self) -> pl.DataFrame:
        """tix -> ticker_id lookup, for turning internal results back into ticker ids."""
        return pl.DataFrame(
            {"tix": pl.Series(range(len(self.tickers)), dtype=pl.UInt32), "ticker_id": self.tickers}
        )


def bar_end(session: date, bar: int) -> datetime:
    """UTC end time of a 30-minute bar of a session."""
    local = datetime.combine(session, time(OPEN.hour, OPEN.minute), tzinfo=NY)
    return (local + timedelta(minutes=(bar + 1) * BAR_MINUTES)).astimezone(ZoneInfo("UTC"))


def make_grid(session_bars: dict[date, int]) -> pl.DataFrame:
    """Every (session, bar) of the calendar with its end timestamp and a global bar index."""
    rows = [
        (d, i, n - 1, bar_end(d, b), b)
        for i, (d, n) in enumerate(sorted(session_bars.items()))
        for b in range(n)
    ]
    return pl.DataFrame(
        {
            "session_date": [r[0] for r in rows],
            "sidx": pl.Series([r[1] for r in rows], dtype=pl.UInt32),
            "last_bar": pl.Series([r[2] for r in rows], dtype=pl.Int8),
            "ts": pl.Series([r[3] for r in rows], dtype=pl.Datetime("us", "UTC")),
            "bar": pl.Series([r[4] for r in rows], dtype=pl.Int8),
        }
    ).with_row_index("tsi")


def derive_fields(df: pl.DataFrame) -> pl.DataFrame:
    """Add ret, dayopen and prevclose. Every one uses only the current and earlier bars."""
    by_ticker = ["tix"]
    by_session = ["tix", "sidx"]
    out = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over(by_ticker) - 1).alias("ret"),
        pl.col("bar").cast(pl.Float64).alias("barnum"),
        pl.when(pl.col("bar") == 0)
        .then(pl.col("open"))
        .otherwise(None)
        .forward_fill()
        .over(by_session)
        .alias("dayopen"),
    )
    last_close = df.filter(pl.col("bar") == pl.col("last_bar")).select(
        "tix", (pl.col("sidx") + 1).alias("sidx"), pl.col("close").alias("prevclose")
    )
    return out.join(last_close, on=["tix", "sidx"], how="left").with_columns(
        pl.col("ret").fill_nan(None).alias("ret")
    )


def build_panel(
    bars: pl.DataFrame,
    universe: pl.DataFrame,
    sectors: pl.DataFrame,
    session_bars: dict[date, int],
    tickers: list[str] | None = None,
) -> Panel:
    """Densify bars over each ticker's life and attach membership, sector and derived fields.

    bars: Plan 2 bars_30m rows. universe: Plan 2 universe rows. sectors: ticker_id, sector.
    session_bars: bars in each session (13, or 7 on half-days).
    tickers: the ticker_id -> tix numbering. Pass the build's full list (load_panel does) so that
    tix means the same thing for every date range; the default numbers the tickers seen in bars.
    """
    grid = make_grid(session_bars)
    tickers = tickers if tickers is not None else sorted(bars["ticker_id"].unique().to_list())
    tmap = pl.DataFrame(
        {"ticker_id": tickers, "tix": pl.Series(range(len(tickers)), dtype=pl.UInt32)}
    )
    keyed = (
        bars.join(grid.select("tsi", "session_date", "bar"), on=["session_date", "bar"])
        .join(tmap, on="ticker_id")
        .select("tix", "tsi", *VALUE_COLUMNS)
    )
    life = keyed.group_by("tix").agg(
        pl.col("tsi").min().alias("first"), pl.col("tsi").max().alias("last")
    )
    rows = (
        life.with_columns(pl.int_ranges("first", pl.col("last") + 1, dtype=pl.UInt32).alias("tsi"))
        .select("tix", "tsi")
        .explode("tsi")
    )
    members = universe.join(tmap, on="ticker_id").select("tix", "session_date", "rank", "tradable")
    sector_map = sectors.join(tmap, on="ticker_id").select(
        "tix", pl.col("sector").cast(pl.Categorical)
    )
    df = (
        rows.join(grid, on="tsi")
        .join(keyed, on=["tix", "tsi"], how="left")
        .join(members, on=["tix", "session_date"], how="left")
        .join(sector_map, on="tix", how="left")
        .with_columns(
            pl.col("rank").is_not_null().alias("is_member"),
            pl.col("tradable").fill_null(False),
        )
        .sort("tix", "tsi")
    )
    return Panel(derive_fields(df), tickers)


def load_panel(data: MarketData, start: date, end: date, *, warmup_sessions: int = 15) -> Panel:
    """Panel for sessions [start, end] plus `warmup_sessions` earlier ones for window history."""
    sessions = data.sessions()
    lo = max(0, sum(1 for d in sessions if d < start) - warmup_sessions)
    chosen = [d for d in sessions[lo:] if d <= end]
    columns = ["session_date", "bar", "ticker_id", *VALUE_COLUMNS]
    bars = data.bars(chosen[0], chosen[-1], columns=columns, members_only=False).collect()
    # Bars per session come from the data (13, or 7 on half-days), so no calendar is needed.
    counts = bars.group_by("session_date").agg((pl.col("bar").max() + 1).alias("n"))
    session_bars = {d: int(n) for d, n in zip(counts["session_date"], counts["n"], strict=True)}
    universe = data.universe(chosen[0], chosen[-1]).collect()
    sectors = data.sectors()
    return build_panel(
        bars, universe, sectors, session_bars, sorted(sectors["ticker_id"].to_list())
    )
