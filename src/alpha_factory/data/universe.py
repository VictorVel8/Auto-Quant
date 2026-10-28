"""Point-in-time universe: session d is fixed before d opens, from data through d-1 only."""

from datetime import date

import numpy as np
import polars as pl

from alpha_factory.data.config import UniverseConfig

UNIVERSE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "session_date": pl.Date,
    "ticker_id": pl.String,
    "rank": pl.Int32,
    "adv_usd_20d": pl.Float64,
    "raw_prev_close": pl.Float64,
    "tradable": pl.Boolean,
}


def build_universe(
    daily: pl.DataFrame, instruments: pl.DataFrame, sessions: list[date], cfg: UniverseConfig
) -> pl.DataFrame:
    """Top `top_n` eligible names per session by trailing dollar volume.

    daily: ticker_id, session_date, raw_close (last bar), dollar_volume (sum of raw close * volume).
    The first `adv_window_days` sessions have no universe. adv is the sum of daily dollar volume
    over the previous window divided by the window length (absent days count as zero).
    """
    window = cfg.adv_window_days
    if daily.is_empty() or len(sessions) <= window:
        return pl.DataFrame(schema=UNIVERSE_SCHEMA)

    ticker_ids = sorted(daily["ticker_id"].unique().to_list())
    t_index = {t: i for i, t in enumerate(ticker_ids)}
    s_index = {d: i for i, d in enumerate(sessions)}
    n_t, n_s = len(ticker_ids), len(sessions)

    rows_t = np.array([t_index[t] for t in daily["ticker_id"].to_list()])
    rows_s = np.array([s_index[d] for d in daily["session_date"].to_list()])
    dv = np.zeros((n_t, n_s))
    close = np.full((n_t, n_s), np.nan)
    present = np.zeros((n_t, n_s), dtype=np.int64)
    dv[rows_t, rows_s] = daily["dollar_volume"].to_numpy()
    close[rows_t, rows_s] = daily["raw_close"].to_numpy()
    present[rows_t, rows_s] = 1

    # Prefix sums: window for session i covers columns i-window .. i-1.
    cum_dv = np.concatenate([np.zeros((n_t, 1)), np.cumsum(dv, axis=1)], axis=1)
    cum_pr = np.concatenate(
        [np.zeros((n_t, 1), dtype=np.int64), np.cumsum(present, axis=1)], axis=1
    )
    # Last known raw close at or before each session.
    last_idx = np.maximum.accumulate(np.where(present == 1, np.arange(n_s)[None, :], -1), axis=1)
    ffill = np.where(
        last_idx >= 0, np.take_along_axis(close, np.maximum(last_idx, 0), axis=1), np.nan
    )

    info = instruments.select("ticker_id", "list_date", "delisted_date").to_dict(as_series=False)
    lookup = {
        t: (ld, dd)
        for t, ld, dd in zip(
            info["ticker_id"], info["list_date"], info["delisted_date"], strict=True
        )
    }
    list_dates = [lookup.get(t, (None, None))[0] for t in ticker_ids]
    delist_dates = [lookup.get(t, (None, None))[1] for t in ticker_ids]

    out_session: list[date] = []
    out_ticker: list[str] = []
    out_rank: list[int] = []
    out_adv: list[float] = []
    out_close: list[float] = []
    out_tradable: list[bool] = []
    for i in range(window, n_s):
        d = sessions[i]
        count = cum_pr[:, i] - cum_pr[:, i - window]
        adv = (cum_dv[:, i] - cum_dv[:, i - window]) / window
        prev_close = ffill[:, i - 1]
        listed = np.array([ld is None or ld < d for ld in list_dates])
        alive = np.array([dd is None or dd >= d for dd in delist_dates])
        eligible = (
            (count >= cfg.min_sessions_present)
            & (np.nan_to_num(prev_close, nan=0.0) >= cfg.min_price)
            & listed
            & alive
        )
        cand = np.nonzero(eligible)[0]
        # Highest adv first; ticker_ids are sorted so a stable sort breaks ties by id.
        order = cand[np.argsort(-adv[cand], kind="stable")][: cfg.top_n]
        for rank, t in enumerate(order, start=1):
            out_session.append(d)
            out_ticker.append(ticker_ids[t])
            out_rank.append(rank)
            out_adv.append(float(adv[t]))
            out_close.append(float(prev_close[t]))
            out_tradable.append(rank <= cfg.tradable_top_n)

    return pl.DataFrame(
        {
            "session_date": out_session,
            "ticker_id": out_ticker,
            "rank": out_rank,
            "adv_usd_20d": out_adv,
            "raw_prev_close": out_close,
            "tradable": out_tradable,
        }
    ).select([pl.col(c).cast(t) for c, t in UNIVERSE_SCHEMA.items()])
