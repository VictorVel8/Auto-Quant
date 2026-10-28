"""Synthetic *raw* inputs in Massive's formats: minute flat files and cached reference JSON.

Used to test Plan 2's whole pipeline (bars, adjustment, universe, QA) without licensed data. The
raw price path really drops on split and ex-dividend dates, so a correct adjustment makes returns
continuous. Extended-hours minutes, missing minutes, an ETF and an unmapped symbol are included.
"""

import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl

from alpha_factory.data import massive
from alpha_factory.data.calendar import TradingCalendar
from alpha_factory.data.layout import DataLayout

NY = ZoneInfo("America/New_York")
FETCH_DATE = "2025-01-11"


@dataclass(frozen=True)
class RawAnswers:
    sessions: tuple[date, ...]
    cs_tickers: tuple[str, ...]
    figiless_tickers: tuple[str, ...]
    split: tuple[str, date, float]  # forward split
    reverse_split: tuple[str, date, float]
    dividends: tuple[tuple[str, date, float], ...]
    delisted: tuple[str, date]  # last session with bars
    late_listing: tuple[str, date]  # first session with bars
    penny_ticker: str  # raw price below the $5 filter
    etf_ticker: str
    unmapped_ticker: str


def _sic(i: int) -> str | None:
    return [None, "3674", "2834", "6021", "7372", "4911"][i % 6]


def write_raw_market(
    root: Path,
    seed: int = 11,
    start: date = date(2024, 11, 1),
    end: date = date(2025, 1, 10),
    n_cs: int = 40,
) -> RawAnswers:
    rng = np.random.default_rng(seed)
    layout = DataLayout(root)
    sessions = TradingCalendar(start, end).sessions(start, end)
    cal = TradingCalendar(start, end)
    n_s = len(sessions)

    cs = [f"C{i:03d}" for i in range(n_cs)]
    etf, unmapped = "ETF1", "ZZZZ"
    symbols = [*cs, etf, unmapped]
    n_sym = len(symbols)
    figiless = tuple(t for i, t in enumerate(cs) if i % 5 == 0)

    delisted = (cs[7], sessions[32])
    late = (cs[n_cs - 1], sessions[n_s - 7])
    split = (cs[3], sessions[25], 2.0)
    reverse = (cs[11], sessions[15], 0.25)
    penny = cs[20]
    divs = (
        (cs[5], sessions[20], 0.5),
        (cs[3], sessions[30], 0.4),
        (cs[13], sessions[35], 0.3),
    )
    first_idx = {t: 0 for t in symbols}
    last_idx = {t: n_s - 1 for t in symbols}
    first_idx[late[0]] = sessions.index(late[1])
    last_idx[delisted[0]] = sessions.index(delisted[1])

    # --- reference JSON in the vendor's field names
    tf = massive.TICKER_FIELDS
    active: list[dict[str, object]] = []
    inactive: list[dict[str, object]] = []
    for i, t in enumerate(cs):
        row: dict[str, object] = {
            tf["ticker"]: t,
            tf["name"]: f"Company {t}",
            tf["type"]: "CS",
            tf["primary_exchange"]: "XNAS",
            tf["composite_figi"]: f"BBG{i:09d}",
            tf["share_class_figi"]: None if t in figiless else f"BBGSC{i:07d}",
            tf["delisted_date"]: None,
        }
        if t == delisted[0]:
            row[tf["delisted_date"]] = f"{delisted[1].isoformat()}T00:00:00Z"
            inactive.append(row)
        else:
            active.append(row)
    active.append({tf["ticker"]: etf, tf["name"]: "An ETF", tf["type"]: "ETF"})
    sf, df_ = massive.SPLIT_FIELDS, massive.DIVIDEND_FIELDS
    split_rows = [
        {
            sf["ticker"]: split[0],
            sf["execution_date"]: split[1].isoformat(),
            sf["split_from"]: 1,
            sf["split_to"]: 2,
            sf["adjustment_type"]: "forward_split",
        },
        {
            sf["ticker"]: reverse[0],
            sf["execution_date"]: reverse[1].isoformat(),
            sf["split_from"]: 4,
            sf["split_to"]: 1,
            sf["adjustment_type"]: "reverse_split",
        },
    ]
    div_rows = [
        {
            df_["ticker"]: t,
            df_["ex_date"]: d.isoformat(),
            df_["cash_amount"]: c,
            massive.DIVIDEND_CURRENCY_FIELD: "USD",
        }
        for t, d, c in divs
    ]
    # Like the real service: only active tickers have details (list_date, SIC); delisted 404.
    details: dict[str, dict[str, object]] = {}
    for i, t in enumerate(cs):
        if t == delisted[0]:
            details[t] = {"results": None}
            continue
        details[t] = {
            "results": {
                massive.DETAIL_LIST_DATE_FIELD: late[1].isoformat()
                if t == late[0]
                else "2015-01-02",
                massive.DETAIL_SIC_FIELD: _sic(i),
            }
        }
    payloads = {
        "tickers_active": {"results": active},
        "tickers_inactive": {"results": inactive},
        "splits": {"results": split_rows},
        "dividends": {"results": div_rows},
        massive.DETAILS_ENDPOINT: {"results": details},
    }
    for name, payload in payloads.items():
        d = layout.raw_ref_dir(name)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{FETCH_DATE}.json").write_text(json.dumps(payload, sort_keys=True))

    # --- minute files
    raw_price = rng.uniform(20, 150, n_sym)
    raw_price[symbols.index(penny)] = 3.0
    raw_price[symbols.index(unmapped)] = 12.0
    vol_scale = rng.lognormal(np.log(400), 0.7, n_sym)
    vol_scale[symbols.index(unmapped)] = 3.0  # negligible dollar volume
    # Liquid enough to be members, so their corporate actions and delisting reach the bars.
    for t in {delisted[0], split[0], reverse[0], *(d[0] for d in divs)}:
        vol_scale[symbols.index(t)] *= 10
    sigma = rng.uniform(0.0003, 0.0007, n_sym)
    for si, d in enumerate(sessions):
        for t, _, ratio in (split, reverse):
            if d == (split if t == split[0] else reverse)[1]:
                raw_price[symbols.index(t)] /= ratio
                vol_scale[symbols.index(t)] *= ratio  # raw share volume rises with the share count
        for t, ex, cash in divs:
            if d == ex:
                raw_price[symbols.index(t)] -= cash
        raw_price *= np.exp(rng.standard_normal(n_sym) * 0.004)  # overnight gap

        n_reg = cal.n_bars(d) * 30
        midnight_ns = int(datetime(d.year, d.month, d.day, tzinfo=NY).timestamp()) * 10**9
        # regular minutes plus 5 pre-market and 5 post-market minutes per symbol
        minutes = np.concatenate(
            [np.arange(240, 245), 570 + np.arange(n_reg), 570 + n_reg + np.arange(5)]
        )
        is_regular = (minutes >= 570) & (minutes < 570 + n_reg)
        n_m = len(minutes)
        steps = rng.standard_normal((n_sym, n_m)) * sigma[:, None]
        path = raw_price[:, None] * np.exp(np.cumsum(steps, axis=1))
        open_ = np.concatenate([raw_price[:, None], path[:, :-1]], axis=1)
        raw_price = path[:, -1].copy()
        spread = np.abs(rng.standard_normal((n_sym, n_m))) * 0.0003
        high = np.maximum(open_, path) * (1 + spread)
        low = np.minimum(open_, path) * (1 - spread)
        volume = np.round(vol_scale[:, None] * rng.lognormal(0, 0.5, (n_sym, n_m))) + 1
        keep = np.where(is_regular[None, :], rng.random((n_sym, n_m)) < 0.97, True)
        active_sym = np.array([first_idx[t] <= si <= last_idx[t] for t in symbols])
        keep &= active_sym[:, None]
        sym_i, min_i = np.nonzero(keep)
        frame = pl.DataFrame(
            {
                "ticker": np.array(symbols)[sym_i],
                "volume": volume[sym_i, min_i],
                "open": np.round(open_[sym_i, min_i], 4),
                "close": np.round(path[sym_i, min_i], 4),
                "high": np.round(high[sym_i, min_i], 4),
                "low": np.round(low[sym_i, min_i], 4),
                "window_start": midnight_ns + minutes[min_i].astype(np.int64) * 60 * 10**9,
                "transactions": np.maximum(1, volume[sym_i, min_i] // 100).astype(np.int64),
            }
        ).select(massive.MINUTE_COLUMNS)
        out = layout.raw_minute(d)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as raw_f, gzip.GzipFile(fileobj=raw_f, mode="wb", mtime=0) as f:
            f.write(frame.write_csv().encode())

    return RawAnswers(
        sessions=tuple(sessions),
        cs_tickers=tuple(cs),
        figiless_tickers=figiless,
        split=split,
        reverse_split=reverse,
        dividends=divs,
        delisted=delisted,
        late_listing=late,
        penny_ticker=penny,
        etf_ticker=etf,
        unmapped_ticker=unmapped,
    )
