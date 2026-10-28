# pyright: basic
"""XNYS sessions, half-days and bars per session."""

from datetime import date, time, timedelta
from typing import Any

import exchange_calendars as xc
import pandas as pd


def _ts(d: date) -> Any:
    return pd.Timestamp(d.isoformat())


OPEN = time(9, 30)
BAR_MINUTES = 30


class TradingCalendar:
    def __init__(self, start: date, end: date) -> None:
        # Bounds a little wider than requested so close lookups at the edges work.
        self._cal = xc.get_calendar(
            "XNYS",
            start=_ts(start - timedelta(days=7)),
            end=_ts(end + timedelta(days=7)),
        )

    def sessions(self, start: date, end: date) -> list[date]:
        idx = self._cal.sessions_in_range(_ts(start), _ts(end))
        return [ts.date() for ts in idx]

    def n_bars(self, session: date) -> int:
        """13 on a normal day, 7 when the market closes at 13:00 ET."""
        close = self._cal.session_close(_ts(session)).tz_convert("America/New_York")
        minutes = close.hour * 60 + close.minute - (OPEN.hour * 60 + OPEN.minute)
        return -(-minutes // BAR_MINUTES)

    def is_session(self, session: date) -> bool:
        return bool(self._cal.is_session(_ts(session)))
