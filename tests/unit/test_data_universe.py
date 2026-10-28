from datetime import date, timedelta

import polars as pl

from alpha_factory.data.config import UniverseConfig
from alpha_factory.data.universe import build_universe

S = [date(2024, 1, 1) + timedelta(days=i) for i in range(30)]
CFG = UniverseConfig(
    top_n=2, tradable_top_n=1, adv_window_days=5, min_price=5.0, min_sessions_present=4
)


def rows(ticker: str, dv: float, close: float = 50.0, days: range | list[int] | None = None):
    return [
        {"ticker_id": ticker, "session_date": S[i], "raw_close": close, "dollar_volume": dv}
        for i in (days if days is not None else range(len(S)))
    ]


def inst(*specs: tuple[str, date | None, date | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker_id": [s[0] for s in specs],
            "list_date": [s[1] for s in specs],
            "delisted_date": [s[2] for s in specs],
        },
        schema={"ticker_id": pl.String, "list_date": pl.Date, "delisted_date": pl.Date},
    )


def frame(*parts: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame([r for p in parts for r in p])


INSTRUMENTS = inst(
    ("A", None, None),
    ("B", None, None),
    ("C", None, None),
    ("PENNY", None, None),
    ("SPARSE", None, None),
    ("GONE", None, S[12]),
    ("NEW", S[8], None),
)


def test_warmup_and_membership() -> None:
    daily = frame(rows("A", 100), rows("B", 50), rows("C", 10), rows("PENNY", 1000, close=3.0))
    u = build_universe(daily, INSTRUMENTS, S, CFG)
    assert u["session_date"].min() == S[5]  # the 6th session: 5 sessions of history
    per = u.group_by("session_date").len()
    assert per["len"].to_list() == [2] * (len(S) - 5)
    first = u.filter(pl.col("session_date") == S[5]).sort("rank")
    assert first["ticker_id"].to_list() == ["A", "B"]
    assert first["tradable"].to_list() == [True, False]
    assert first["adv_usd_20d"].to_list() == [100.0, 50.0]
    assert first["raw_prev_close"].to_list() == [50.0, 50.0]
    assert "PENNY" not in u["ticker_id"].to_list()  # price below $5


def test_min_sessions_present() -> None:
    daily = frame(rows("A", 100), rows("B", 50), rows("SPARSE", 500, days=[0, 1, 2]))
    u = build_universe(daily, INSTRUMENTS, S, CFG)
    assert "SPARSE" not in u["ticker_id"].to_list()  # only 3 of 5 previous sessions
    daily = frame(rows("A", 100), rows("B", 50), rows("SPARSE", 500, days=[0, 1, 2, 3]))
    u = build_universe(daily, INSTRUMENTS, S, CFG)
    assert u.filter(pl.col("session_date") == S[5]).sort("rank")["ticker_id"][0] == "SPARSE"


def test_a_name_ranks_only_from_the_session_after_its_volume_arrives() -> None:
    spike = [
        {
            "ticker_id": "C",
            "session_date": S[i],
            "raw_close": 50.0,
            "dollar_volume": 10_000.0 if i == 12 else 1.0,
        }
        for i in range(len(S))
    ]
    u = build_universe(frame(rows("A", 100), rows("B", 50), spike), INSTRUMENTS, S, CFG)
    assert "C" not in u.filter(pl.col("session_date") <= S[12])["ticker_id"].to_list()
    assert u.filter(pl.col("session_date") == S[13]).sort("rank")["ticker_id"][0] == "C"


def test_delisted_names_stay_members_through_their_last_session() -> None:
    daily = frame(rows("A", 100), rows("B", 50), rows("GONE", 500, days=range(0, 13)))
    u = build_universe(daily, INSTRUMENTS, S, CFG)
    gone = u.filter(pl.col("ticker_id") == "GONE")["session_date"]
    assert gone.min() == S[5] and gone.max() == S[12]  # member on the delisting session itself


def test_new_listings_wait_for_their_list_date() -> None:
    daily = frame(rows("A", 100), rows("B", 50), rows("NEW", 500, days=range(0, 30)))
    u = build_universe(daily, INSTRUMENTS, S, CFG)
    assert u.filter(pl.col("ticker_id") == "NEW")["session_date"].min() == S[9]  # list_date < d


def test_ties_break_by_ticker_id() -> None:
    daily = frame(rows("B", 10), rows("A", 10), rows("C", 10))
    u = build_universe(daily, INSTRUMENTS, S, CFG)
    assert u.filter(pl.col("session_date") == S[5]).sort("rank")["ticker_id"].to_list() == [
        "A",
        "B",
    ]


def test_universe_is_causal() -> None:
    """Rows for early sessions are identical whether or not later sessions exist."""
    daily = frame(
        rows("A", 100),
        rows("B", 50),
        rows("C", 10),
        [{"ticker_id": "C", "session_date": S[20], "raw_close": 50.0, "dollar_volume": 9e5}],
    )
    full = build_universe(daily, INSTRUMENTS, S, CFG)
    for k in (8, 15, 22):
        cut = build_universe(
            daily.filter(pl.col("session_date") <= S[k - 1]), INSTRUMENTS, S[:k], CFG
        )
        assert cut.equals(full.filter(pl.col("session_date") <= S[k - 1]))


def test_too_short_history_gives_an_empty_universe() -> None:
    assert build_universe(frame(rows("A", 1)), INSTRUMENTS, S[:5], CFG).is_empty()
