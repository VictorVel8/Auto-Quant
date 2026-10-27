import time
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import polars as pl
import pytest

from alpha_factory.contracts import DataManifest
from alpha_factory.contracts.data import BAR_SCHEMA
from alpha_factory.hashing import sha256_file
from alpha_factory.synthetic.market import SyntheticMarket


def bars(d: Path) -> pl.DataFrame:
    return pl.read_parquet(d / "bars" / "*.parquet")


def test_build_under_ten_seconds(tmp_path: Path) -> None:
    t0 = time.perf_counter()
    m = SyntheticMarket().build(tmp_path)
    assert time.perf_counter() - t0 < 10
    assert 1.2e6 < sum(f.rows for f in m.files if f.path.startswith("bars/")) < 1.35e6


def test_same_seed_same_bytes(tmp_path: Path) -> None:
    a = SyntheticMarket(seed=1, n_tickers=40, n_sessions=80).build(tmp_path / "a")
    b = SyntheticMarket(seed=1, n_tickers=40, n_sessions=80).build(tmp_path / "b")
    c = SyntheticMarket(seed=2, n_tickers=40, n_sessions=80).build(tmp_path / "c")
    assert a.dataset_hash == b.dataset_hash
    assert a.dataset_hash != c.dataset_hash
    for f in a.files:
        assert sha256_file(tmp_path / "a" / f.path) == sha256_file(tmp_path / "b" / f.path)


def test_manifest_is_consistent(synth_dir: Path) -> None:
    m = DataManifest.model_validate_json((synth_dir / "manifest.json").read_text())
    assert m.source == "synthetic"
    assert m.n_sessions == 500
    for f in m.files:
        p = synth_dir / f.path
        assert sha256_file(p) == f.sha256
        assert p.stat().st_size == f.bytes
        assert pl.read_parquet(p).height == f.rows


def test_bar_schema_exact(synth_dir: Path) -> None:
    df = bars(synth_dir)
    assert dict(df.schema) == dict(BAR_SCHEMA)
    assert df["bar"].min() == 0
    assert df["bar"].max() == 12
    assert df.select((pl.col("high") >= pl.col("low")).all()).item()
    assert df.select(
        ((pl.col("vwap") >= pl.col("low")) & (pl.col("vwap") <= pl.col("high"))).all()
    ).item()


def test_half_days_have_seven_bars(synth_dir: Path) -> None:
    answers = SyntheticMarket().answers
    df = bars(synth_dir)
    assert len(answers.half_days) == 3
    for day in answers.half_days:
        assert df.filter(pl.col("session_date") == day)["bar"].max() == 6
    full = df.filter(~pl.col("session_date").is_in(list(answers.half_days)))
    assert full["bar"].max() == 12


def test_missing_bars_near_half_percent(synth_dir: Path) -> None:
    answers = SyntheticMarket().answers
    df = bars(synth_dir).filter(~pl.col("session_date").is_in(list(answers.half_days)))
    # Each ticker-session should have 13 bars; a fully missing session is negligible.
    expected = df.select(pl.struct("session_date", "ticker_id").n_unique()).item() * 13
    missing = 1 - df.height / expected
    assert 0.003 < missing < 0.008


def test_splits_are_continuous_in_adjusted_prices(synth_dir: Path) -> None:
    answers = SyntheticMarket().answers
    assert len(answers.splits) == 5
    df = bars(synth_dir)
    for ev in answers.splits:
        t = df.filter(pl.col("ticker_id") == ev.ticker_id).sort("ts")
        daily = t.group_by("session_date", maintain_order=True).agg(
            pl.col("close").last().alias("adj_close"),
            pl.col("raw_close").last().alias("raw"),
            pl.col("adj_factor").last().alias("f"),
        )
        i = daily["session_date"].to_list().index(ev.session_date)
        before, after = daily.row(i - 1, named=True), daily.row(i, named=True)
        assert after["f"] == before["f"] * ev.ratio
        # Raw price roughly halves while adjusted price stays continuous (daily moves are ~1%).
        assert after["raw"] / before["raw"] == pytest.approx(0.5, rel=0.1)
        assert after["adj_close"] / before["adj_close"] == pytest.approx(1.0, rel=0.1)


def test_lifecycle_and_universe(synth_dir: Path) -> None:
    answers = SyntheticMarket().answers
    df = bars(synth_dir)
    uni = pl.read_parquet(synth_dir / "universe.parquet")
    assert len(answers.delistings) == 10 and len(answers.listings) == 5
    for ev in answers.delistings:
        last = df.filter(pl.col("ticker_id") == ev.ticker_id)["session_date"].max()
        assert last == ev.session_date
        # Present in the universe before it delists, absent afterwards.
        u = uni.filter((pl.col("ticker_id") == ev.ticker_id) & pl.col("in_universe"))
        assert u["session_date"].min() is not None
        assert cast(date, u["session_date"].max()) <= ev.session_date
        assert u.filter(pl.col("session_date") < ev.session_date).height > 0
    for ev in answers.listings:
        first = df.filter(pl.col("ticker_id") == ev.ticker_id)["session_date"].min()
        assert first == ev.session_date


def test_universe_is_point_in_time(synth_dir: Path) -> None:
    uni = pl.read_parquet(synth_dir / "universe.parquet")
    first_session = cast(date, bars(synth_dir)["session_date"].min())
    assert uni["session_date"].min() == first_session + timedelta(days=1)
    assert set(uni.columns) == {"session_date", "ticker_id", "adv20_usd", "rank", "in_universe"}


def test_sectors(synth_dir: Path) -> None:
    s = pl.read_parquet(synth_dir / "sectors.parquet")
    assert s.height == 200
    assert s["sector"].n_unique() == 10


def test_planted_ic_is_recovered(synth_dir: Path) -> None:
    """Mean per-bar Spearman IC of the planted signal against the h1 label is near planted_ic."""
    ic = _mean_ic(bars(synth_dir), k=2)
    assert ic == pytest.approx(0.03, abs=0.01)


def test_planted_ic_scales_and_null_is_centred(tmp_path: Path) -> None:
    hi = SyntheticMarket(seed=3, n_tickers=200, n_sessions=300, planted_ic=0.08)
    hi.build(tmp_path / "hi")
    assert _mean_ic(bars(tmp_path / "hi"), k=2) == pytest.approx(0.08, abs=0.015)
    zero = SyntheticMarket(seed=3, n_tickers=200, n_sessions=300, planted_ic=0.0)
    zero.build(tmp_path / "zero")
    assert _mean_ic(bars(tmp_path / "zero"), k=2) == pytest.approx(0.0, abs=0.01)


def test_unplantable_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot be planted"):
        SyntheticMarket(planted_horizon="h4")
    assert SyntheticMarket(planted_horizon="h2").answers.signal_expression == "-Pct($close, 4)"


def _mean_ic(df: pl.DataFrame, k: int) -> float:
    """Spearman IC of -(close_t / close_t-k - 1) vs close_t+k / close_t - 1, per session and bar."""
    g = ["session_date", "ticker_id"]
    d = (
        df.sort(["ticker_id", "ts"])
        .with_columns(
            pl.col("close").shift(k).over(g).alias("c_prev"),
            pl.col("close").shift(-k).over(g).alias("c_next"),
            pl.col("bar").shift(k).over(g).alias("b_prev"),
            pl.col("bar").shift(-k).over(g).alias("b_next"),
        )
        .filter((pl.col("b_prev") == pl.col("bar") - k) & (pl.col("b_next") == pl.col("bar") + k))
        .with_columns(
            (-(pl.col("close") / pl.col("c_prev") - 1)).alias("sig"),
            (pl.col("c_next") / pl.col("close") - 1).alias("lab"),
        )
        .with_columns(
            pl.col("sig").rank().over(["session_date", "bar"]).alias("rs"),
            pl.col("lab").rank().over(["session_date", "bar"]).alias("rl"),
        )
    )
    per = d.group_by(["session_date", "bar"]).agg(pl.corr("rs", "rl").alias("ic"))
    return float(per["ic"].mean())  # type: ignore[arg-type]
