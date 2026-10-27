"""Seeded synthetic market that writes exactly the files Plan 2 produces from Massive.

Price model, per 30-min bar ``t`` of a session (log returns):

    r_t = beta_i * m_t + s_sector(i),t + eta_t + tau_t - tau_(t-1)

``m_t`` is a market factor with two volatility regimes, ``s`` are 10 sector factors, ``eta`` is
permanent idiosyncratic noise and ``tau`` is a transient shock that reverts one bar later. The
k-bar past return is ``sum(eta) + tau_t - tau_(t-k)`` and the next k-bar return is
``sum(eta) + tau_(t+k) - tau_t``, so ``-past`` predicts ``next`` with covariance ``var(tau)``.
``var(tau)`` is solved from the target IC, which makes the planted signal closed-form.

Conventions:
- Sessions are plain weekdays (no holidays); half-days have 7 bars.
- Forward adjustment: ``adj_factor`` is the product of split ratios on or before the session,
  adjusted price = raw price * adj_factor, adjusted volume = raw volume / adj_factor. Builds
  therefore never change when later data is added.
- The universe file is point-in-time: ADV and the price filter use only prior sessions, so it
  starts on the second session.
"""

import subprocess
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from alpha_factory.contracts.base import Horizon, Record
from alpha_factory.contracts.data import BAR_SCHEMA, DataManifest, FileEntry
from alpha_factory.hashing import sha256_canonical, sha256_file

N_BARS = 13
HALF_DAY_BARS = 7
N_SECTORS = 10
N_SPLITS = 5
N_DELISTINGS = 10
N_LISTINGS = 5
N_HALF_DAYS = 3
MISSING_BAR_FRAC = 0.005
SIGMA_ETA = 0.003  # permanent idiosyncratic vol per bar
SIGMA_SECTOR = 0.0015
SIGMA_MARKET = (0.001, 0.0025)  # normal and high-vol regimes
HORIZON_BARS: dict[str, int] = {"h1": 2, "h2": 4, "h4": 8}
# Fixed so that "same seed, same bytes" holds for the manifest too.
BUILD_TIMESTAMP = datetime(2000, 1, 1, tzinfo=UTC)


class SplitEvent(Record):
    ticker_id: str
    session_date: date
    ratio: int


class LifecycleEvent(Record):
    ticker_id: str
    session_date: date  # first session (listing) or last session (delisting)


class SyntheticAnswers(Record):
    """Planted facts that later plans test against."""

    planted_ic: float
    planted_horizon: Horizon
    signal_expression: str  # the DSL expression whose IC against the label equals planted_ic
    splits: tuple[SplitEvent, ...]
    delistings: tuple[LifecycleEvent, ...]  # session_date is the last session with bars
    listings: tuple[LifecycleEvent, ...]  # session_date is the first session with bars
    half_days: tuple[date, ...]
    high_vol_sessions: tuple[date, ...]
    missing_bar_frac: float


@dataclass(frozen=True)
class SyntheticFrames:
    bars: pl.DataFrame
    universe: pl.DataFrame
    sectors: pl.DataFrame


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _weekdays(start: date, n: int) -> list[date]:
    out: list[date] = []
    d = start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


class SyntheticMarket:
    def __init__(
        self,
        seed: int = 7,
        n_tickers: int = 200,
        n_sessions: int = 500,
        start: date = date(2022, 1, 3),
        planted_ic: float = 0.03,
        planted_horizon: Horizon = "h1",
    ) -> None:
        if n_tickers < 30 or n_sessions < 60:
            raise ValueError("need at least 30 tickers and 60 sessions")
        if not 0 <= planted_ic < 0.2:
            raise ValueError("planted_ic must be in [0, 0.2)")
        k = HORIZON_BARS[planted_horizon]
        if 2 * k > N_BARS - 1 or k < 2:
            raise ValueError(
                f"{planted_horizon} cannot be planted: no bar has both a {k}-bar lookback "
                f"and a {k}-bar forward window inside one {N_BARS}-bar session"
            )
        self.seed = seed
        self.n_tickers = n_tickers
        self.n_sessions = n_sessions
        self.start = start
        self.planted_ic = planted_ic
        self.planted_horizon: Horizon = planted_horizon
        self._k = k
        self.sessions = _weekdays(start, n_sessions)
        self._plan_events()

    # ------------------------------------------------------------------ planning

    def _params(self) -> dict[str, object]:
        return {
            "generator": "synthetic-v1",
            "seed": self.seed,
            "n_tickers": self.n_tickers,
            "n_sessions": self.n_sessions,
            "start": self.start.isoformat(),
            "planted_ic": self.planted_ic,
            "planted_horizon": self.planted_horizon,
        }

    def _tid(self, i: int) -> str:
        return f"syn_{i:04d}"

    def _plan_events(self) -> None:
        """Draw the corporate actions, lifecycle, calendar quirks and regimes."""
        rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(2)[0])
        n, s = self.n_tickers, self.n_sessions
        picks = rng.permutation(n)
        delist_idx = picks[:N_DELISTINGS]
        list_idx = picks[N_DELISTINGS : N_DELISTINGS + N_LISTINGS]
        split_idx = picks[N_DELISTINGS + N_LISTINGS : N_DELISTINGS + N_LISTINGS + N_SPLITS]

        self.first_session = np.zeros(n, dtype=np.int64)
        self.last_session = np.full(n, s - 1, dtype=np.int64)
        self.last_session[delist_idx] = rng.integers(int(0.3 * s), int(0.9 * s), N_DELISTINGS)
        self.first_session[list_idx] = rng.integers(int(0.1 * s), int(0.7 * s), N_LISTINGS)
        self.split_session = {
            int(i): int(rng.integers(int(0.2 * s), int(0.8 * s))) for i in split_idx
        }
        half = rng.choice(np.arange(30, s - 1), N_HALF_DAYS, replace=False)
        self.half_day_idx = sorted(int(x) for x in half)

        # Two-state daily volatility regime (Markov chain), starting calm.
        regime = np.zeros(s, dtype=np.int64)
        for d in range(1, s):
            stay = 0.97 if regime[d - 1] == 0 else 0.90
            regime[d] = regime[d - 1] if rng.random() < stay else 1 - regime[d - 1]
        self.regime = regime

        self._answers = SyntheticAnswers(
            planted_ic=self.planted_ic,
            planted_horizon=self.planted_horizon,
            signal_expression=f"-Pct($close, {self._k})",
            splits=tuple(
                SplitEvent(ticker_id=self._tid(i), session_date=self.sessions[d], ratio=2)
                for i, d in sorted(self.split_session.items())
            ),
            delistings=tuple(
                LifecycleEvent(ticker_id=self._tid(int(i)), session_date=self.sessions[int(d)])
                for i, d in sorted((int(i), self.last_session[i]) for i in delist_idx)
            ),
            listings=tuple(
                LifecycleEvent(ticker_id=self._tid(int(i)), session_date=self.sessions[int(d)])
                for i, d in sorted((int(i), self.first_session[i]) for i in list_idx)
            ),
            half_days=tuple(self.sessions[d] for d in self.half_day_idx),
            high_vol_sessions=tuple(self.sessions[d] for d in range(s) if regime[d] == 1),
            missing_bar_frac=MISSING_BAR_FRAC,
        )

    @property
    def answers(self) -> SyntheticAnswers:
        """Planted facts later plans test against."""
        return self._answers

    # ---------------------------------------------------------------- generation

    def _tau_sigma(self) -> float:
        """Std of the transient shock so the Spearman IC of -Pct(close, k) equals planted_ic."""
        # Gaussian Spearman -> Pearson: rho_p = 2 sin(pi * rho_s / 6).
        rho = 2 * np.sin(np.pi * self.planted_ic / 6)
        # corr = var_tau / (k * (var_eta + var_sector) + 2 * var_tau), solved for var_tau.
        var_tau = rho * self._k * (SIGMA_ETA**2 + SIGMA_SECTOR**2) / (1 - 2 * rho)
        return float(np.sqrt(var_tau))

    def frames(self) -> SyntheticFrames:
        """Generate bars, the point-in-time universe and sectors in memory."""
        rng = np.random.default_rng(np.random.SeedSequence(self.seed).spawn(2)[1])
        s, b, n = self.n_sessions, N_BARS, self.n_tickers

        # Return components, shape (session, bar, ticker).
        vol_scale = rng.lognormal(0.0, 0.25, n)
        beta = rng.normal(1.0, 0.2, n)
        sector_of = rng.permutation(np.arange(n) % N_SECTORS)
        sigma_m = np.where(self.regime == 1, SIGMA_MARKET[1], SIGMA_MARKET[0])
        market = rng.standard_normal((s, b)) * sigma_m[:, None]
        sector = rng.standard_normal((s, b, N_SECTORS)) * SIGMA_SECTOR
        eta = rng.standard_normal((s, b, n)) * SIGMA_ETA
        tau = rng.standard_normal((s, b, n)) * self._tau_sigma()
        tau_prev = np.concatenate([np.zeros((s, 1, n)), tau[:, :-1, :]], axis=1)
        ret = (
            market[:, :, None] * beta[None, None, :]
            + sector[:, :, sector_of]
            + (eta + tau - tau_prev)
        ) * vol_scale[None, None, :]
        gap = rng.standard_normal((s, n)) * 0.004 * vol_scale[None, :]

        # Half-days only have 7 bars: later steps do not exist.
        bar_exists = np.ones((s, b), dtype=bool)
        for d in self.half_day_idx:
            bar_exists[d, HALF_DAY_BARS:] = False
        ret *= bar_exists[:, :, None]

        # Log-price path. Bar 0 opens at the previous close plus the overnight gap.
        step = ret.copy()
        step[:, 0, :] += gap
        close_log = np.cumsum(step.reshape(s * b, n), axis=0).reshape(s, b, n)
        open_log = np.concatenate(
            [np.zeros((1, 1, n)), close_log[:-1, -1:, :]], axis=0
        )  # previous session's last close
        open_log = np.concatenate([open_log + gap[:, None, :], close_log[:, :-1, :]], axis=1)
        p0 = rng.uniform(20, 200, n)
        close_p = p0 * np.exp(close_log)
        open_p = p0 * np.exp(open_log)
        hl_up = np.abs(rng.standard_normal((s, b, n))) * 0.0015
        hl_dn = np.abs(rng.standard_normal((s, b, n))) * 0.0015
        high_p = np.maximum(open_p, close_p) * np.exp(hl_up)
        low_p = np.minimum(open_p, close_p) * np.exp(-hl_dn)
        vwap_p = np.clip(
            (open_p + close_p + high_p + low_p) / 4 * np.exp(rng.standard_normal((s, b, n)) * 1e-4),
            low_p,
            high_p,
        )

        # Split adjustment factors and volume (U-shaped intraday profile).
        adj = np.ones((s, n))
        for i, d in self.split_session.items():
            adj[d:, i] *= 2.0
        u = 1.0 + 1.5 * ((np.arange(b) - (b - 1) / 2) / ((b - 1) / 2)) ** 2
        u /= u.mean()
        base_vol = rng.lognormal(np.log(2e5), 0.8, n)  # adjusted shares per bar
        volume = base_vol[None, None, :] * u[None, :, None] * rng.lognormal(0, 0.4, (s, b, n))
        volume = np.round(volume)
        trades = np.maximum(1, np.round(volume / rng.uniform(80, 150, (s, b, n)))).astype(np.int64)
        n_min = rng.binomial(30, 0.97, (s, b, n)).astype(np.int16)

        # Which (session, bar, ticker) rows exist.
        sess_idx = np.arange(s)[:, None, None]
        active = (sess_idx >= self.first_session) & (sess_idx <= self.last_session)
        present = active & bar_exists[:, :, None] & (rng.random((s, b, n)) >= MISSING_BAR_FRAC)
        si, bi, ti = np.nonzero(present)

        # Timestamps: bar b ends at 09:30 + 30 * (b + 1) minutes New York time.
        local = np.array(
            [
                np.datetime64(datetime.combine(d, datetime.min.time()), "us")
                + np.timedelta64(9 * 60 + 30, "m")
                for d in self.sessions
            ]
        )
        local_ends = local[:, None] + (np.arange(1, b + 1) * np.timedelta64(30, "m"))[None, :]
        ts = (
            pl.Series(local_ends.reshape(-1).astype("datetime64[us]"))
            .dt.replace_time_zone("America/New_York")
            .dt.convert_time_zone("UTC")
            .to_numpy()
            .reshape(s, b)
        )
        ts_us = ts.astype("datetime64[us]").astype(np.int64)

        ids = np.array([self._tid(i) for i in range(n)])
        symbols = np.array([f"S{i:04d}" for i in range(n)])
        close_adj = np.round(close_p, 4)
        factor = adj[si, ti]
        bars = pl.DataFrame(
            {
                "session_date": pl.Series([self.sessions[x] for x in range(s)])[si],
                "ts": pl.Series(ts_us[si, bi]).cast(pl.Datetime("us")).dt.replace_time_zone("UTC"),
                "bar": bi.astype(np.int8),
                "ticker_id": ids[ti],
                "ticker": symbols[ti],
                "open": np.round(open_p[si, bi, ti], 4),
                "high": np.round(high_p[si, bi, ti], 4),
                "low": np.round(low_p[si, bi, ti], 4),
                "close": close_adj[si, bi, ti],
                "raw_close": np.round(close_adj[si, bi, ti] / factor, 4),
                "adj_factor": factor,
                "volume": volume[si, bi, ti],
                "trades": trades[si, bi, ti],
                "vwap": np.round(vwap_p[si, bi, ti], 4),
                "n_minutes": n_min[si, bi, ti],
            }
        ).select([pl.col(c).cast(t) for c, t in BAR_SCHEMA.items()])

        sectors = pl.DataFrame({"ticker_id": ids, "sector": [f"sector_{x:02d}" for x in sector_of]})
        return SyntheticFrames(bars=bars, universe=self._universe(bars), sectors=sectors)

    def _universe(
        self, bars: pl.DataFrame, top_n: int = 1000, min_price: float = 5.0
    ) -> pl.DataFrame:
        """Point-in-time universe: ADV and price from prior sessions only."""
        daily = (
            bars.sort(["ticker_id", "session_date", "bar"])
            .group_by(["ticker_id", "session_date"], maintain_order=True)
            .agg(
                (pl.col("vwap") * pl.col("volume")).sum().alias("dollar_volume"),
                pl.col("raw_close").last().alias("raw_close"),
            )
        )
        return (
            daily.with_columns(
                pl.col("dollar_volume")
                .shift(1)
                .rolling_mean(20, min_samples=1)
                .over("ticker_id")
                .alias("adv20_usd"),
                pl.col("raw_close").shift(1).over("ticker_id").alias("prev_raw_close"),
            )
            .drop_nulls("adv20_usd")
            .with_columns(
                pl.col("adv20_usd")
                .rank("ordinal", descending=True)
                .over("session_date")
                .cast(pl.Int32)
                .alias("rank")
            )
            .with_columns(
                ((pl.col("rank") <= top_n) & (pl.col("prev_raw_close") > min_price)).alias(
                    "in_universe"
                )
            )
            .select(["session_date", "ticker_id", "adv20_usd", "rank", "in_universe"])
            .sort(["session_date", "rank"])
        )

    def build(self, out_dir: Path) -> DataManifest:
        """Write bars, universe, sectors and a manifest. Same seed, same bytes."""
        frames = self.frames()
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bars").mkdir(exist_ok=True)

        written: list[tuple[Path, int]] = []
        years = frames.bars["session_date"].dt.year()
        for year in sorted(years.unique().to_list()):
            part = frames.bars.filter(years == year)
            path = out_dir / "bars" / f"bars_{year}.parquet"
            part.write_parquet(path, compression="zstd", statistics=False)
            written.append((path, part.height))
        for name, df in (("universe", frames.universe), ("sectors", frames.sectors)):
            path = out_dir / f"{name}.parquet"
            df.write_parquet(path, compression="zstd", statistics=False)
            written.append((path, df.height))

        files = tuple(
            FileEntry(
                path=p.relative_to(out_dir).as_posix(),
                sha256=sha256_file(p),
                rows=rows,
                bytes=p.stat().st_size,
            )
            for p, rows in written
        )
        params_hash = sha256_canonical(self._params())
        dataset_hash = sha256_canonical(
            {"files": sorted(f.sha256 for f in files), "params_hash": params_hash}
        )
        manifest = DataManifest(
            dataset_id=f"syn_{dataset_hash[:12]}",
            dataset_hash=dataset_hash,
            kind="full",
            source="synthetic",
            start=self.sessions[0],
            end=self.sessions[-1],
            n_sessions=self.n_sessions,
            n_tickers=self.n_tickers,
            bar_minutes=30,
            adjustment="forward",
            params_hash=params_hash,
            code_commit=_git_commit(),
            created_at=BUILD_TIMESTAMP,
            files=files,
        )
        (out_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        (out_dir / "answers.json").write_text(self._answers.model_dump_json(indent=2))
        return manifest
