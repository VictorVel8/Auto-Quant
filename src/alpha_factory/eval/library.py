"""A frozen library as a scoring function: the pool's factors plus a combiner.

LibraryScorer is what final_eval scores (it implements SealedScorer) and what the freeze report
uses, so the number you see at freeze time and the sealed number come from one code path.
"""

from datetime import date

import numpy as np
import polars as pl

from alpha_factory.contracts import FrozenLibrary, Horizon, Record, SandboxSpec
from alpha_factory.data.marketdata import MarketData
from alpha_factory.dsl import CompiledFactor, compile_expr
from alpha_factory.eval.combine import (
    RidgeModel,
    rank_average,
    rank_features,
    ridge_fit,
    walk_forward_ridge,
)
from alpha_factory.eval.context import EvalContext, zone_range
from alpha_factory.eval.stats import daily_ic, intraday_ics, newey_west_tstat
from alpha_factory.hashing import sha256_canonical
from alpha_factory.sandbox.labels import compute_labels
from alpha_factory.sandbox.panel import load_panel

WARMUP = 45


def combiner_hash(lib: FrozenLibrary) -> str:
    """Identifies the combiner and the factors it combines (not the freeze time)."""
    return sha256_canonical(
        {
            "combiner": lib.combiner.model_dump(mode="json"),
            "factors": [f.model_dump(mode="json") for f in lib.factors],
        }
    )


def _compiled(lib: FrozenLibrary) -> dict[str, tuple[CompiledFactor, int]]:
    return {f.factor_id: (compile_expr(f.canonical_expr), f.sign) for f in lib.factors}


def factor_frames(lib: FrozenLibrary, panel: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        fid: cf.run(panel).select("tix", "ts", "session_date", "tsi", "score")
        for fid, (cf, _) in _compiled(lib).items()
    }


class LibraryScorer:
    """Scores a panel with the library's combiner. Ridge weights are fit on the search folds."""

    def __init__(self, lib: FrozenLibrary, sandbox: SandboxSpec, data: MarketData) -> None:
        if sandbox.content_hash() != lib.sandbox_hash:
            raise ValueError("the library was frozen under a different sandbox")
        self.lib, self.sandbox, self.data = lib, sandbox, data
        self.version = lib.lib_version
        self.horizon: Horizon = lib.combiner.target_horizon
        self._model: RidgeModel | None = None

    def combiner_hash(self) -> str:
        return combiner_hash(self.lib)

    def _train(self) -> RidgeModel:
        """Ridge on all search folds F1..Fk: the final model that scores holdout and sealed data."""
        if self._model is None:
            start, end = zone_range(self.sandbox, "search")
            panel = load_panel(self.data, start, end, warmup_sessions=WARMUP)
            keep = pl.col("session_date").is_between(start, end)
            feats = rank_features(factor_frames(self.lib, panel.df)).filter(keep)
            labels = compute_labels(panel, self.sandbox)
            self._model = ridge_fit(
                feats, labels, [f.factor_id for f in self.lib.factors],
                f"d_{self.horizon}", self.lib.combiner.ridge_alpha,
            )  # fmt: skip
        return self._model

    def score(self, panel: pl.DataFrame) -> pl.DataFrame:
        """tix, ts, session_date, score: the combined signal on every row of the panel."""
        frames = factor_frames(self.lib, panel)
        if self.lib.combiner.kind == "rank_average":
            signs = {f.factor_id: f.sign for f in self.lib.factors}
            out = rank_average(frames, signs)
        else:
            out = self._train().predict(rank_features(frames))
        return out.select("tix", "ts", "session_date", "score")


class LibraryReport(Record):
    """Numbers for you only. They are never put in a context packet."""

    lib_version: str
    combiner: str
    n_factors: int
    search_ic: float
    search_tstat: float
    search_ic_by_year: dict[int, float]
    search_net_sharpe: float | None
    holdout_ic: float
    holdout_tstat: float
    holdout_net_sharpe: float | None
    holdout_net_sharpe_by_year: dict[int, float]


def _ic(scores: pl.DataFrame, ctx: EvalContext, h: Horizon, lo: date, hi: date) -> pl.DataFrame:
    s = scores.filter(pl.col("session_date").is_between(lo, hi))
    return daily_ic(intraday_ics(s, ctx.labels, f"y_{h}", ctx.min_names), 1)


def _sharpe(scores: pl.DataFrame, ctx: EvalContext):  # type: ignore[no-untyped-def]
    from alpha_factory.backtest.engine import simulate

    sim = ctx.sim_inputs()
    ids = ctx.panel.ticker_ids()
    s = scores.filter(pl.col("session_date").is_between(ctx.start, ctx.end))
    return simulate(
        s.join(ids, on="tix").select("ticker_id", "ts", "score"), sim.bars, sim.members,
        sim.sectors, ctx.sandbox,
    )  # fmt: skip


def combined_report(
    lib: FrozenLibrary,
    sandbox: SandboxSpec,
    search: EvalContext,
    holdout: EvalContext,
    data: MarketData,
) -> LibraryReport:  # fmt: skip
    """The combined signal on the search folds (out of sample for ridge) and on the holdout."""
    scorer = LibraryScorer(lib, sandbox, data)
    h = scorer.horizon
    if lib.combiner.kind == "rank_average":
        s_scores = scorer.score(search.panel.df)
    else:
        frames = factor_frames(lib, search.panel.df)
        feats = rank_features(frames)
        sessions = sorted(search.panel.df["session_date"].unique().to_list())
        s_scores = walk_forward_ridge(
            feats, search.labels, [f.factor_id for f in lib.factors], f"d_{h}",
            [f.val for f in sandbox.folds], sessions, lib.combiner.ridge_alpha,
        ).join(feats.select("tix", "ts", "session_date"), on=["tix", "ts"], how="left")  # fmt: skip
    s_scores = s_scores.filter(pl.col("score").is_not_null())
    s_daily = _ic(s_scores, search, h, search.start, search.end)
    s_x = s_daily["ic"].to_numpy()
    s_back = _sharpe(s_scores, search) if s_scores.height else None

    h_scores = scorer.score(holdout.panel.df).filter(pl.col("score").is_not_null())
    h_daily = _ic(h_scores, holdout, h, holdout.start, holdout.end)
    h_x = h_daily["ic"].to_numpy()
    h_back = _sharpe(h_scores, holdout) if h_scores.height else None
    yearly = s_daily.group_by(pl.col("session_date").dt.year().alias("y")).agg(pl.col("ic").mean())
    by_year = {int(y): float(v) for y, v in yearly.iter_rows()}  # fmt: skip
    return LibraryReport(
        lib_version=lib.lib_version, combiner=lib.combiner.kind, n_factors=len(lib.factors),
        search_ic=float(np.mean(s_x)) if len(s_x) else 0.0, search_tstat=newey_west_tstat(s_x),
        search_ic_by_year=by_year,
        search_net_sharpe=s_back.summary.net_sharpe if s_back else None,
        holdout_ic=float(np.mean(h_x)) if len(h_x) else 0.0, holdout_tstat=newey_west_tstat(h_x),
        holdout_net_sharpe=h_back.summary.net_sharpe if h_back else None,
        holdout_net_sharpe_by_year=h_back.net_sharpe_by_year if h_back else {},
    )  # fmt: skip
