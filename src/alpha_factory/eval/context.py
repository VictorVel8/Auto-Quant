"""Everything an evaluation needs that does not depend on the factor, built once per batch."""

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import polars as pl

from alpha_factory.contracts.sandbox import SandboxSpec
from alpha_factory.data.marketdata import MarketData
from alpha_factory.eval.baselines import baseline_scores
from alpha_factory.eval.regimes import regime_table
from alpha_factory.sandbox.labels import compute_labels
from alpha_factory.sandbox.panel import Panel, load_panel

WARMUP_SESSIONS = 45  # history before the first scored session: windows and the regime lookback


def min_names(sandbox: SandboxSpec) -> int:
    """IC is skipped below this many names: 200 for a 1,000-name universe."""
    return min(200, sandbox.universe.top_n // 5)


def min_condition_names(sandbox: SandboxSpec) -> int:
    return min(100, sandbox.universe.top_n // 10)


@dataclass
class SimInputs:
    bars: pl.DataFrame
    members: pl.DataFrame
    sectors: pl.DataFrame


@dataclass
class EvalContext:
    sandbox: SandboxSpec
    dataset_hash: str
    panel: Panel
    labels: pl.DataFrame
    baselines: pl.DataFrame
    regimes: pl.DataFrame
    start: date  # first scored session
    end: date
    zone: str
    _sim: SimInputs | None = field(default=None, repr=False)

    @property
    def min_names(self) -> int:
        return min_names(self.sandbox)

    def sim_inputs(self) -> SimInputs:
        """Prices, tradability and sectors for the scored sessions, in backtester form."""
        if self._sim is None:
            ids = self.panel.ticker_ids()
            df = self.panel.df.filter(pl.col("session_date").is_between(self.start, self.end))
            bars = (
                df.drop_nulls("close")
                .join(ids, on="tix")
                .select("session_date", "bar", "ticker_id", "open", "raw_close", "adj_factor")
            )
            members = (
                df.filter(pl.col("is_member"))
                .join(ids, on="tix")
                .select("session_date", "ticker_id", "tradable")
                .unique()
            )
            sectors = (
                df.select("tix", pl.col("sector").cast(pl.String))
                .unique("tix")
                .join(ids, on="tix")
                .select("ticker_id", "sector")
            )
            self._sim = SimInputs(bars, members, sectors)
        return self._sim


def zone_range(sandbox: SandboxSpec, zone: Literal["search", "holdout"]) -> tuple[date, date]:
    if zone == "holdout":
        return sandbox.gate_holdout.start, sandbox.gate_holdout.end
    return min(f.val.start for f in sandbox.folds), max(f.val.end for f in sandbox.folds)


def build_context(
    data: MarketData, sandbox: SandboxSpec, zone: Literal["search", "holdout"] = "search"
) -> EvalContext:
    """Load the panel for a zone and precompute labels, baselines and regimes.

    The search zone ends where the gate holdout begins, so searching can never read holdout rows.
    """
    if data.kind != "dev":
        raise PermissionError("evaluation contexts are built on the dev dataset")
    if data.manifest.dataset_hash != sandbox.dataset_hash:
        raise ValueError("the sandbox was written for a different dataset build")
    start, end = zone_range(sandbox, zone)
    panel = load_panel(data, start, end, warmup_sessions=WARMUP_SESSIONS)
    keep = pl.col("session_date").is_between(start, end)
    labels = compute_labels(panel, sandbox)
    labels = labels.join(panel.df.select("tix", "ts", "session_date"), on=["tix", "ts"]).filter(
        keep
    )
    baselines = baseline_scores(panel.df)
    return EvalContext(
        sandbox=sandbox,
        dataset_hash=data.manifest.dataset_hash,
        panel=panel,
        labels=labels,
        baselines=baselines,
        regimes=regime_table(panel.df),
        start=start,
        end=end,
        zone=zone,
    )
