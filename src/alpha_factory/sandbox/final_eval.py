"""The sealed final evaluation: the only code that reads the sealed window.

Sealed by where it can run, not by discipline: only the MacBook and a sealed Modal volume hold the
full dataset. Guards run in order before any data loads; the data guard is MarketData itself.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import polars as pl

from alpha_factory.backtest.engine import simulate
from alpha_factory.contracts.base import Horizon
from alpha_factory.contracts.sandbox import FinalEvalReport, FinalEvalRequest, SandboxSpec
from alpha_factory.data.marketdata import MarketData
from alpha_factory.eval.stats import daily_ic, intraday_ics, newey_west_tstat
from alpha_factory.sandbox.labels import compute_labels
from alpha_factory.sandbox.panel import load_panel


class FinalEvalError(RuntimeError):
    """A guard refused the request."""


class SealedScorer(Protocol):
    """What final_eval needs from a frozen library and its combiner (Plan 4 provides it)."""

    @property
    def version(self) -> str: ...

    @property
    def horizon(self) -> Horizon: ...

    def combiner_hash(self) -> str: ...

    def score(self, panel: pl.DataFrame) -> pl.DataFrame:
        """tix, ts, session_date, score: the combined signal over the panel."""
        ...


class FinalEvalLedger:
    """Append-only JSON-lines file of every final evaluation ever run."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def entries(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def has(self, request_hash: str) -> bool:
        return any(e["request_hash"] == request_hash for e in self.entries())

    def count(self) -> int:
        """How many times the sealed window has been used."""
        return len(self.entries())

    def append(self, report: FinalEvalReport) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(report.model_dump_json() + "\n")


def final_eval(
    request: FinalEvalRequest,
    *,
    library: SealedScorer,
    ledger: FinalEvalLedger,
    sandbox: SandboxSpec,
    data: MarketData | None = None,
) -> FinalEvalReport:
    """Score a frozen library once on the sealed window.

    Guards, in order: (1) library and combiner match the request, (2) the request's sandbox is the
    current one, (3) the request is not already in the ledger, (4) the full dataset opens with
    sealed=True (and so only exists where it is allowed to).
    """
    if library.version != request.lib_version or library.combiner_hash() != request.combiner_hash:
        raise FinalEvalError("the library or combiner does not match the request")
    if request.sandbox_hash != sandbox.content_hash():
        raise FinalEvalError("the request was made for a different sandbox")
    request_hash = request.request_hash()
    if ledger.has(request_hash):
        raise FinalEvalError("this request was already evaluated; the sealed window is used once")
    data = data or MarketData("full", sealed=True)
    if data.kind != "full":
        raise FinalEvalError("final evaluation needs the full dataset")

    window = sandbox.sealed_test
    panel = load_panel(data, window.start, window.end)
    keep = pl.col("session_date").is_between(window.start, window.end)
    labels = compute_labels(panel, sandbox)
    labels = labels.join(panel.df.select("tix", "ts", "session_date"), on=["tix", "ts"]).filter(
        keep
    )
    scores = library.score(panel.df).filter(keep).drop_nulls("score")

    min_names = min(200, sandbox.universe.top_n // 5)
    ics = intraday_ics(scores, labels, f"y_{library.horizon}", min_names)
    daily = daily_ic(ics, 1)
    x = daily["ic"].to_numpy()
    ic_mean = float(x.mean()) if len(x) else 0.0

    ids = panel.ticker_ids()
    df = panel.df.filter(keep)
    back = simulate(
        scores.join(ids, on="tix").select("ticker_id", "ts", "score"),
        df.drop_nulls("close")
        .join(ids, on="tix")
        .select("session_date", "bar", "ticker_id", "open", "raw_close", "adj_factor"),
        df.filter(pl.col("is_member"))
        .join(ids, on="tix")
        .select("session_date", "ticker_id", "tradable")
        .unique(),
        df.select("tix", pl.col("sector").cast(pl.String))
        .unique("tix")
        .join(ids, on="tix")
        .select("ticker_id", "sector"),
        sandbox,
    )
    gross = float(back.daily["gross_pnl"].sum()) if back.daily.height else 0.0
    costs = float(back.daily["costs"].sum()) if back.daily.height else 0.0
    report = FinalEvalReport(
        request_hash=request_hash,
        ic_mean=ic_mean,
        ic_tstat=newey_west_tstat(x),
        net_sharpe=back.summary.net_sharpe,
        net_sharpe_by_year=back.net_sharpe_by_year,
        turnover=float(back.daily["turnover"].mean()) if back.daily.height else 0.0,  # type: ignore[arg-type]
        cost_share_of_gross=costs / gross if gross > 0 else 1.0,  # no gross profit: all cost
        created_at=datetime.now(UTC),
    )
    ledger.append(report)
    return report
