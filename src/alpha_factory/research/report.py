"""Reports from the store: a static HTML file, and the template-versus-LLM comparison.

Reports never include sealed-window numbers except in their own section, which lists the final
evaluations and how many times the sealed window has been used.
"""

import html
import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from alpha_factory.contracts import FinalEvalReport, LibFactor, ReasonCode, ResearchGoal
from alpha_factory.research.store import ResearchStore


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class ABRow:
    proposer: str
    evaluated: int  # trials counted (the equal budget)
    accepted: int
    rate: float
    ci: tuple[float, float]
    best_combined_ic: float | None  # combined validation IC of its accepted factors
    spend_usd: float
    spend_per_acceptance: float | None


def ab_comparison(
    store: ResearchStore, combined_ic: Callable[[list[LibFactor]], float | None] | None = None
) -> list[ABRow]:
    """Template versus LLM at an equal trial budget: each proposer's first N evaluated trials,
    where N is the smaller of the two counts. Spend is the LLM's total (proposer, reflector, repair)."""
    trials = {p: store.proposer_trials(p) for p in ("template", "llm")}
    n = min(len(t) for t in trials.values())
    spend = sum(store.llm_spend_by_role().values())
    rows = []
    for proposer, ts in trials.items():
        used = ts[:n]
        accepted = [t.trial_id for t in used if t.reason == ReasonCode.ACCEPTED]
        k = len(accepted)
        cost = spend if proposer == "llm" else 0.0
        factors = store.lib_factors_for_trials(accepted)
        rows.append(
            ABRow(
                proposer,
                n,
                k,
                k / n if n else 0.0,
                wilson(k, n),
                combined_ic(factors) if combined_ic and factors else None,
                cost,
                cost / k if k else None,
            )
        )
    return rows


def _svg(fig: Figure) -> str:
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _bar(counts: dict[str, int], title: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    keys = sorted(counts, key=lambda k: -counts[k])
    ax.barh(keys[::-1], [counts[k] for k in keys[::-1]], color="#3b6ea8")
    ax.set_title(title, fontsize=10)
    return _svg(fig)


def _acceptance_curve(store: ResearchStore) -> str:
    fig, ax = plt.subplots(figsize=(6, 2.8))
    rows = store._q(
        "SELECT family, reason FROM trials WHERE evaluated=1 AND family IS NOT NULL ORDER BY rowid"
    )  # pyright: ignore[reportPrivateUsage]
    cum: dict[str, list[int]] = {}
    for i, r in enumerate(rows):
        cum.setdefault(r["family"], [])
        cum[r["family"]].append(i)
    for fam in sorted(cum):
        xs = [i for i, r in enumerate(rows) if r["family"] == fam and r["reason"] == "accepted"]
        ax.step(xs or [0], list(range(1, len(xs) + 1)) or [0], where="post", label=fam)
    ax.set_xlabel("evaluated trial")
    ax.set_ylabel("accepted")
    ax.set_title("Acceptances by family over time", fontsize=10)
    if cum:
        ax.legend(fontsize=7)
    return _svg(fig)


def _table(headers: list[str], rows: Sequence[Sequence[object]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    return f"<table><tr>{head}</tr>{body}</table>"


def render_html(store: ResearchStore, ab: list[ABRow] | None = None) -> str:
    goals = []
    for g in store.goals():
        goal = ResearchGoal.model_validate_json(g["goal_json"])
        goals.append(
            [
                g["status"],
                goal.title,
                f"{g['trials_used']}/{goal.budget_trials}",
                store.goal_acceptances(g["goal_id"]),
                g["outcome"] or "",
            ]
        )
    pool = [
        [p.factor_id, p.family, p.horizon, p.canonical_expr, f"{p.ic_tstat:.1f}"]
        for p in store.pool()
    ]
    beliefs = [
        [b.status.value, b.family, b.horizon, b.statement, f"{b.n_support}/{b.n_contradict}"]
        for b in store.beliefs()
    ]
    spend = store.llm_spend_by_role()
    sections = [
        "<h2>Goal progress</h2>"
        + _table(["status", "goal", "trials", "accepted", "outcome"], goals),
        "<h2>Reasons</h2>" + _bar(store.reason_counts(), "Trials by reason"),
        "<h2>Acceptance by family</h2>" + _acceptance_curve(store),
        "<h2>Pool</h2>"
        + _table(["factor", "family (cluster)", "horizon", "expression", "t"], pool),
        "<h2>Beliefs</h2>"
        + _table(["status", "family", "horizon", "statement", "for/against"], beliefs),
        "<h2>LLM spend</h2>"
        + _table(["role", "USD"], [[k, f"{v:.2f}"] for k, v in sorted(spend.items())]),
    ]
    if ab:
        sections.append(
            "<h2>Template versus LLM</h2>"
            + _table(
                [
                    "proposer",
                    "trials",
                    "accepted",
                    "rate (95% CI)",
                    "best combined IC",
                    "USD per acceptance",
                ],
                [
                    [
                        r.proposer,
                        r.evaluated,
                        r.accepted,
                        f"{r.rate:.1%} ({r.ci[0]:.1%} to {r.ci[1]:.1%})",
                        "n/a" if r.best_combined_ic is None else f"{r.best_combined_ic:.4f}",
                        "n/a"
                        if r.spend_per_acceptance is None
                        else f"{r.spend_per_acceptance:.2f}",
                    ]
                    for r in ab
                ],
            )
        )
    finals = store.final_evals()
    rows = []
    for f in finals:
        rep = FinalEvalReport.model_validate_json(f["report_json"])
        rows.append(
            [
                f["lib_version"],
                f"{rep.ic_mean:.4f}",
                f"{rep.ic_tstat:.2f}",
                f"{rep.net_sharpe:.2f}",
                f["imported_at"][:10],
            ]
        )
    sections.append(
        f"<h2>Sealed window (used {len(finals)} time(s))</h2>"
        "<p>Each use weakens the sealed window: treat the first result as the honest one.</p>"
        + _table(["library", "IC", "t", "net Sharpe", "imported"], rows)
    )
    style = "body{font-family:sans-serif;max-width:900px;margin:2em auto}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:2px 6px;font-size:12px}"
    return (
        f"<!doctype html><meta charset=utf-8><title>Alpha Factory report</title><style>{style}</style><h1>Alpha Factory</h1>"
        + "".join(sections)
    )
