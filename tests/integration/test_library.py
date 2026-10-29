"""Frozen libraries: the combined signal, its freeze report, and the sealed evaluation through it."""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest

from alpha_factory.contracts import (
    CombinerSpec,
    FinalEvalRequest,
    FrozenLibrary,
    LibFactor,
    SandboxSpec,
)
from alpha_factory.data import MarketData
from alpha_factory.eval.context import EvalContext, build_context
from alpha_factory.eval.library import LibraryScorer, combined_report, combiner_hash
from alpha_factory.sandbox.final_eval import FinalEvalLedger, final_eval
from alpha_factory.settings import Settings
from alpha_factory.synthetic.dataset import synthetic_sandbox, write_datasets
from alpha_factory.synthetic.market import SyntheticMarket

PLANTED = "Neg(Pct($close, 2))"
NOISE = "Div($volume, Mean($volume, 26))"


@pytest.fixture(scope="module")
def world(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Path, MarketData, SandboxSpec, EvalContext, EvalContext]]:
    market = SyntheticMarket()
    root = tmp_path_factory.mktemp("lib")
    write_datasets(market, root)
    md = MarketData("dev", settings=Settings(data_dir=root))
    sb = synthetic_sandbox(market, md.manifest.dataset_hash)
    yield root, md, sb, build_context(md, sb), build_context(md, sb, zone="holdout")


def library(sb: SandboxSpec, kind: str, *exprs: str) -> FrozenLibrary:
    return FrozenLibrary(
        lib_version="lib_v1", sandbox_hash=sb.content_hash(),
        factors=tuple(LibFactor(factor_id=f"fac_{i}", canonical_expr=e, sign=1, horizon="h1") for i, e in enumerate(exprs)),
        combiner=CombinerSpec(kind=kind, target_horizon="h1"),  # type: ignore[arg-type]
        frozen_at=datetime(2026, 1, 1, tzinfo=UTC),
    )  # fmt: skip


def test_the_combiner_hash_tracks_the_factors_and_the_combiner_not_the_freeze_time(world) -> None:  # type: ignore[no-untyped-def]
    sb = world[2]
    a = library(sb, "rank_average", PLANTED, NOISE)
    assert combiner_hash(a) == combiner_hash(
        a.model_copy(update={"frozen_at": datetime(2030, 1, 1, tzinfo=UTC)})
    )
    assert combiner_hash(a) != combiner_hash(library(sb, "rank_average", PLANTED))
    assert combiner_hash(a) != combiner_hash(library(sb, "ridge", PLANTED, NOISE))


def test_a_scorer_refuses_a_library_frozen_under_another_sandbox(world) -> None:  # type: ignore[no-untyped-def]
    _, md, sb, _, _ = world
    stale = library(sb, "rank_average", PLANTED).model_copy(update={"sandbox_hash": "f" * 64})
    with pytest.raises(ValueError, match="different sandbox"):
        LibraryScorer(stale, sb, md)


def test_rank_average_report_finds_the_signal_in_search_and_holdout(world) -> None:  # type: ignore[no-untyped-def]
    _, md, sb, search, holdout = world
    lib = library(sb, "rank_average", PLANTED, NOISE)
    r = combined_report(lib, sb, search, holdout, md)
    assert r.n_factors == 2 and r.combiner == "rank_average"
    assert (
        r.search_ic > 0.012 and r.search_tstat > 5 and r.holdout_ic > 0.01
    )  # diluted by the noise factor
    assert (
        set(r.search_ic_by_year)
        and r.search_net_sharpe is not None
        and r.holdout_net_sharpe is not None
    )
    assert LibraryReportRoundTrip.ok(r)


class LibraryReportRoundTrip:
    @staticmethod
    def ok(r) -> bool:  # type: ignore[no-untyped-def]
        return type(r).model_validate_json(r.model_dump_json()) == r


def test_ridge_learns_to_weight_the_real_factor_and_works_out_of_sample(world) -> None:  # type: ignore[no-untyped-def]
    _, md, sb, search, holdout = world
    lib = library(sb, "ridge", PLANTED, NOISE)
    scorer = LibraryScorer(lib, sb, md)
    w = scorer._train().weights  # pyright: ignore[reportPrivateUsage]
    assert abs(w[0]) > 5 * abs(w[1])  # the reversal factor dominates the noise factor
    r = combined_report(lib, sb, search, holdout, md)
    assert r.search_ic > 0.01 and r.holdout_ic > 0.01  # walk-forward scores exist for folds 2 to 4


def test_final_eval_scores_a_library_once_through_the_same_code(world, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    root, _, sb, _, _ = world
    lib = library(sb, "rank_average", PLANTED, NOISE)
    full = MarketData("full", settings=Settings(data_dir=root), sealed=True)
    scorer = LibraryScorer(lib, sb, full)
    request = FinalEvalRequest(
        lib_version=lib.lib_version,
        combiner_hash=scorer.combiner_hash(),
        sandbox_hash=sb.content_hash(),
    )
    ledger = FinalEvalLedger(tmp_path / "ledger.jsonl")
    report = final_eval(request, library=scorer, ledger=ledger, sandbox=sb, data=full)
    assert report.ic_mean > 0.01 and ledger.count() == 1
    with pytest.raises(Exception, match="already evaluated"):
        final_eval(request, library=scorer, ledger=ledger, sandbox=sb, data=full)
