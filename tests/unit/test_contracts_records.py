from datetime import UTC, date, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from alpha_factory.contracts import (
    Belief,
    BeliefStatus,
    DataManifest,
    DateRange,
    FinalEvalRequest,
    Fold,
    GateSpec,
    LabelSpec,
    PortfolioSpec,
    QACheck,
    QAReport,
    SandboxSpec,
    UniverseSpec,
)
from alpha_factory.contracts.sandbox import CostSpec

NOW = datetime(2026, 1, 1, tzinfo=UTC)
H = "a" * 64


def belief(support: int = 0, contradict: int = 0, retired: bool = False) -> Belief:
    return Belief(
        belief_id="blf_1",
        statement="s",
        family="fam",
        horizon="h1",
        n_support=support,
        n_contradict=contradict,
        retired=retired,
        created_at=NOW,
        updated_at=NOW,
    )


@pytest.mark.parametrize(
    ("support", "contradict", "retired", "mean", "status"),
    [
        (0, 0, False, 0.5, BeliefStatus.UNTESTED),
        (2, 0, False, 0.75, BeliefStatus.UNTESTED),  # fewer than 3 observations
        (4, 1, False, 5 / 7, BeliefStatus.SUPPORTED),
        (1, 4, False, 2 / 7, BeliefStatus.REFUTED),
        (2, 2, False, 0.5, BeliefStatus.CONTESTED),
        (9, 0, True, 10 / 11, BeliefStatus.RETIRED),
    ],
)
def test_belief_golden_numbers(
    support: int, contradict: int, retired: bool, mean: float, status: BeliefStatus
) -> None:
    b = belief(support, contradict, retired)
    assert b.posterior_mean == pytest.approx(mean)
    assert b.status == status


def test_record_roundtrips_with_computed_fields() -> None:
    b = belief(4, 1)
    dumped = b.model_dump(mode="json")
    assert "status" in dumped
    assert Belief.model_validate(dumped) == b
    assert Belief.model_validate_json(b.model_dump_json()) == b


def test_record_forbids_extras_and_is_frozen() -> None:
    with pytest.raises(ValidationError):
        GateSpec.model_validate({"nope": 1})
    with pytest.raises(ValidationError):
        GateSpec().min_ic_tstat = 1.0  # type: ignore[misc]
    assert GateSpec().schema_version == 1


def test_qa_report_passed_is_computed() -> None:
    ok = QACheck(name="a", passed=True, value=0, threshold=1, detail="")
    bad = QACheck(name="b", passed=False, value=2, threshold=1, detail="")
    assert QAReport(dataset_hash=H, checks=(ok,)).passed
    assert not QAReport(dataset_hash=H, checks=(ok, bad)).passed


def sandbox(**over: Any) -> SandboxSpec:
    base: dict[str, Any] = {
        "sandbox_version": "1",
        "dataset_hash": H,
        "universe": UniverseSpec(),
        "labels": (
            LabelSpec(name="h1", hours=1),
            LabelSpec(name="h4", hours=4),
        ),
        "folds": (
            Fold(
                fold_id="f1",
                train=None,
                val=DateRange(start=date(2022, 1, 3), end=date(2022, 6, 1)),
            ),
        ),
        "gate_holdout": DateRange(start=date(2022, 7, 1), end=date(2022, 9, 1)),
        "sealed_test": DateRange(start=date(2022, 10, 1), end=date(2022, 12, 31)),
        "costs": CostSpec(sell_fee_bps=0.2),
        "portfolio": PortfolioSpec(),
        "gate": GateSpec(),
    }
    return SandboxSpec(**{**base, **over})


def test_sandbox_valid_and_hash_is_stable() -> None:
    a, b = sandbox(), sandbox()
    assert a.content_hash() == b.content_hash()
    assert len(a.content_hash()) == 64
    assert sandbox(embargo_days=6).content_hash() != a.content_hash()


def test_sandbox_rejects_fold_overlapping_gate_holdout() -> None:
    late = Fold(
        fold_id="late", train=None, val=DateRange(start=date(2022, 6, 1), end=date(2022, 7, 1))
    )
    with pytest.raises(ValidationError, match="overlaps the gate holdout"):
        sandbox(folds=(late,))


def test_sandbox_rejects_holdout_overlapping_sealed_test() -> None:
    with pytest.raises(ValidationError, match="before the sealed test"):
        sandbox(gate_holdout=DateRange(start=date(2022, 7, 1), end=date(2022, 10, 1)))


def test_sandbox_purge_must_cover_longest_label() -> None:
    with pytest.raises(ValidationError, match="purge_bars"):
        sandbox(purge_bars=7)  # h4 = 8 bars
    sandbox(purge_bars=8)


def test_date_range_must_be_ordered() -> None:
    with pytest.raises(ValidationError):
        DateRange(start=date(2022, 1, 2), end=date(2022, 1, 1))


def test_final_eval_request_hash() -> None:
    r = FinalEvalRequest(lib_version="v1", combiner_hash=H, sandbox_hash=H)
    assert (
        r.request_hash()
        == FinalEvalRequest(lib_version="v1", combiner_hash=H, sandbox_hash=H).request_hash()
    )
    assert (
        r.request_hash()
        != FinalEvalRequest(lib_version="v2", combiner_hash=H, sandbox_hash=H).request_hash()
    )


def test_manifest_rejects_bad_hash_and_adjustment() -> None:
    good: dict[str, Any] = {
        "dataset_id": "d",
        "dataset_hash": H,
        "kind": "dev",
        "source": "synthetic",
        "start": date(2022, 1, 3),
        "end": date(2022, 2, 1),
        "n_sessions": 20,
        "n_tickers": 5,
        "bar_minutes": 30,
        "adjustment": "forward",
        "params_hash": H,
        "code_commit": "abc",
        "created_at": NOW,
        "files": (),
    }
    assert DataManifest.model_validate(good)
    with pytest.raises(ValidationError):
        DataManifest.model_validate({**good, "dataset_hash": "xyz"})
    with pytest.raises(ValidationError):
        DataManifest.model_validate({**good, "adjustment": "backward"})
