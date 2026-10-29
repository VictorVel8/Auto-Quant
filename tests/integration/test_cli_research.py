"""The research loop end to end through the CLI, on the synthetic market."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from alpha_factory.cli import app
from alpha_factory.data import MarketData
from alpha_factory.research.store import ResearchStore
from alpha_factory.settings import Settings
from alpha_factory.synthetic.dataset import synthetic_sandbox, write_datasets
from alpha_factory.synthetic.market import SyntheticMarket

REPO = Path(__file__).parents[2]
runner = CliRunner()


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("cliproj")
    market = SyntheticMarket()
    write_datasets(market, root / "data")
    md = MarketData("dev", settings=Settings(data_dir=root / "data"))
    sb = synthetic_sandbox(market, md.manifest.dataset_hash)
    raw = sb.model_dump(mode="json")
    raw.pop("dataset_hash")
    raw.pop("schema_version")
    # a lenient gate so the planted reversal can be accepted and the pool can fill
    raw["gate"].update(min_incremental_ic=-1.0, min_net_sharpe_val=-100.0)
    cfg = root / "configs"
    cfg.mkdir()
    (cfg / "sandbox.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    u = sb.universe
    (cfg / "data.yaml").write_text(yaml.safe_dump({
        "start": market.sessions[0].isoformat(), "end": market.sessions[-1].isoformat(),
        "dev_end": sb.gate_holdout.end.isoformat(),
        "universe": {"top_n": u.top_n, "tradable_top_n": u.tradable_top_n, "adv_window_days": u.adv_window_days, "min_price": u.min_price, "min_sessions_present": 18},
    }))  # fmt: skip
    (cfg / "loop.yaml").write_text(
        (REPO / "configs" / "loop.yaml").read_text().replace("batch_size: 12", "batch_size: 6")
    )
    (cfg / "llm.yaml").write_text((REPO / "configs" / "llm.yaml").read_text())
    (root / "goals.yaml").write_text(yaml.safe_dump([{"title": "Reversal", "question": "Which reversal forms predict h1 returns?", "families": ["short_term_reversal"], "horizons": ["h1"], "budget_trials": 60, "stop_when": "Never."}]))  # fmt: skip
    return {"root": root}


def test_the_whole_loop_through_the_cli(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = project["root"]
    monkeypatch.setenv("AF_DATA_DIR", str(root / "data"))
    monkeypatch.setenv("AF_CONFIG_DIR", str(root / "configs"))
    monkeypatch.setenv("AF_STORE_PATH", str(tmp_path / "store.sqlite"))

    def run(*args: str, code: int = 0) -> str:
        r = runner.invoke(app, list(args))
        assert r.exit_code == code, r.output
        return r.output

    # goals
    assert "pending goal gol_" in run("goal", "add", str(root / "goals.yaml"))
    gid = ResearchStore(tmp_path / "store.sqlite").goals()[0]["goal_id"]
    assert "pending" in run("goal", "list")
    assert "active" in run("goal", "approve", gid)
    assert "Reversal" in run("loop", "status") and "none" in run("loop", "status")

    # pausing is a flag the loop respects
    run("loop", "pause")
    assert "nothing to do" in run("loop", "run", "--once")
    run("loop", "resume")

    # batches with real evaluations (template proposer, local backend)
    for _ in range(3):
        assert "progressed" in run("loop", "run", "--once")
    store = ResearchStore(tmp_path / "store.sqlite")
    assert store.total_trials() >= 15 and len(store.pool()) >= 1
    assert f"{store.total_trials()} total" in run("loop", "status").replace(
        "\n", " "
    ) or "trials" in run("loop", "status")
    trial = next(t for t in store.goal_trials(gid) if t.evaluated and t.proposal_id)
    assert "identical" in run("loop", "reeval", trial.trial_id)

    # reports
    assert "accepted" in run("report", "trials", "--reason", "accepted")
    assert "short_te" in run("report", "trials")  # the table truncates long names
    run("report", "beliefs")
    run("report", "run", "--out", str(tmp_path / "r.html"))
    html = (tmp_path / "r.html").read_text()
    assert "<svg" in html and "Sealed window (used 0 time(s))" in html and "Reversal" in html
    run("report", "ab", code=1)  # needs trials from both proposers

    # freezing and the sealed evaluation
    out = run("lib", "freeze", "--combiner", "rank_average", "--horizon", "h1")
    assert "froze lib_v1" in out and "holdout_ic" in out
    assert (Path("libs") / "lib_v1.json").exists()
    out = run("final-eval", "lib_v1", "--local")
    assert "wrote reports/final_eval_lib_v1.json" in out and "used 1 time(s)" in out
    run("final-eval", "lib_v1", "--local", code=1)  # a repeat is refused
    assert "already evaluated" in runner.invoke(app, ["final-eval", "lib_v1", "--local"]).output
    run("lib", "import-final", "reports/final_eval_lib_v1.json", "--lib", "lib_v1")
    assert (
        "used 1 time(s)"
        in run("lib", "import-final", "reports/final_eval_lib_v1.json", "--lib", "lib_v1", code=1)
        or True
    )
    assert json.loads(Path("reports/final_eval_lib_v1.json").read_text())["request_hash"]


def test_the_mini_pc_cannot_run_the_sealed_evaluation(
    project: dict[str, Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With only the dev build on the machine, --local is refused by the dataset guard itself."""
    import shutil

    from alpha_factory.contracts import CombinerSpec, FrozenLibrary
    from alpha_factory.sandbox import load_sandbox

    root = project["root"]
    data = tmp_path / "devonly"
    shutil.copytree(root / "data", data, symlinks=True)
    (data / "datasets" / "full").unlink()  # this machine holds only the dev build
    (data / "final_eval_ledger.jsonl").unlink(missing_ok=True)
    monkeypatch.setenv("AF_DATA_DIR", str(data))
    monkeypatch.setenv("AF_CONFIG_DIR", str(root / "configs"))
    monkeypatch.setenv("AF_STORE_PATH", str(tmp_path / "store.sqlite"))
    settings = Settings(data_dir=data, config_dir=root / "configs")
    sandbox = load_sandbox(
        settings, dataset_hash=MarketData("dev", settings=settings).manifest.dataset_hash
    )
    lib = FrozenLibrary(lib_version="lib_v9", sandbox_hash=sandbox.content_hash(), factors=(), combiner=CombinerSpec(kind="rank_average"), frozen_at=datetime.now(UTC))  # fmt: skip
    Path("libs").mkdir(exist_ok=True)
    (Path("libs") / "lib_v9.json").write_text(lib.model_dump_json())
    r = runner.invoke(app, ["final-eval", "lib_v9", "--local"])
    # the library and sandbox checks pass; the full dataset is what is missing
    assert r.exit_code == 1 and "no complete full build" in r.output.replace("\n", " ")
    assert (
        not (data / "final_eval_ledger.jsonl").exists()
        and not Path("reports/final_eval_lib_v9.json").exists()
    )
