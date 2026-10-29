from pathlib import Path

import pytest
from typer.testing import CliRunner

from alpha_factory.cli import app

runner = CliRunner()
REPO_CONFIGS = Path(__file__).parents[2] / "configs"

GROUPS = [
    "config",
    "synth",
    "data",
    "eval",
    "loop",
    "lib",
    "family",
    "goal",
    "report",
    "final-eval",
]


def test_help_lists_every_group() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for g in GROUPS:
        assert g in result.output


def test_the_research_commands_work_on_an_empty_store() -> None:
    """These were stubs before Plan 4; on an empty store they run and say what is missing."""
    assert runner.invoke(app, ["goal", "list"]).exit_code == 0
    assert runner.invoke(app, ["loop", "status"]).exit_code == 0
    assert runner.invoke(app, ["report", "trials"]).exit_code == 0
    assert runner.invoke(app, ["report", "beliefs"]).exit_code == 0
    assert runner.invoke(app, ["family", "activate", "momentum"]).exit_code == 0
    assert runner.invoke(app, ["loop", "pause"]).exit_code == 0
    assert runner.invoke(app, ["loop", "resume"]).exit_code == 0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["lib", "freeze"], "pool is empty"),
        (["report", "ab"], "both proposers"),
        (["final-eval", "lib_v1"], "no library file"),
        (["data", "publish"], "no dev build"),
        (["loop", "run", "--once"], "dev"),  # no dev dataset in an empty data directory
    ],
)
def test_commands_that_need_something_say_what_is_missing(
    args: list[str], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AF_CONFIG_DIR", str(REPO_CONFIGS))
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert message in result.output.replace("\n", " ")


def test_config_show_redacts_every_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    secrets = {
        "MASSIVE_API_KEY": "s3cr3t-massive-key",
        "MASSIVE_S3_ACCESS_KEY_ID": "s3cr3t-s3-id",
        "MASSIVE_S3_SECRET_ACCESS_KEY": "s3cr3t-s3-secret",
        "LLM_API_KEY": "s3cr3t-llm",
        "DISCORD_WEBHOOK_URL": "https://discord.example/s3cr3t-hook",
    }
    for k, v in secrets.items():
        monkeypatch.setenv(k, v)
    result = runner.invoke(app, ["config", "show"])
    assert result.exit_code == 0
    assert "s3cr3t" not in result.output
    assert "**********" in result.output


def test_synth_build_small(tmp_path: Path) -> None:
    out = str(tmp_path / "out")
    args = ["synth", "build", "--out", out, "--seed", "3", "--tickers", "30", "--sessions", "60"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "dataset_hash" in result.output


def test_data_download_names_missing_s3_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AF_CONFIG_DIR", str(REPO_CONFIGS))
    result = runner.invoke(app, ["data", "download"])
    assert result.exit_code == 1
    for name in ("MASSIVE_S3_ACCESS_KEY_ID", "MASSIVE_S3_SECRET_ACCESS_KEY", "MASSIVE_S3_BUCKET"):
        assert name in result.output.replace("\n", "")


def test_data_ref_names_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AF_CONFIG_DIR", str(REPO_CONFIGS))
    result = runner.invoke(app, ["data", "ref"])
    assert result.exit_code == 1
    assert "MASSIVE_API_KEY" in result.output


def test_eval_factor_reports_dsl_errors_with_the_reason_code() -> None:
    result = runner.invoke(app, ["eval", "factor", "Nope($close)"])
    assert result.exit_code == 1
    assert "invalid_parse" in result.output


def test_eval_factor_accepts_a_leading_minus_and_needs_a_dev_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AF_CONFIG_DIR", str(REPO_CONFIGS))
    result = runner.invoke(app, ["eval", "factor", "-Pct($close, 2)"])
    assert result.exit_code == 1  # parsed fine, then no dev dataset in this empty data dir
    assert "invalid_parse" not in result.output
