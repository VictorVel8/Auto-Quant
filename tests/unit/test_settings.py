from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from alpha_factory.hashing import sha256_file
from alpha_factory.settings import MissingSettingError, Settings


class Demo(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    a: int


def test_defaults() -> None:
    s = Settings()
    assert s.env == "dev"
    assert s.config_dir == Path("configs")
    assert s.llm_api_key is None


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AF_ENV", "prod")
    monkeypatch.setenv("AF_DATA_DIR", "/x")
    monkeypatch.setenv("LLM_API_KEY", "sk-secret")
    s = Settings()
    assert s.env == "prod"
    assert s.data_dir == Path("/x")
    assert s.llm_api_key is not None
    assert s.llm_api_key.get_secret_value() == "sk-secret"


def test_secrets_never_print(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASSIVE_API_KEY", "supersecret")
    s = Settings()
    assert "supersecret" not in repr(s)
    assert "supersecret" not in str(s.redacted())


def test_require_names_the_missing_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "http://x")
    s = Settings()
    s.require("llm_base_url")
    with pytest.raises(MissingSettingError, match="LLM_API_KEY"):
        s.require("llm_base_url", "llm_api_key")


def test_load_yaml_and_unknown_keys_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "demo.yaml").write_text("a: 3\n")
    (tmp_path / "bad.yaml").write_text("a: 3\nb: 4\n")
    monkeypatch.setenv("AF_CONFIG_DIR", str(tmp_path))
    s = Settings()
    assert s.load("demo", Demo).a == 3
    with pytest.raises(ValueError):
        s.load("bad", Demo)


def test_config_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "loop.yaml").write_text("x: 1\n")
    monkeypatch.setenv("AF_CONFIG_DIR", str(tmp_path))
    assert Settings().config_hashes() == {"loop": sha256_file(tmp_path / "loop.yaml")}


def test_prod_refuses_dirty_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "f").write_text("x")
    monkeypatch.setenv("AF_ENV", "prod")
    with pytest.raises(RuntimeError, match="clean"):
        Settings().check_prod_tree_clean()
    Settings(env="dev").check_prod_tree_clean()


def test_empty_values_from_a_copied_env_example_use_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("AF_ENV=\nAF_LOG_JSON=\nLLM_API_KEY=\n")
    s = Settings()
    assert s.env == "dev"
    assert s.log_json is False
    assert s.llm_api_key is None
