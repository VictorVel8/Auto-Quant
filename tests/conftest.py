from pathlib import Path

import pytest

from alpha_factory.synthetic.market import SyntheticMarket


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Isolate tests from the developer's .env and AF_/secret variables."""
    import os

    for key in list(os.environ):
        if key.startswith(("AF_", "MASSIVE_", "LLM_", "MODAL_", "DISCORD_")):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synth")
    SyntheticMarket(n_tickers=200, n_sessions=500).build(out)
    return out
