"""configs/sandbox.yaml -> SandboxSpec, checked against configs/data.yaml and the dev build."""

from pathlib import Path
from typing import Any

import yaml

from alpha_factory.contracts.sandbox import SandboxSpec
from alpha_factory.data.config import DataConfig
from alpha_factory.settings import Settings


class SandboxConfigError(ValueError):
    """The sandbox and the data configuration disagree."""


def load_sandbox(
    settings: Settings | None = None,
    *,
    dataset_hash: str,
    data_cfg: DataConfig | None = None,
    path: Path | None = None,
) -> SandboxSpec:
    """Build the SandboxSpec. `dataset_hash` is the dev build's hash (MarketData("dev").manifest).

    Raises SandboxConfigError when the universe block, the holdout end or the sealed window do
    not line up with data.yaml, so the two files cannot drift apart silently.
    """
    settings = settings or Settings()
    raw: dict[str, Any] = yaml.safe_load((path or settings.config_dir / "sandbox.yaml").read_text())
    spec = SandboxSpec.model_validate({**raw, "dataset_hash": dataset_hash})
    cfg = data_cfg or settings.load("data", DataConfig)

    problems: list[str] = []
    for name in ("top_n", "adv_window_days", "min_price", "tradable_top_n"):
        a, b = getattr(spec.universe, name), getattr(cfg.universe, name)
        if a != b:
            problems.append(f"universe.{name}: sandbox {a} != data {b}")
    if spec.gate_holdout.end != cfg.dev_end:
        problems.append(f"gate_holdout.end {spec.gate_holdout.end} != data dev_end {cfg.dev_end}")
    if spec.sealed_test.end > cfg.end:
        problems.append(f"sealed_test.end {spec.sealed_test.end} is after data end {cfg.end}")
    if min(f.val.start for f in spec.folds) < cfg.start:
        problems.append("a fold starts before the data does")
    if problems:
        raise SandboxConfigError("; ".join(problems))
    return spec
