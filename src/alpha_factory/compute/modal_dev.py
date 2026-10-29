"""Modal deployment of the DEV evaluation worker (the sealed side is modal_sealed.py).

Written against modal 1.6. It has NOT been deployed or run: that needs a Modal account and
credentials, and the dev and sealed tokens must be separate (see docs/research.md).

This module is imported by the research loop, so it must not contain or import anything sealed:
the final evaluation lives in modal_sealed.py, and an import-linter contract keeps it that way.

Two deployments, one boundary, which is a credential the mini PC does not have (the app
definitions are in deploy/modal_dev.py and deploy/modal_sealed.py):

| | dev | sealed |
|---|---|---|
| app | alpha-factory-eval | alpha-factory-final |
| volume | af-data-dev (dev build, calibration) | af-data-full (full build, final-eval ledger) |
| function | evaluate_batch, pool_loo | final_eval_remote |
| called from | the orchestrator on the mini PC | `af final-eval` on your MacBook |

The functions take and return JSON strings, so no pydantic class has to exist on the caller's
image. The worker itself is alpha_factory.compute.worker; nothing here touches the research store.
"""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from alpha_factory.compute.specs import (
    EvalBatchRequest,
    EvalBatchResponse,
    PoolLooRequest,
    PoolLooResponse,
)
from alpha_factory.compute.worker import EvalBackend

DEV_APP, DEV_VOLUME = "alpha-factory-eval", "af-data-dev"
MOUNT = "/data"
CONFIGS = "/app/configs"
CHUNK = 3  # specs per container: amortises loading the data


def prepare_data_dir(mount: Path, kind: str, workdir: Path) -> Path:
    """A data directory in Plan 2's layout, built from a volume that holds build/ and a pointer.

    The volume has `build/<dataset_id>/...`, `current_<kind>.txt` naming the build to use, and
    optionally `calibration/`. MarketData expects datasets/<kind> to be a link to a build.
    """
    dataset_id = (mount / f"current_{kind}.txt").read_text().strip()
    (workdir / "datasets").mkdir(parents=True, exist_ok=True)
    for link, target in (
        (workdir / "build", mount / "build"),
        (workdir / "datasets" / kind, Path("..") / "build" / dataset_id),
        (workdir / "calibration", mount / "calibration"),
    ):
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(target)
    return workdir


# --- the remote functions ---------------------------------------------------------------------
_EVALUATOR: Any = None


def _evaluator() -> Any:
    """One Evaluator per container, built on first use from the mounted volume."""
    global _EVALUATOR
    if _EVALUATOR is None:
        from alpha_factory.compute.worker import Evaluator
        from alpha_factory.data import MarketData
        from alpha_factory.eval.calibration import load_calibration
        from alpha_factory.sandbox import load_sandbox
        from alpha_factory.settings import Settings

        data_dir = prepare_data_dir(Path(MOUNT), "dev", Path("/tmp/afdata"))
        settings = Settings(data_dir=data_dir, config_dir=Path(CONFIGS))
        data = MarketData("dev", settings=settings)
        sandbox = load_sandbox(settings, dataset_hash=data.manifest.dataset_hash)
        cal = load_calibration(data_dir, sandbox.content_hash(), data.manifest.dataset_hash)
        floor = {str(k): v for k, v in cal.ic_floor.items()} if cal else None
        _EVALUATOR = Evaluator(data, sandbox, floor)
    return _EVALUATOR


def run_evaluate_batch(request_json: str) -> str:
    request = EvalBatchRequest.model_validate_json(request_json)
    return _evaluator().evaluate_batch(request).model_dump_json()


def run_pool_loo(request_json: str) -> str:
    return _evaluator().pool_loo(PoolLooRequest.model_validate_json(request_json)).model_dump_json()


def image(project_dir: str = ".") -> Any:
    """The container image: the project's lockfile (data, engine and loop groups), the package
    source and the configs. Modal needs the decorated functions at module level, so the apps are
    defined in deploy/modal_dev.py and deploy/modal_sealed.py and use this helper."""
    import modal

    return (
        modal.Image.debian_slim(python_version="3.12")
        .uv_sync(
            project_dir, groups=["data", "engine", "loop"], extra_options="--no-install-project"
        )
        .add_local_python_source("alpha_factory")
        .add_local_dir(f"{project_dir}/configs", CONFIGS)
    )


# --- the caller's side ----------------------------------------------------------------------------
def chunks[T](items: Sequence[T], size: int) -> list[list[T]]:
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


class ModalBackend(EvalBackend):
    """Fans a batch out to the dev deployment in chunks of a few specs per container."""

    def __init__(
        self,
        environment: str = "dev",
        chunk: int = CHUNK,
        evaluate_fn: Any = None,
        loo_fn: Any = None,
    ) -> None:
        self.chunk = chunk
        if evaluate_fn is None or loo_fn is None:
            import modal

            evaluate_fn = evaluate_fn or modal.Function.from_name(
                DEV_APP, "evaluate_batch", environment_name=environment
            )
            loo_fn = loo_fn or modal.Function.from_name(
                DEV_APP, "pool_loo", environment_name=environment
            )
        self._evaluate, self._loo = evaluate_fn, loo_fn

    def evaluate(self, request: EvalBatchRequest) -> EvalBatchResponse:
        total = request.batch_size or len(request.specs)
        parts = [
            request.model_copy(update={"specs": tuple(c), "batch_size": total})
            for c in chunks(request.specs, self.chunk)
        ]
        replies = list(self._evaluate.map([p.model_dump_json() for p in parts], order_outputs=True))
        outcomes = [o for r in replies for o in EvalBatchResponse.model_validate_json(r).outcomes]
        return EvalBatchResponse(outcomes=tuple(outcomes))

    def pool_loo(self, request: PoolLooRequest) -> PoolLooResponse:
        return PoolLooResponse.model_validate_json(self._loo.remote(request.model_dump_json()))


def publish_build(
    build_dir: Path,
    kind: str,
    environment: str,
    volume_name: str = DEV_VOLUME,
    calibration: Path | None = None,
) -> str:  # fmt: skip
    """Upload a build (and the null calibration) to the volume for `kind`. Returns the dataset id.

    The dev build goes to af-data-dev from anywhere. The full build is uploaded only from the
    MacBook, with the sealed credential, to af-data-full (the caller passes that volume's name).
    """
    import modal

    dataset_id = build_dir.name
    volume = modal.Volume.from_name(
        volume_name, environment_name=environment, create_if_missing=True
    )
    with volume.batch_upload(force=True) as up:
        up.put_directory(str(build_dir), f"/build/{dataset_id}")
        pointer = Path(os.environ.get("TMPDIR", "/tmp")) / f"current_{kind}.txt"
        pointer.write_text(dataset_id)
        up.put_file(str(pointer), f"/current_{kind}.txt")
        if calibration is not None:
            up.put_file(str(calibration), f"/calibration/{calibration.name}")
    return dataset_id
