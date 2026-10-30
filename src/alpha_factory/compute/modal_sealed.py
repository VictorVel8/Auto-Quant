"""Modal deployment of the SEALED final evaluation. Never imported by the research loop.

Written against modal 1.6 and NOT deployed or run. The sealed app lives in its own Modal
environment; `af final-eval` calls it from either machine.
"""

from pathlib import Path

from alpha_factory.compute.modal_dev import CONFIGS, MOUNT, prepare_data_dir

FULL_APP, FULL_VOLUME = "alpha-factory-final", "af-data-full"


def run_final_eval(request_json: str, library_json: str, dev_dataset_hash: str) -> str:
    """Score a frozen library once on the sealed window. The caller supplies the dev dataset hash
    (the sandbox is hashed with it); a wrong one simply fails the sandbox-hash guard."""
    from alpha_factory.contracts import FinalEvalRequest, FrozenLibrary
    from alpha_factory.data import MarketData
    from alpha_factory.eval.library import LibraryScorer
    from alpha_factory.sandbox import load_sandbox
    from alpha_factory.sandbox.final_eval import FinalEvalLedger, final_eval
    from alpha_factory.settings import Settings

    data_dir = prepare_data_dir(Path(MOUNT), "full", Path("/tmp/afdata_full"))
    settings = Settings(data_dir=data_dir, config_dir=Path(CONFIGS))
    sandbox = load_sandbox(settings, dataset_hash=dev_dataset_hash)
    data = MarketData("full", settings=settings, sealed=True)
    lib = FrozenLibrary.model_validate_json(library_json)
    request = FinalEvalRequest.model_validate_json(request_json)
    ledger = FinalEvalLedger(Path(MOUNT) / "ledger.jsonl")
    report = final_eval(request, library=LibraryScorer(lib, sandbox, data), ledger=ledger, sandbox=sandbox, data=data)  # fmt: skip
    return report.model_dump_json()
