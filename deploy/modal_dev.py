"""Deploy the dev evaluation app:  modal deploy --env dev deploy/modal_dev.py

Run it from the repository root so the lockfile and configs/ are found. Use a token that is
restricted to the dev environment: the mini PC must never hold the sealed one.
"""

import modal

from alpha_factory.compute import modal_dev as m

app = modal.App(m.DEV_APP)
volume = modal.Volume.from_name(m.DEV_VOLUME, create_if_missing=True)
common = dict(
    image=m.image("."), volumes={m.MOUNT: volume}, cpu=4, memory=16 * 1024, timeout=20 * 60
)


@app.function(max_containers=12, **common)  # type: ignore[arg-type]
def evaluate_batch(request_json: str) -> str:
    volume.reload()
    return m.run_evaluate_batch(request_json)


@app.function(max_containers=2, **common)  # type: ignore[arg-type]
def pool_loo(request_json: str) -> str:
    volume.reload()
    return m.run_pool_loo(request_json)
