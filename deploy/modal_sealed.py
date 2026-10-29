"""Deploy the sealed final-evaluation app, from your MacBook only:
    modal deploy --env sealed deploy/modal_sealed.py

The sealed app must live in a separate Modal environment, or a separate workspace if your plan
cannot restrict a token to one environment. The guarantee holds only if the mini PC's credential
cannot reach it.
"""

import modal

from alpha_factory.compute import modal_dev as dev
from alpha_factory.compute import modal_sealed as m

app = modal.App(m.FULL_APP)
volume = modal.Volume.from_name(m.FULL_VOLUME, create_if_missing=True)


@app.function(
    image=dev.image("."),
    volumes={m.MOUNT: volume},
    cpu=4,
    memory=16 * 1024,
    timeout=30 * 60,
    max_containers=1,
)
def final_eval_remote(request_json: str, library_json: str, dev_dataset_hash: str) -> str:
    volume.reload()
    out = m.run_final_eval(request_json, library_json, dev_dataset_hash)
    volume.commit()  # the ledger must persist
    return out
