"""Forward-return labels: enter at the next bar's open, exit at the close of the k-th bar after.

y_h(t) = close(t + k) / open(t + 1) - 1 with k = 2 * hours, defined only when t + k is still in
the session, so h4 always means four hours. A missing entry or exit bar gives null. Labels are
stored raw (y_*) and cross-sectionally demeaned over members with a label at t (d_*).
"""

from pathlib import Path

import polars as pl

from alpha_factory.contracts.sandbox import SandboxSpec
from alpha_factory.sandbox.panel import Panel

SESSION = ["tix", "sidx"]


def label_columns(sandbox: SandboxSpec) -> list[str]:
    return [c for s in sandbox.labels for c in (f"y_{s.name}", f"d_{s.name}")]


def compute_labels(panel: Panel, sandbox: SandboxSpec) -> pl.DataFrame:
    """Labels for every member row of the panel: tix, ts, then y_<h> and d_<h> per horizon."""
    entry = pl.col("open").shift(-1).over(SESSION)
    raw = [
        pl.when(pl.col("is_member") & (entry > 0))
        .then(
            pl.col("close").shift(-(s.hours * 60 // sandbox.bar_minutes)).over(SESSION) / entry - 1
        )
        .otherwise(None)
        .alias(f"y_{s.name}")
        for s in sandbox.labels
    ]
    df = panel.df.with_columns(raw).with_columns(
        [
            (pl.col(f"y_{s.name}") - pl.col(f"y_{s.name}").mean().over("tsi")).alias(f"d_{s.name}")
            for s in sandbox.labels
        ]
    )
    ys = [f"y_{s.name}" for s in sandbox.labels]
    return df.filter(pl.any_horizontal([pl.col(c).is_not_null() for c in ys])).select(
        "tix", "ts", *label_columns(sandbox)
    )


def labels_cache_path(data_dir: Path, dataset_hash: str, sandbox_hash: str) -> Path:
    """Labels are cached outside the build folder, whose contents the manifest lists."""
    return data_dir / "cache" / "labels" / dataset_hash / f"{sandbox_hash}.parquet"
