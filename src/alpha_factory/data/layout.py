"""Where everything lives under AF_DATA_DIR."""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

Kind = Literal["dev", "full"]

REF_TABLES = ("instruments", "symbols", "splits", "dividends", "sic")


@dataclass(frozen=True)
class DataLayout:
    root: Path

    # raw: immutable downloads
    def raw_minute(self, session: date) -> Path:
        return (
            self.root / "raw" / "minute" / f"{session:%Y}" / f"{session:%m}" / f"{session}.csv.gz"
        )

    def raw_ref_dir(self, endpoint: str) -> Path:
        return self.root / "raw" / "ref" / endpoint

    # normalised reference tables
    def ref_table(self, name: str) -> Path:
        return self.root / "ref" / f"{name}.parquet"

    # per-session raw 30-min bars shared by dev and full builds (before adjustment)
    def bars_cache(self, session: date) -> Path:
        return self.root / "cache" / "bars_raw" / f"session_date={session}" / "part-0.parquet"

    def bars_cache_meta(self, session: date) -> Path:
        return self.bars_cache(session).with_suffix(".meta.json")

    # versioned builds
    @property
    def build_root(self) -> Path:
        return self.root / "build"

    def build_dir(self, dataset_id: str) -> Path:
        return self.build_root / dataset_id

    @property
    def datasets_root(self) -> Path:
        return self.root / "datasets"

    def dataset_link(self, kind: Kind) -> Path:
        return self.datasets_root / kind

    def current_build(self, kind: Kind) -> Path | None:
        link = self.dataset_link(kind)
        return link.resolve() if link.is_symlink() and link.exists() else None
