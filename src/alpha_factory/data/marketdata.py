"""MarketData: the only way Plans 3 and 4 read market data."""

from collections.abc import Sequence
from datetime import date
from typing import Literal

import polars as pl

from alpha_factory.contracts.data import BAR_SCHEMA, DataManifest, QAReport
from alpha_factory.data.build import read_manifest
from alpha_factory.data.layout import DataLayout
from alpha_factory.settings import Settings

UNIVERSE_COLUMNS = ("rank", "tradable", "is_member")


class SealedDatasetError(PermissionError):
    """The full dataset was opened without the sealed flag."""


class DatasetNotReadyError(RuntimeError):
    """The build is missing, unfinished, or its QA did not pass."""


class MarketData:
    """Read access to a build as Polars LazyFrames.

    Raises if QA failed, or if kind="full" without sealed=True (only final_eval passes it).
    """

    def __init__(
        self,
        kind: Literal["dev", "full"] = "dev",
        *,
        settings: Settings | None = None,
        sealed: bool = False,
    ) -> None:
        if kind == "full" and not sealed:
            raise SealedDatasetError("the full dataset contains the sealed test window")
        self.kind = kind
        layout = DataLayout((settings or Settings()).data_dir)
        build = layout.current_build(kind)
        if build is None or not (build / "manifest.json").exists():
            raise DatasetNotReadyError(f"no complete {kind} build; run `af data build`")
        self._dir = build
        self._manifest = read_manifest(build)
        qa_path = build / "qa.json"
        if not qa_path.exists():
            raise DatasetNotReadyError(f"{kind} build has no QA report; run `af data qa`")
        qa = QAReport.model_validate_json(qa_path.read_text())
        if qa.dataset_hash != self._manifest.dataset_hash:
            raise DatasetNotReadyError("QA report belongs to a different build; rerun `af data qa`")
        if not qa.passed:
            failed = [c.name for c in qa.checks if not c.passed]
            raise DatasetNotReadyError(f"{kind} build failed QA: {failed}")
        if self._manifest.kind != kind:
            raise DatasetNotReadyError(f"datasets/{kind} points at a {self._manifest.kind} build")

    @property
    def manifest(self) -> DataManifest:
        return self._manifest

    def _scan(self, table: str) -> pl.LazyFrame:
        return pl.scan_parquet(self._dir / table / "*" / "part-0.parquet", hive_partitioning=False)

    @staticmethod
    def _window(lf: pl.LazyFrame, start: date | None, end: date | None) -> pl.LazyFrame:
        if start is not None:
            lf = lf.filter(pl.col("session_date") >= start)
        if end is not None:
            lf = lf.filter(pl.col("session_date") <= end)
        return lf

    def sessions(self) -> list[date]:
        names = (p.name for p in (self._dir / "bars_30m").glob("session_date=*"))
        return sorted(date.fromisoformat(n.split("=", 1)[1]) for n in names)

    def bars(
        self,
        start: date | None = None,
        end: date | None = None,
        columns: Sequence[str] | None = None,
        members_only: bool = True,
    ) -> pl.LazyFrame:
        """Bars joined with universe membership (rank, tradable, is_member).

        members_only drops non-members; otherwise they stay with is_member=False.
        """
        if columns is not None:
            unknown = set(columns) - set(BAR_SCHEMA) - set(UNIVERSE_COLUMNS)
            if unknown:
                raise ValueError(f"unknown columns: {sorted(unknown)}")
        bars = self._window(self._scan("bars_30m"), start, end)
        members = self._window(self._scan("universe"), start, end).select(
            "session_date", "ticker_id", "rank", "tradable"
        )
        if members_only:
            out = bars.join(members, on=["session_date", "ticker_id"], how="inner")
            out = out.with_columns(pl.lit(True).alias("is_member"))
        else:
            out = bars.join(members, on=["session_date", "ticker_id"], how="left").with_columns(
                pl.col("rank").is_not_null().alias("is_member"),
                pl.col("tradable").fill_null(False),
            )
        return out.select(list(columns)) if columns is not None else out

    def universe(self, start: date | None = None, end: date | None = None) -> pl.LazyFrame:
        return self._window(self._scan("universe"), start, end)

    def sectors(self) -> pl.DataFrame:
        return pl.read_parquet(self._dir / "sectors.parquet")
