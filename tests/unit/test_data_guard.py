import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "data_guard", Path(__file__).parents[2] / "scripts" / "data_guard.py"
)
assert spec is not None and spec.loader is not None
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_flags_data_files_outside_fixtures() -> None:
    paths = ["a.py", "x/bars.parquet", "b.csv.gz", "store.sqlite", "tests/fixtures/ok.parquet"]
    assert guard.offenders(paths) == ["x/bars.parquet", "b.csv.gz", "store.sqlite"]


def test_clean_paths_pass() -> None:
    assert guard.main(["README.md", "tests/fixtures/ok.parquet"]) == 0
    assert guard.main(["data.parquet"]) == 1
