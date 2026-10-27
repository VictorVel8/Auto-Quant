"""Fail if a data file is tracked outside tests/fixtures/. Used by pre-commit and CI.

With file arguments (pre-commit) it checks those paths; without, every tracked file.
"""

import subprocess
import sys

FORBIDDEN_SUFFIXES = (".parquet", ".csv.gz", ".sqlite")
ALLOWED_PREFIX = "tests/fixtures/"


def offenders(paths: list[str]) -> list[str]:
    return [p for p in paths if p.endswith(FORBIDDEN_SUFFIXES) and not p.startswith(ALLOWED_PREFIX)]


def main(argv: list[str]) -> int:
    paths = (
        argv
        or subprocess.run(
            ["git", "ls-files"], capture_output=True, text=True, check=True
        ).stdout.splitlines()
    )
    bad = offenders(paths)
    for p in bad:
        print(f"data guard: {p} must not be tracked (only tests/fixtures/ may hold data)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
