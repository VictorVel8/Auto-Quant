"""Prefixed, time-sortable ids: new_id("trl") -> "trl_01J..."."""

import re

from ulid import ULID

_PREFIX = re.compile(r"^[a-z]{2,5}$")


def new_id(prefix: str) -> str:
    """Return `<prefix>_<ULID>`. Safe to generate on any host."""
    if not _PREFIX.match(prefix):
        raise ValueError(f"invalid id prefix: {prefix!r}")
    return f"{prefix}_{ULID()}"
