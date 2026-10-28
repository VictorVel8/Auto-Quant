# pyright: basic
"""Network clients: S3 flat files and the REST reference API. Both are injectable for tests."""

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import httpx

from alpha_factory.data.massive import REST_BASE_URL
from alpha_factory.hashing import sha256_file


def with_retries[T](fn: Callable[[], T], retries: int, base_delay: float = 1.0) -> T:
    """Call fn, retrying with exponential backoff on any exception."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception:
            if attempt == retries:
                raise
            time.sleep(base_delay * 2**attempt)
    raise AssertionError("unreachable")


class ObjectStore(Protocol):
    def size(self, key: str) -> int | None:
        """Object size in bytes, or None if the key does not exist."""
        ...

    def get(self, key: str, dest: Path) -> None: ...


class S3Store:
    """boto3 client for the Massive flat-file bucket."""

    def __init__(self, endpoint: str, bucket: str, key_id: str, secret: str) -> None:
        import boto3
        from botocore.config import Config

        self._bucket = bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            config=Config(signature_version="s3v4", retries={"max_attempts": 1}),
        )

    def size(self, key: str) -> int | None:
        from botocore.exceptions import ClientError

        try:
            return int(self._client.head_object(Bucket=self._bucket, Key=key)["ContentLength"])
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
                return None
            raise

    def get(self, key: str, dest: Path) -> None:
        self._client.download_file(self._bucket, key, str(dest))


def download_file(store: ObjectStore, key: str, dest: Path, retries: int) -> str:
    """Download once, resumable at file level. Returns "downloaded", "skipped" or "missing".

    A file already present with the remote size is never touched again (raw is immutable). The
    download goes to a temp name, then is renamed and checksummed.
    """
    remote = with_retries(lambda: store.size(key), retries)
    if remote is None:
        return "missing"
    if dest.exists() and dest.stat().st_size == remote:
        return "skipped"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with_retries(lambda: store.get(key, tmp), retries)
    if tmp.stat().st_size != remote:
        tmp.unlink()
        raise OSError(f"size mismatch for {key}")
    tmp.rename(dest)
    dest.with_name(dest.name + ".sha256").write_text(sha256_file(dest) + "\n")
    return "downloaded"


class RestClient:
    """Paginated GET against the Massive reference API."""

    def __init__(
        self, api_key: str, retries: int = 5, transport: httpx.BaseTransport | None = None
    ) -> None:
        self._retries = retries
        self._http = httpx.Client(
            base_url=REST_BASE_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60,
            transport=transport,
        )

    def get_json(self, url: str, params: dict[str, str] | None = None) -> dict[str, Any]:
        def call() -> dict[str, Any]:
            r = self._http.get(url, params=params)
            r.raise_for_status()
            return r.json()

        return with_retries(call, self._retries)

    def get_all(self, path: str, params: dict[str, str]) -> list[dict[str, Any]]:
        """Follow next_url until exhausted and return every result row."""
        rows: list[dict[str, Any]] = []
        page = self.get_json(path, params)
        while True:
            rows.extend(page.get("results") or [])
            nxt = page.get("next_url")
            if not nxt:
                return rows
            page = self.get_json(nxt)


def write_cached_json(directory: Path, fetch_date: str, payload: dict[str, Any]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{fetch_date}.json"
    path.write_text(json.dumps(payload, sort_keys=True))
    return path
