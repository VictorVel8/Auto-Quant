from pathlib import Path

import httpx
import pytest

from alpha_factory.data import clients
from alpha_factory.hashing import sha256_file


class FakeStore:
    def __init__(
        self, objects: dict[str, bytes], fail_first: int = 0, corrupt: bool = False
    ) -> None:
        self.objects, self.fail_first, self.corrupt, self.gets = objects, fail_first, corrupt, 0

    def size(self, key: str) -> int | None:
        return len(self.objects[key]) if key in self.objects else None

    def get(self, key: str, dest: Path) -> None:
        self.gets += 1
        if self.gets <= self.fail_first:
            raise ConnectionError("boom")
        data = self.objects[key]
        dest.write_bytes(data[:-1] if self.corrupt else data)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(clients.time, "sleep", lambda _s: None)


def test_download_then_skip_and_checksum(tmp_path: Path) -> None:
    store = FakeStore({"k": b"hello"})
    dest = tmp_path / "a" / "k.csv.gz"
    assert clients.download_file(store, "k", dest, retries=2) == "downloaded"
    assert dest.read_bytes() == b"hello"
    assert dest.with_name("k.csv.gz.sha256").read_text().strip() == sha256_file(dest)
    assert clients.download_file(store, "k", dest, retries=2) == "skipped"
    assert store.gets == 1


def test_missing_key(tmp_path: Path) -> None:
    assert clients.download_file(FakeStore({}), "nope", tmp_path / "x", retries=0) == "missing"
    assert not (tmp_path / "x").exists()


def test_retries_transient_failures(tmp_path: Path) -> None:
    store = FakeStore({"k": b"data"}, fail_first=2)
    assert clients.download_file(store, "k", tmp_path / "k", retries=3) == "downloaded"
    assert store.gets == 3


def test_gives_up_after_retries(tmp_path: Path) -> None:
    store = FakeStore({"k": b"data"}, fail_first=99)
    with pytest.raises(ConnectionError):
        clients.download_file(store, "k", tmp_path / "k", retries=2)
    assert not (tmp_path / "k").exists()


def test_size_mismatch_is_not_kept(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="size mismatch"):
        clients.download_file(
            FakeStore({"k": b"data"}, corrupt=True), "k", tmp_path / "k", retries=0
        )
    assert not (tmp_path / "k").exists()
    assert not (tmp_path / "k.part").exists()


def test_rest_client_follows_next_url_and_sends_bearer() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "page2" in str(request.url):
            return httpx.Response(200, json={"results": [{"n": 3}]})
        return httpx.Response(
            200,
            json={"results": [{"n": 1}, {"n": 2}], "next_url": "https://api.massive.com/x?page2=1"},
        )

    client = clients.RestClient("KEY", transport=httpx.MockTransport(handler))
    rows = client.get_all("/stocks/v1/splits", {"limit": "2"})
    assert [r["n"] for r in rows] == [1, 2, 3]
    assert all(r.headers["authorization"] == "Bearer KEY" for r in seen)


def test_rest_client_retries_server_errors() -> None:
    calls = {"n": 0}

    def handler(_r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] < 3 else httpx.Response(200, json={"results": []})

    client = clients.RestClient("KEY", retries=3, transport=httpx.MockTransport(handler))
    assert client.get_all("/p", {}) == []
    assert calls["n"] == 3
