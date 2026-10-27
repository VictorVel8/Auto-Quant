from pathlib import Path

from alpha_factory.hashing import canonical_json, sha256_bytes, sha256_canonical, sha256_file


def test_key_order_does_not_change_hash() -> None:
    a = {"x": 1, "y": {"b": [1, 2], "a": None}}
    b = {"y": {"a": None, "b": [1, 2]}, "x": 1}
    assert sha256_canonical(a) == sha256_canonical(b)


def test_different_values_change_hash() -> None:
    assert sha256_canonical({"x": 1}) != sha256_canonical({"x": 2})


def test_canonical_json_has_no_whitespace_and_sorted_keys() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_golden_value() -> None:
    assert (
        sha256_canonical({"a": 1})
        == "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
    )


def test_file_and_bytes_agree(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == sha256_bytes(b"hello")
