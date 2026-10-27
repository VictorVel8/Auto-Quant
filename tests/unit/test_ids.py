import pytest

from alpha_factory.ids import new_id


def test_prefix_and_ulid_shape() -> None:
    i = new_id("trl")
    prefix, _, ulid = i.partition("_")
    assert prefix == "trl"
    assert len(ulid) == 26


def test_ids_are_unique_and_time_sortable() -> None:
    import time

    a = new_id("prp")
    time.sleep(0.002)
    b = new_id("prp")
    assert a != b
    assert a < b


@pytest.mark.parametrize("bad", ["", "T", "trial_x", "TRL", "tr1"])
def test_bad_prefix_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        new_id(bad)
