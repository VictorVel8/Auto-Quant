import pytest

from alpha_factory.data.sectors import _RANGES, PREFIX_TABLE, SECTORS, sector_for_sic  # pyright: ignore[reportPrivateUsage]


def test_every_sic_code_lands_in_exactly_one_known_group() -> None:
    for code in range(100, 10000):
        assert sector_for_sic(f"{code:04d}") in SECTORS


def test_two_digit_ranges_do_not_overlap() -> None:
    majors = [m for rs in _RANGES.values() for lo, hi in rs for m in range(lo, hi + 1)]
    assert len(majors) == len(set(majors))
    assert set(PREFIX_TABLE.values()) <= set(SECTORS)


@pytest.mark.parametrize(
    ("sic", "sector"),
    [
        ("1311", "energy_mining"),
        ("2911", "energy_mining"),
        ("2810", "materials_chemicals"),  # 28 except 283
        ("2834", "pharma_health"),  # 283 beats 28
        ("3841", "pharma_health"),  # 384 beats 38
        ("3826", "tech_hardware_semis"),
        ("3571", "tech_hardware_semis"),  # 357 beats 35
        ("3523", "industrials"),
        ("3674", "tech_hardware_semis"),
        ("7372", "software_it"),  # 737 only
        ("7389", "other"),  # rest of 73
        ("3711", "industrials"),
        ("4813", "media_telecom"),
        ("4911", "utilities"),
        ("5812", "consumer"),
        ("6021", "financials"),
        ("8000", "pharma_health"),
        ("0100", "other"),
        ("9999", "other"),
        (None, "other"),
        ("", "other"),
        ("abc", "other"),
        (2834, "pharma_health"),
    ],
)
def test_known_mappings(sic: str | int | None, sector: str) -> None:
    assert sector_for_sic(sic) == sector
