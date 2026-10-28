"""SIC code -> one of 11 coarse sector groups. More specific prefixes win."""

SECTORS = (
    "energy_mining",
    "materials_chemicals",
    "pharma_health",
    "tech_hardware_semis",
    "software_it",
    "industrials",
    "media_telecom",
    "utilities",
    "consumer",
    "financials",
    "other",
)

_RANGES: dict[str, list[tuple[int, int]]] = {
    "energy_mining": [(10, 14), (29, 29)],
    "materials_chemicals": [(26, 26), (28, 28), (30, 33)],
    "pharma_health": [(80, 80)],
    "tech_hardware_semis": [(36, 36), (38, 38)],
    "industrials": [(15, 17), (34, 35), (37, 37), (40, 47)],
    "media_telecom": [(27, 27), (48, 48), (78, 79)],
    "utilities": [(49, 49)],
    "consumer": [(20, 23), (25, 25), (50, 59), (70, 70), (72, 72)],
    "financials": [(60, 67)],
}

# Prefix (2 or 3 digits) -> sector. Three-digit prefixes override their two-digit parent.
_THREE_DIGIT = {
    "283": "pharma_health",
    "384": "pharma_health",
    "357": "tech_hardware_semis",
    "737": "software_it",
}


def _prefix_table() -> dict[str, str]:
    table = {
        f"{major:02d}": sector
        for sector, rs in _RANGES.items()
        for lo, hi in rs
        for major in range(lo, hi + 1)
    }
    return {**table, **_THREE_DIGIT}


PREFIX_TABLE = _prefix_table()


def sector_for_sic(sic_code: str | int | None) -> str:
    """Map a SIC code to its sector; missing or unknown codes go to "other"."""
    if sic_code is None:
        return "other"
    code = f"{int(sic_code):04d}" if str(sic_code).isdigit() else ""
    if len(code) != 4:
        return "other"
    return PREFIX_TABLE.get(code[:3]) or PREFIX_TABLE.get(code[:2]) or "other"
