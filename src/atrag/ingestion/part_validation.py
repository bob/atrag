"""Validate source slices before accepting prepared or reconstructed evidence."""


def validate_part(unit, part):
    if (part.snapshot_id, part.legal_unit_id) != (unit.snapshot_id, unit.legal_unit_id):
        raise ValueError("Part belongs to another snapshot or legal unit")
    if not 0 <= part.source_start < part.source_end <= len(unit.text_original):
        raise ValueError("Invalid part source range")
    if part.text_original != unit.text_original[part.source_start : part.source_end]:
        raise ValueError("Part text differs from its source range")
    if not 0 <= part.part_index < part.part_count:
        raise ValueError("Invalid part ordinal")
    if part.locators != unit.locators:
        raise ValueError("Part locators differ from the source legal unit")
    if (part.location, part.faq) != (unit.location, unit.faq):
        raise ValueError("Part HTML provenance differs from the source unit")


def reassemble_unit(unit, parts):
    if not parts:
        raise ValueError("No parts to reassemble")
    cursor = 0
    for index, part in enumerate(parts):
        validate_part(unit, part)
        if (part.part_index, part.part_count, part.source_start) != (
            index,
            len(parts),
            cursor,
        ):
            raise ValueError("Part order, count or source coverage mismatch")
        cursor = part.source_end
    if cursor != len(unit.text_original):
        raise ValueError("Incomplete source coverage")
    return "".join(part.text_original for part in parts)
