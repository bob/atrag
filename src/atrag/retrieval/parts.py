"""Explicit original-text and continuation operations, without embeddings."""

from collections import defaultdict

import tiktoken

from atrag.ingestion.part_validation import reassemble_unit
from atrag.ingestion.text_parts import text_ranges
from atrag.retrieval.context import evidence_cost
from atrag.schemas.retrieval import NeighborParts, UnitCoverage, UnitText


def hit_coverage(hits):
    groups = {}
    for hit in hits:
        groups.setdefault(hit.fragment.legal_unit_id, []).append(hit.fragment)
    return [
        UnitCoverage(
            legal_unit_id=identity,
            part_count=parts[0].part_count,
            returned_part_indices=[p.part_index for p in parts],
            source_ranges=[(p.source_start, p.source_end) for p in parts],
            complete=len(parts) == parts[0].part_count,
        )
        for identity, parts in groups.items()
    ]


def _available_notes(unit, units):
    owners = {unit.legal_unit_id}
    parent = units.get(unit.parent_id)
    while parent is not None:
        owners.add(parent.legal_unit_id)
        parent = units.get(parent.parent_id)
    return [
        u.legal_unit_id
        for u in units.values()
        if u.kind == "note" and u.parent_id in owners
    ]


def _available_references(unit, units):
    children = defaultdict(list)
    for child in units.values():
        if child.kind != "note":
            children[child.parent_id].append(child)
    targets, pending = set(), [unit]
    while pending:
        current = pending.pop()
        targets.update(
            i for reference in current.references for i in reference.target_ids
        )
        pending.extend(children[current.legal_unit_id])
    return sorted(targets)


def unit_text(unit, source, units, *, start=0, context_tokens=6000):
    if not 0 <= start < len(unit.text_original):
        raise ValueError("Invalid original-text start")
    encoding = tiktoken.get_encoding("cl100k_base")
    text = unit.text_original[start:]
    measure = lambda a, b: len(
        encoding.encode_ordinary(unit.citation_label + "\n\n" + text[a:b])
    )
    ranges = text_ranges(text, measure, context_tokens, context_tokens)
    _, length = ranges[0]
    end = start + length
    return UnitText(
        snapshot_id=unit.snapshot_id,
        legal_unit_id=unit.legal_unit_id,
        citation_label=unit.citation_label,
        role=unit.role,
        legal_status=unit.legal_status,
        source=source,
        locators=unit.locators,
        source_start=start,
        source_end=end,
        source_length=len(unit.text_original),
        text_original=text[:length],
        complete=start == 0 and end == len(unit.text_original),
        next_start=end if end < len(unit.text_original) else None,
        available_note_ids=_available_notes(unit, units),
        available_reference_ids=_available_references(unit, units),
        context_tokens_used=measure(0, length),
    )


def neighbor_parts(unit, parts, part_index, *, before=0, after=1, context_tokens=6000):
    reassemble_unit(unit, parts)
    if not 0 <= part_index < len(parts):
        raise ValueError("Invalid neighboring part index")
    if min(before, after) < 0:
        raise ValueError("Neighbor counts must be nonnegative")
    if context_tokens <= 0:
        raise ValueError("Context budget must be positive")
    encoding = tiktoken.get_encoding("cl100k_base")
    left, right = max(0, part_index - before), part_index + after + 1
    selected, omitted, used = [], [], 0
    for part in parts[left:right]:
        cost = evidence_cost(part, [], [], encoding)
        if used + cost > context_tokens:
            omitted.append(part.part_index)
        else:
            selected.append(part)
            used += cost
    return NeighborParts(
        snapshot_id=unit.snapshot_id,
        legal_unit_id=unit.legal_unit_id,
        parts=selected,
        omitted_part_indices=omitted,
        complete=len(selected) == len(parts),
        context_tokens_used=used,
    )
