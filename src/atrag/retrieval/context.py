"""Reserve primary evidence first, then add complete one-hop reference groups."""

import json

from atrag.schemas.retrieval import SearchHit


def reference_targets(fragment):
    return sorted(
        {
            target
            for reference in fragment.references
            if reference.status == "resolved"
            for target in reference.target_ids
        }
    )


def _reference_group(target, units):
    selected = set()
    unit = units[target]
    while unit.kind != "document":
        selected.add(unit.legal_unit_id)
        if unit.parent_id is None:
            break
        unit = units[unit.parent_id]
    notes = {
        u.legal_unit_id
        for u in units.values()
        if u.kind == "note" and u.parent_id in selected
    }
    return selected | notes


def _cost(hit, encoding):
    return evidence_cost(hit.fragment, hit.related_units, hit.related_tables, encoding)


def evidence_cost(fragment, related_units, related_tables, encoding):
    # The budget covers legal evidence; PDF geometry and processing metadata
    # remain available to the caller without displacing the actual source text.
    content = {
        "text": fragment.embedding_text,
        "notes": [n.raw_text for n in fragment.editorial_notes],
        "related_units": [
            {
                "citation": u.citation_label,
                "text": u.text_original,
                "notes": [n.raw_text for n in u.editorial_notes],
            }
            for u in related_units
        ],
        "tables": [
            {"headers": t.header_rows, "rows": t.rows}
            for t in [*fragment.tables, *related_tables]
        ],
    }
    return len(encoding.encode_ordinary(json.dumps(content, ensure_ascii=False)))


def _primary_hits(rankings, fragments, request, encoding):
    hits, used, limited = [], 0, False
    for ranking in rankings:
        fragment = fragments[ranking.fragment_id]
        hit = SearchHit(
            **ranking.model_dump(),
            fragment=fragment,
            related_units=[],
            related_tables=[],
            omitted_reference_ids=reference_targets(fragment),
        )
        cost = _cost(hit, encoding)
        if used + cost > request.context_tokens:
            limited = True
            continue
        hits.append(hit)
        used += cost
        if len(hits) == request.limit:
            break
    return hits, used, limited


def _with_reference(hit, target, units, tables):
    ids = {u.legal_unit_id for u in hit.related_units} | _reference_group(target, units)
    related = [units[i] for i in sorted(ids)]
    table_ids = {identity for unit in related for identity in unit.table_ids}
    return hit.model_copy(
        update={
            "related_units": related,
            "related_tables": [tables[i] for i in sorted(table_ids)],
            "omitted_reference_ids": [
                i for i in hit.omitted_reference_ids if i != target
            ],
        }
    )


def assemble_hits(rankings, fragments, units, tables, request, encoding):
    hits, used, limited = _primary_hits(rankings, fragments, request, encoding)
    for index, hit in enumerate(hits):
        for target in reference_targets(hit.fragment):
            expanded = _with_reference(hit, target, units, tables)
            extra = _cost(expanded, encoding) - _cost(hit, encoding)
            if used + extra > request.context_tokens:
                limited = True
                continue
            hit = expanded
            used += extra
        hits[index] = hit
    return hits, used, limited
