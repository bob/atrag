"""Expand explicitly selected evidence and caller-reviewed source dependencies."""

import json
from collections import defaultdict, deque

import tiktoken

from atrag.retrieval.parts import (
    available_notes,
    available_references,
    unit_text,
    unit_text_cost,
)
from atrag.schemas.retrieval import ExpandedContext


def _selection(request, units):
    root = units.get(request.legal_unit_id)
    if root is None:
        raise ValueError("Legal unit is absent from the selected snapshot")
    if not set(request.reference_ids) <= set(available_references(root, units)):
        raise ValueError("Selected reference is not available from this unit")
    if not set(request.note_ids) <= set(available_notes(root, units)):
        raise ValueError("Selected note is not available from this unit")
    edges = _dependencies(request, units)
    selected = _walk(
        [root.legal_unit_id, *request.reference_ids, *request.note_ids], edges
    )
    if not set(edges) <= set(selected):
        raise ValueError("Dependency is not connected to the selected evidence")
    if any(units[i].snapshot_id != request.snapshot_id for i in selected):
        raise ValueError("Evidence belongs to another snapshot")
    return selected


def _dependencies(request, units):
    edges = defaultdict(list)
    for source, target in request.dependency_edges:
        if source not in units or target not in units:
            raise ValueError("Dependency is absent from the selected snapshot")
        forward = target in available_references(units[source], units)
        reverse = source in available_references(units[target], units)
        if not (forward or reverse):
            raise ValueError("Reviewed dependency has no source reference")
        edges[source].append(target)
    return edges


def _walk(seeds, edges):
    pending = deque(seeds)
    selected = {}
    while pending:
        identity = pending.popleft()
        if identity not in selected:
            selected[identity] = None
            pending.extend(edges[identity])
    return list(selected)


def table_cost(table, encoding):
    return len(
        encoding.encode_ordinary(
            json.dumps(
                {"headers": table.header_rows, "rows": table.rows}, ensure_ascii=False
            )
        )
    )


def _text(unit, source, units, remaining, encoding):
    minimum = unit_text_cost(unit, unit.text_original[0], encoding)
    if remaining < minimum:
        return None
    return unit_text(unit, source, units, context_tokens=remaining)


def _tables(text, unit, tables, delivered, remaining, encoding):
    omitted, used = set(), 0
    for identity in unit.table_ids:
        if identity in delivered:
            continue
        table = tables[identity]
        cost = table_cost(table, encoding)
        if (
            text.complete
            and table.review_status == "verified"
            and used + cost <= remaining
        ):
            delivered[identity] = table
            used += cost
        else:
            omitted.add(identity)
    return omitted, used


def expand_context(request, source, units, tables):
    selected = _selection(request, units)
    encoding = tiktoken.get_encoding("cl100k_base")
    texts, omitted, delivered_tables, omitted_tables = [], [], {}, set()
    used = 0
    for identity in selected:
        unit = units[identity]
        text = _text(unit, source, units, request.context_tokens - used, encoding)
        if text is None:
            omitted.append(identity)
            omitted_tables.update(unit.table_ids)
            continue
        texts.append(text)
        used += text.context_tokens_used
        missing, cost = _tables(
            text,
            unit,
            tables,
            delivered_tables,
            request.context_tokens - used,
            encoding,
        )
        omitted_tables.update(missing)
        used += cost
    missing_tables = sorted(omitted_tables - delivered_tables.keys())
    return ExpandedContext(
        snapshot_id=request.snapshot_id,
        legal_unit_id=request.legal_unit_id,
        units=texts,
        tables=list(delivered_tables.values()),
        omitted_unit_ids=omitted,
        omitted_table_ids=missing_tables,
        complete=not omitted and not missing_tables and all(t.complete for t in texts),
        context_tokens_used=used,
    )
