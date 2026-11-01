"""Validate relationships once, before any corpus transaction is opened."""

import hashlib
from collections import defaultdict
from graphlib import CycleError, TopologicalSorter

import tiktoken

from atrag.ingestion.part_validation import reassemble_unit, validate_part
from atrag.ingestion.table_parts import (
    select_tables,
    table_rows,
    table_text,
    validate_table_boundaries,
)
from atrag.schemas.ingestion.packages import ChunkQuality


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def unique_by(items, field):
    result = {getattr(item, field): item for item in items}
    if len(result) != len(items):
        raise ValueError(f"Duplicate {field}")
    return result


def validate_units(document):
    units = unique_by(document.units, "legal_unit_id")
    tables = unique_by(document.tables, "table_id")
    graph = {}
    for unit in units.values():
        validate_pages(unit)
        if unit.snapshot_id != document.snapshot.snapshot_id:
            raise ValueError("Legal unit belongs to another snapshot")
        parents = [] if unit.parent_id is None else [unit.parent_id]
        targets = [t for r in unit.references for t in r.target_ids]
        if not {*parents, *targets}.issubset(units):
            raise ValueError("Missing legal unit relationship")
        if not set(unit.table_ids).issubset(tables):
            raise ValueError("Missing legal table")
        graph[unit.legal_unit_id] = parents
    try:
        list(TopologicalSorter(graph).static_order())
    except CycleError:
        raise ValueError("Cyclic legal unit parent relationship") from None
    return units, tables


def validate_fragment(fragment, document, units, tables, encoding):
    validate_pages(fragment)
    if fragment.snapshot_id != document.snapshot.snapshot_id:
        raise ValueError("Fragment belongs to another snapshot")
    if fragment.source != document.snapshot.source:
        raise ValueError("Fragment source differs from snapshot")
    targets = {
        target for reference in fragment.references for target in reference.target_ids
    }
    ids = {
        fragment.legal_unit_id,
        *fragment.legal_unit_ids,
        *fragment.context_unit_ids,
        *targets,
    }
    if not ids.issubset(units):
        raise ValueError("Missing fragment legal unit relationship")
    if fragment.legal_unit_id not in fragment.legal_unit_ids:
        raise ValueError("Fragment content must include its primary legal unit")
    if [c.legal_unit_id for c in fragment.context] != fragment.context_unit_ids:
        raise ValueError("Fragment context relationships differ")
    unit = units[fragment.legal_unit_id]
    validate_part(unit, fragment)
    validate_fragment_tables(fragment, unit, tables)
    validate_fragment_identity(fragment, document.manifest, encoding)


def validate_fragment_tables(fragment, unit, tables):
    if {t.table_id for t in fragment.tables} != set(fragment.table_ids):
        raise ValueError("Fragment tables do not match its table identifiers")
    rows = table_rows(unit, tables)
    validate_table_boundaries(rows, fragment.source_start, fragment.source_end)
    selected, ranges = select_tables(
        rows, tables, fragment.source_start, fragment.source_end
    )
    if fragment.tables != selected or fragment.table_ranges != ranges:
        raise ValueError("Fragment tables differ from prepared tables")


def validate_pages(unit):
    if unit.page_end < unit.page_start:
        raise ValueError("Invalid physical page range")


def validate_fragment_identity(fragment, manifest, encoding):
    if (fragment.parser_version, fragment.chunker_version) != (
        manifest.parser_version,
        manifest.chunker_version,
    ):
        raise ValueError("Fragment processing versions differ from package")
    if fragment.embedding_input_hash != digest(fragment.embedding_text.encode()):
        raise ValueError("Fragment embedding input hash mismatch")
    if fragment.token_count != len(encoding.encode_ordinary(fragment.embedding_text)):
        raise ValueError("Fragment token count mismatch")
    if (
        max(
            fragment.token_count,
            len(encoding.encode_ordinary(table_text(fragment.tables))),
        )
        > fragment.parameters.max_tokens
    ):
        raise ValueError("Fragment exceeds its token limit")
    identity = fragment.model_dump_json(exclude={"fragment_id", "source"})
    if fragment.fragment_id != digest(identity.encode()):
        raise ValueError("Fragment identity mismatch")


def validate_part_groups(fragments, units):
    groups = defaultdict(list)
    for part in fragments:
        groups[part.legal_unit_id].append(part)
    for identity, parts in groups.items():
        reassemble_unit(units[identity], sorted(parts, key=lambda p: p.part_index))


def validate_embeddings(document, fragments):
    vectors = unique_by(document.embeddings, "fragment_id")
    if vectors.keys() != fragments.keys():
        raise ValueError("Embeddings do not cover the complete fragment set")
    inputs = {}
    for identity, vector in vectors.items():
        expected = (
            document.snapshot.snapshot_id,
            document.manifest.model,
            document.manifest.dimensions,
            fragments[identity].embedding_input_hash,
        )
        if (
            vector.snapshot_id,
            vector.model,
            vector.dimensions,
            vector.input_hash,
        ) != expected:
            raise ValueError("Embedding provenance or space mismatch")
        previous = inputs.setdefault(vector.input_hash, vector.embedding)
        if previous != vector.embedding:
            raise ValueError("Conflicting package vectors for one embedding input")


def validate_document(document):
    report = ChunkQuality.model_validate(document.reports.get("chunk"))
    if (report.snapshot_id, report.fragment_count) != (
        document.manifest.snapshot_id,
        document.manifest.fragment_count,
    ):
        raise ValueError("Chunk report identity or count mismatch")
    units, tables = validate_units(document)
    fragments = unique_by(document.fragments, "fragment_id")
    if len(fragments) != document.manifest.fragment_count:
        raise ValueError("Package fragment count mismatch")
    encoding = tiktoken.get_encoding("cl100k_base")
    for fragment in fragments.values():
        validate_fragment(fragment, document, units, tables, encoding)
    validate_part_groups(fragments.values(), units)
    validate_embeddings(document, fragments)
