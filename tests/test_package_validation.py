"""Relationship checks reject corrupt prepared evidence before indexing."""

from dataclasses import replace

import pytest

from atrag.ingestion.package_validation import validate_document
from atrag.ingestion.services.packages import build_package, load_package


def with_fragment(document, fragment):
    from atrag.ingestion.package_validation import digest

    identity = digest(
        fragment.model_dump_json(exclude={"fragment_id", "source"}).encode()
    )
    fragment = fragment.model_copy(update={"fragment_id": identity})
    vector = document.embeddings[0].model_copy(update={"fragment_id": identity})
    return replace(
        document,
        fragments=[fragment, *document.fragments[1:]],
        embeddings=[vector, *document.embeddings[1:]],
    )


@pytest.fixture
def document(prepared_document, tmp_path):
    data, parsed, chunked, embedded = prepared_document
    return load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "package")
    )


@pytest.mark.parametrize(
    "field,item",
    [
        ("units", "legal_unit_id"),
        ("tables", "table_id"),
        ("fragments", "fragment_id"),
        ("embeddings", "fragment_id"),
    ],
)
def test_duplicate_identities_are_rejected(document, field, item):
    values = getattr(document, field)
    with pytest.raises(ValueError, match=f"Duplicate {item}"):
        validate_document(replace(document, **{field: [*values, values[0]]}))


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"snapshot_id": "b" * 64}, "another snapshot"),
        ({"parent_id": "missing"}, "relationship"),
        ({"table_ids": ["missing"]}, "table"),
        ({"page_start": 2, "page_end": 1}, "page"),
    ],
)
def test_legal_units_are_checked(document, updates, reason):
    unit = document.units[0].model_copy(update=updates)
    with pytest.raises(ValueError, match=reason):
        validate_document(replace(document, units=[unit, *document.units[1:]]))


def test_parent_cycles_are_rejected(document):
    first = document.units[0]
    unit = first.model_copy(update={"parent_id": first.legal_unit_id})
    with pytest.raises(ValueError, match="Cyclic"):
        validate_document(replace(document, units=[unit, *document.units[1:]]))


@pytest.mark.parametrize(
    "updates,reason",
    [
        ({"snapshot_id": "b" * 64}, "another snapshot"),
        ({"legal_unit_id": "missing"}, "relationship"),
        ({"context_unit_ids": []}, "context"),
        ({"embedding_input_hash": "b" * 64}, "input hash"),
        ({"token_count": 1}, "token count"),
        ({"fragment_id": "b" * 64}, "identity"),
        ({"parser_version": "other"}, "versions"),
        ({"legal_unit_ids": []}, "content"),
    ],
)
def test_fragments_are_checked(document, updates, reason):
    fragment = document.fragments[0].model_copy(update=updates)
    with pytest.raises(ValueError, match=reason):
        validate_document(
            replace(document, fragments=[fragment, *document.fragments[1:]])
        )


def test_fragment_source_and_tables_must_match(document):
    first = document.fragments[0]
    other = first.model_copy(
        update={"source": first.source.model_copy(update={"publisher": "other"})}
    )
    with pytest.raises(ValueError, match="source"):
        validate_document(replace(document, fragments=[other, *document.fragments[1:]]))
    other = first.model_copy(
        update={"tables": [document.tables[0].model_copy(update={"rows": [["wrong"]]})]}
    )
    with pytest.raises(ValueError, match="tables"):
        validate_document(replace(document, fragments=[other, *document.fragments[1:]]))
    other = first.model_copy(update={"table_ids": ["absent"]})
    with pytest.raises(ValueError, match="tables"):
        validate_document(with_fragment(document, other))


def test_fragment_cannot_introduce_unknown_reference_target(document):
    first = document.fragments[0]
    reference = first.references[0].model_copy(update={"target_ids": ["absent"]})
    other = first.model_copy(update={"references": [reference]})
    with pytest.raises(ValueError, match="relationship"):
        validate_document(with_fragment(document, other))


def test_quality_report_fields_are_checked_before_indexing(document):
    reports = {
        **document.reports,
        "chunk": {**document.reports["chunk"], "extraction_issues": None},
    }
    with pytest.raises(ValueError):
        validate_document(replace(document, reports=reports))


@pytest.mark.parametrize(
    "updates",
    [
        {"source_start": 1},
        {"source_end": 999999},
        {"part_index": 1},
        {"part_count": 2},
        {"text_original": "not the original"},
        {"locators": []},
    ],
)
def test_rehashed_parts_still_have_to_match_the_original(document, updates):
    fragment = document.fragments[0].model_copy(update=updates)
    with pytest.raises(ValueError):
        validate_document(with_fragment(document, fragment))


def test_fragment_limit_is_checked_even_with_a_correct_hash(document):
    first = document.fragments[0]
    from atrag.schemas.ingestion.fragments import ChunkParameters

    fragment = first.model_copy(
        update={"parameters": ChunkParameters(target_tokens=1, max_tokens=1)}
    )
    with pytest.raises(ValueError, match="limit"):
        validate_document(with_fragment(document, fragment))
    at_limit = first.model_copy(
        update={
            "parameters": ChunkParameters(
                target_tokens=first.token_count, max_tokens=first.token_count
            )
        }
    )
    validate_document(with_fragment(document, at_limit))


def test_table_values_match_the_checked_table(document):
    last = document.fragments[-1]
    wrong = last.tables[0].model_copy(update={"rows": [["wrong"]]})
    fragment = last.model_copy(update={"tables": [wrong]})
    with pytest.raises(ValueError, match="differ from prepared tables"):
        validate_document(
            replace(document, fragments=[*document.fragments[:-1], fragment])
        )


def test_fragment_count_matches_the_prepared_manifest(document):
    with pytest.raises(ValueError, match="Package fragment count"):
        validate_document(replace(document, fragments=document.fragments[1:]))


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "text-embedding-3-large"},
        {"snapshot_id": "b" * 64},
        {"input_hash": "b" * 64},
    ],
)
def test_embedding_provenance_must_match(document, updates):
    vector = document.embeddings[0].model_copy(update=updates)
    with pytest.raises(ValueError, match="provenance or space"):
        validate_document(
            replace(document, embeddings=[vector, *document.embeddings[1:]])
        )


def test_complete_set_and_count_are_required(document):
    with pytest.raises(ValueError, match="complete fragment set"):
        validate_document(replace(document, embeddings=document.embeddings[1:]))
    with pytest.raises(ValueError, match="count"):
        validate_document(
            replace(
                document,
                manifest=document.manifest.model_copy(update={"fragment_count": 1}),
            )
        )


def test_conflicting_vectors_for_same_input_are_rejected(document):
    from atrag.ingestion.package_validation import digest

    first, second, *rest = document.fragments
    second = second.model_copy(
        update={
            "embedding_text": first.embedding_text,
            "embedding_input_hash": first.embedding_input_hash,
            "token_count": first.token_count,
        }
    )
    second = second.model_copy(
        update={
            "fragment_id": digest(
                second.model_dump_json(exclude={"fragment_id", "source"}).encode()
            )
        }
    )
    vector = document.embeddings[1].model_copy(
        update={
            "input_hash": first.embedding_input_hash,
            "fragment_id": second.fragment_id,
        }
    )
    with pytest.raises(ValueError, match="Conflicting package vectors"):
        validate_document(
            replace(
                document,
                fragments=[first, second, *rest],
                embeddings=[document.embeddings[0], vector, *document.embeddings[2:]],
            )
        )
