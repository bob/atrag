"""Citations are stable source slices, and notes share the extraction budget."""

from types import SimpleNamespace

import pytest
import tiktoken

from atrag.chat.sources import register_sources
from atrag.chat.tools import RagTools
from atrag.retrieval.parts import unit_text
from atrag.schemas.ingestion.structure import EditorialNote


def test_citations_deduplicate_only_identical_source_records():
    original = {
        "address": {"source_id": "one", "snapshot_id": "old", "block_id": "same"},
        "text_original": "Original",
        "source_start": 0,
    }
    registry = []
    first = register_sources(registry, [original])[0]
    assert first == {"citation_id": 1, **original}
    assert register_sources(registry, [original, original]) == [first, first]
    for change in (
        {"address": {**original["address"], "source_id": "two"}},
        {"address": {**original["address"], "snapshot_id": "new"}},
        {"source_start": 1},
        {"text_original": "Other"},
    ):
        new = register_sources(registry, [{**original, **change}])[0]
        assert new["citation_id"] == len(registry)
        assert new["citation_id"] > first["citation_id"]
    assert len(registry) == 5


def test_original_text_budget_includes_all_editorial_note_text():
    from atrag.schemas.ingestion.structure import LegalUnit

    unit = LegalUnit(
        legal_unit_id="unit",
        parent_id=None,
        instrument_id="test",
        code=None,
        document_scope="code",
        kind="article",
        role="norm",
        citation_label="Artigo 1",
        snapshot_id="a" * 64,
        page_start=1,
        page_end=1,
        printed_page_labels=[],
        locators=[],
        text_original="Texto original. " * 100,
        text_normalized="",
        table_ids=[],
        issues=[],
        editorial_notes=[
            EditorialNote(
                kind="amendment", raw_text="Redação transitória. " * 50, locators=[]
            )
        ],
    )
    # SourceDefinition is passed through; this pure check needs no storage or API.
    from pathlib import Path

    from atrag.ingestion.services.sources import load_source_registry

    source = load_source_registry(Path("data/sources/cirs.json"))
    result = unit_text(unit, source, {unit.legal_unit_id: unit}, context_tokens=500)
    encoding = tiktoken.get_encoding("cl100k_base")
    assert result.context_tokens_used <= 500
    assert (
        len(encoding.encode_ordinary(result.text_original))
        + len(encoding.encode_ordinary(unit.editorial_notes[0].raw_text))
        <= 500
    )
    assert not result.complete


@pytest.mark.parametrize(
    "start,end,location,locators",
    [
        (None, 1, None, []),
        (1, None, None, []),
        (None, 1, SimpleNamespace(kind="html"), []),
        (1, None, SimpleNamespace(kind="docx"), []),
        (None, None, None, []),
        (None, None, SimpleNamespace(kind="pdf"), []),
        (None, None, SimpleNamespace(kind="html"), [object()]),
    ],
)
def test_incomplete_or_unlocated_page_ranges_are_rejected(
    start, end, location, locators
):
    from atrag.ingestion.package_validation import validate_pages

    with pytest.raises(ValueError):
        validate_pages(
            SimpleNamespace(
                page_start=start, page_end=end, location=location, locators=locators
            )
        )


def test_non_pdf_page_validation_accepts_a_typed_location():
    from atrag.ingestion.package_validation import validate_pages

    validate_pages(
        SimpleNamespace(
            page_start=None,
            page_end=None,
            location=SimpleNamespace(kind="html"),
            locators=[],
        )
    )


@pytest.mark.parametrize(
    "name,arguments",
    [
        (
            "load_unit_text",
            {
                "address": {
                    "kind": "unit",
                    "source_id": "s",
                    "snapshot_id": "bad",
                    "block_id": "u",
                }
            },
        ),
        (
            "load_unit_text",
            {
                "address": {
                    "kind": "unit",
                    "source_id": "s",
                    "snapshot_id": "a" * 64,
                    "block_id": "u",
                },
                "start": -1,
            },
        ),
        (
            "load_unit_text",
            {"address": {"kind": "citation", "turn_id": "bad", "citation_id": 1}},
        ),
        (
            "load_neighbor_parts",
            {
                "address": {
                    "kind": "unit",
                    "source_id": "s",
                    "snapshot_id": "a" * 64,
                    "block_id": "u",
                },
                "after": -1,
            },
        ),
        (
            "load_context",
            {
                "address": {
                    "kind": "citation",
                    "turn_id": "00000000-0000-0000-0000-000000000001",
                    "citation_id": 0,
                }
            },
        ),
    ],
)
def test_reader_schemas_reject_bad_addresses_and_offsets_before_io(name, arguments):
    import json

    tools = RagTools(None, None, None, None).tools
    result = next(tool for tool in tools if tool.name == name).invoke(arguments)
    assert json.loads(result)["status"] == "invalid_arguments"
