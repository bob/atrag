"""Explicit evidence selection preserves source text, budgets and reviewed links."""

import json
from pathlib import Path

import pytest

from atrag.ingestion.legal_references import resolve_references
from atrag.ingestion.parse_cirs import parse_cirs
from atrag.ingestion.services.sources import load_source_registry
from atrag.retrieval.expansion import expand_context
from atrag.schemas.ingestion.extraction import ExtractedPage, ExtractedTable
from atrag.schemas.ingestion.structure import LegalUnit
from atrag.schemas.retrieval import ContextRequest

SOURCE = load_source_registry(Path("data/sources/cirs.json"))
PREFIX = "pt:cirs:code:article:16"


@pytest.fixture
def units():
    original = [
        LegalUnit.model_validate(u)
        for u in json.loads(
            Path("tests/fixtures/cirs/article16-units.json").read_bytes()
        )
    ]
    texts = {u.legal_unit_id: "" for u in original}
    texts[f"{PREFIX}:paragraph:4"] = "salvo nos casos previstos nos n.os 14 e 16"
    texts[f"{PREFIX}:paragraph:14"] = "Sem prejuízo do disposto no número seguinte"
    texts[f"{PREFIX}:paragraph:15"] = "O disposto no número anterior não é aplicável"
    return {u.legal_unit_id: u for u in resolve_references(original, texts)}


def request(units, number=4, **kwargs):
    return ContextRequest(
        snapshot_id=units[PREFIX].snapshot_id,
        legal_unit_id=f"{PREFIX}:paragraph:{number}",
        **kwargs,
    )


def test_reference_selection_does_not_follow_other_mentions(units):
    result = expand_context(
        request(units, reference_ids=[f"{PREFIX}:paragraph:14"]), SOURCE, units, {}
    )
    assert [u.legal_unit_id for u in result.units] == [
        f"{PREFIX}:paragraph:4",
        f"{PREFIX}:paragraph:14",
    ]
    assert result.complete
    assert result.units[1].available_reference_ids == [f"{PREFIX}:paragraph:15"]
    assert all(u.source == SOURCE and u.locators and u.complete for u in result.units)


def test_reviewed_exception_dependencies_terminate_cycles_and_deduplicate(units):
    four, fourteen, fifteen, sixteen = [
        f"{PREFIX}:paragraph:{n}" for n in (4, 14, 15, 16)
    ]
    result = expand_context(
        request(
            units,
            dependency_edges=[
                (four, fourteen),
                (four, sixteen),
                (fourteen, fifteen),
                (fifteen, fourteen),
            ],
        ),
        SOURCE,
        units,
        {},
    )
    assert [u.legal_unit_id for u in result.units] == [four, fourteen, sixteen, fifteen]
    assert result.complete
    assert not result.omitted_unit_ids
    assert not result.omitted_table_ids
    assert result.context_tokens_used <= 6000
    for text in result.units:
        assert text.text_original == units[text.legal_unit_id].text_original


def test_note_is_available_but_only_loaded_when_selected(units):
    note = next(u for u in units.values() if u.kind == "note")
    ordinary = expand_context(request(units, 8), SOURCE, units, {})
    assert len(ordinary.units) == 1
    assert note.legal_unit_id in ordinary.units[0].available_note_ids
    selected = expand_context(
        request(units, 8, note_ids=[note.legal_unit_id]), SOURCE, units, {}
    )
    assert selected.units[-1].role == "transitional"
    assert selected.units[-1].text_original == note.text_original
    assert selected.units[0].legal_status == "repealed"


def test_budget_preserves_root_and_reports_partial_ranges(units):
    selected = request(
        units, reference_ids=[f"{PREFIX}:paragraph:14"], context_tokens=150
    )
    result = expand_context(selected, SOURCE, units, {})
    assert result.units[0].complete
    assert result.units[0].legal_unit_id == selected.legal_unit_id
    assert not result.complete
    assert result.context_tokens_used <= 150
    if len(result.units) == 2:
        part = result.units[-1]
        assert not part.complete
        assert part.source_start == 0
        assert part.next_start == part.source_end < part.source_length
    else:
        assert result.omitted_unit_ids == selected.reference_ids
    partial = expand_context(request(units, 14, context_tokens=100), SOURCE, units, {})
    assert not partial.complete
    assert partial.units[0].next_start


@pytest.mark.parametrize(
    "values",
    [
        {"reference_ids": [f"{PREFIX}:paragraph:1"]},
        {"note_ids": [f"{PREFIX}:paragraph:14"]},
        {"dependency_edges": [(f"{PREFIX}:paragraph:4", f"{PREFIX}:paragraph:1")]},
        {"dependency_edges": [(f"{PREFIX}:paragraph:14", f"{PREFIX}:paragraph:15")]},
        {"snapshot_id": "b" * 64},
        {"legal_unit_id": "absent"},
        {"dependency_edges": [(f"{PREFIX}:paragraph:4", "absent")]},
    ],
)
def test_invalid_or_unconnected_selection_fails_at_boundary(units, values):
    selected = request(units).model_copy(update=values)
    with pytest.raises(ValueError):
        expand_context(selected, SOURCE, units, {})


def test_selected_article_and_both_tables_fit_without_chapter_text():
    fixtures = Path("tests/fixtures/cirs")
    pages = [
        ExtractedPage.model_validate(p)
        for p in json.loads((fixtures / "structure-pages.json").read_bytes())
    ]
    tables = [
        ExtractedTable.model_validate(t)
        for t in json.loads((fixtures / "structure-tables.json").read_bytes())["tables"]
    ]
    structure = parse_cirs(pages, tables)
    units = {u.legal_unit_id: u for u in structure.units}
    chapter = next(
        u for u in units.values() if u.kind == "chapter" and u.chapter == "III"
    )
    units[chapter.legal_unit_id] = chapter.model_copy(
        update={"text_original": "Texto de outros artigos da mesma divisão. " * 5000}
    )
    selected = ContextRequest(
        snapshot_id=structure.units[0].snapshot_id,
        legal_unit_id="pt:cirs:code:article:68-A:paragraph:1",
        reference_ids=["pt:cirs:code:article:68"],
    )
    result = expand_context(selected, SOURCE, units, {t.table_id: t for t in tables})
    assert result.complete
    assert len(result.units) == 2
    assert {t.table_id for t in result.tables} == {"p81-table1", "p82-image2-table"}
    assert result.context_tokens_used < 6000
    assert all("outros artigos" not in t.text_original for t in result.units)
    assert result.tables[0].rows[0] == ["De mais de 80 000 até 250 000", "2,5"]
    assert len(result.tables[1].rows) == 9
    # Incomplete text never counts as a complete table or explanation.
    limited = expand_context(
        selected.model_copy(update={"context_tokens": 250}),
        SOURCE,
        units,
        {t.table_id: t for t in tables},
    )
    assert not limited.complete
    assert limited.omitted_table_ids


def test_expansion_assessment_rejects_partial_text_and_wrong_source(units):
    from types import SimpleNamespace

    from atrag.retrieval.evaluation import assess_expansion
    from atrag.schemas.evaluation import ExpansionCase

    selected = request(units, reference_ids=[f"{PREFIX}:paragraph:14"])
    result = expand_context(selected, SOURCE, units, {})
    case = ExpansionCase(
        case_id="exception",
        language="ru",
        kind="explanation",
        request=selected,
        required_unit_ids=[u.legal_unit_id for u in result.units],
    )
    document = SimpleNamespace(
        units=list(units.values()),
        tables=[],
        snapshot=SimpleNamespace(snapshot_id=selected.snapshot_id, source=SOURCE),
    )
    assert assess_expansion(case, result, document)["passed"]
    partial = expand_context(
        selected.model_copy(update={"context_tokens": 100}), SOURCE, units, {}
    )
    partial_assessment = assess_expansion(case, partial, document)
    assert not partial_assessment["passed"]
    assert partial_assessment["citations_correct"]
    from pydantic import HttpUrl

    result.units[0].source = SOURCE.model_copy(
        update={"source_url": HttpUrl("https://invalid.test/source.pdf")}
    )
    assessed = assess_expansion(case, result, document)
    assert not assessed["citations_correct"]
    assert not assessed["passed"]


@pytest.mark.parametrize(
    "updates",
    [
        {"legal_unit_id": "absent"},
        {"snapshot_id": "b" * 64},
        {"citation_label": "CIVA, artigo 16"},
        {"locators": []},
        {"source_length": 1},
        {"next_start": 0},
        {"complete": False},
        {
            "source_start": 0,
            "source_end": 0,
            "text_original": "",
            "complete": False,
            "next_start": 0,
        },
    ],
)
def test_expansion_citations_require_actual_ranges_and_source_metadata(units, updates):
    from types import SimpleNamespace

    from atrag.retrieval.evaluation import assess_expansion
    from atrag.schemas.evaluation import ExpansionCase

    selected = request(units)
    result = expand_context(selected, SOURCE, units, {})
    case = ExpansionCase(
        case_id="original",
        language="pt",
        kind="original",
        request=selected,
        required_unit_ids=[selected.legal_unit_id],
    )
    document = SimpleNamespace(
        units=list(units.values()),
        tables=[],
        snapshot=SimpleNamespace(snapshot_id=selected.snapshot_id, source=SOURCE),
    )
    result.units[0] = result.units[0].model_copy(update=updates)
    assessed = assess_expansion(case, result, document)
    assert not assessed["citations_correct"]
    assert not assessed["passed"]
    if updates.get("complete") is False:
        assert assessed["missing_unit_ids"] == [selected.legal_unit_id]
    result = expand_context(selected, SOURCE, units, {})
    result.legal_unit_id = "absent"
    assert not assess_expansion(case, result, document)["citations_correct"]


def test_minimum_text_budget_includes_a_complete_multibyte_character(units):
    import tiktoken

    selected = request(units)
    root = units[selected.legal_unit_id].model_copy(update={"text_original": "ΩA"})
    units[root.legal_unit_id] = root
    encoding = tiktoken.get_encoding("cl100k_base")
    minimum = len(encoding.encode_ordinary(root.citation_label + "\n\nΩ"))
    exact = expand_context(
        selected.model_copy(update={"context_tokens": minimum}), SOURCE, units, {}
    )
    assert exact.units[0].text_original == "Ω"
    assert exact.units[0].next_start == 1
    assert exact.context_tokens_used == minimum
    missing = expand_context(
        selected.model_copy(update={"context_tokens": minimum - 1}), SOURCE, units, {}
    )
    assert not missing.units
    assert missing.omitted_unit_ids == [root.legal_unit_id]


def test_reverse_reviewed_reference_and_tiny_budget_are_explicit(units):
    sixteen, four = [f"{PREFIX}:paragraph:{n}" for n in (16, 4)]
    result = expand_context(
        request(units, 16, dependency_edges=[(sixteen, four)]), SOURCE, units, {}
    )
    assert [u.legal_unit_id for u in result.units] == [sixteen, four]
    missing = expand_context(request(units, context_tokens=1), SOURCE, units, {})
    assert missing.units == []
    assert missing.omitted_unit_ids == [four]
    assert not missing.complete
    assert missing.context_tokens_used == 0


def test_shared_table_is_budgeted_once_and_unreviewed_table_stays_omitted(
    prepared_document, tmp_path
):
    import tiktoken

    from atrag.ingestion.services.packages import build_package, load_package
    from atrag.retrieval.expansion import table_cost
    from atrag.schemas.ingestion.structure import LegalReference

    data, parsed, chunked, embedded = prepared_document
    document = load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "pkg")
    )
    units = {u.legal_unit_id: u for u in document.units}
    root = units["pt:cirs:code:article:68-A"]
    child = units[root.legal_unit_id + ":paragraph:1"]
    # A source can explicitly refer to its own numbered rule and shared table.
    units[root.legal_unit_id] = root.model_copy(
        update={
            "references": [
                LegalReference(
                    raw_text="n.º 1",
                    kind="paragraph",
                    status="resolved",
                    target_ids=[child.legal_unit_id],
                )
            ]
        }
    )
    selected = ContextRequest(
        snapshot_id=root.snapshot_id,
        legal_unit_id=root.legal_unit_id,
        reference_ids=[child.legal_unit_id, child.legal_unit_id],
    )
    tables = {t.table_id: t for t in document.tables}
    tables["rates"] = tables["rates"].model_copy(
        update={"header_rows": [["Rendimento (€)", "Taxa"]]}
    )
    result = expand_context(selected, document.snapshot.source, units, tables)
    assert result.complete
    assert len(result.units) == 2
    assert len(result.tables) == 1
    encoding = tiktoken.get_encoding("cl100k_base")
    cost = len(
        encoding.encode_ordinary(
            json.dumps(
                {"headers": [["Rendimento (€)", "Taxa"]], "rows": [["1000", "2,5"]]},
                ensure_ascii=False,
            )
        )
    )
    assert table_cost(result.tables[0], encoding) == cost
    assert (
        result.context_tokens_used
        == sum(u.context_tokens_used for u in result.units) + cost
    )
    exact = expand_context(
        selected.model_copy(update={"context_tokens": result.context_tokens_used}),
        document.snapshot.source,
        units,
        tables,
    )
    assert exact.complete
    other = tables["rates"].model_copy(update={"table_id": "second-rates"})
    tables[other.table_id] = other
    units[root.legal_unit_id] = units[root.legal_unit_id].model_copy(
        update={"table_ids": ["rates", "second-rates"]}
    )
    budget = result.units[0].context_tokens_used + 2 * cost
    two = selected.model_copy(update={"reference_ids": [], "context_tokens": budget})
    assert expand_context(two, document.snapshot.source, units, tables).complete
    limited = expand_context(
        two.model_copy(update={"context_tokens": budget - 1}),
        document.snapshot.source,
        units,
        tables,
    )
    assert not limited.complete
    assert len(limited.tables) == 1
    assert limited.omitted_table_ids == ["second-rates"]
    assert limited.context_tokens_used <= budget - 1
    units[root.legal_unit_id] = units[root.legal_unit_id].model_copy(
        update={"table_ids": ["rates"]}
    )
    tables["rates"] = tables["rates"].model_copy(
        update={"review_status": "needs_review"}
    )
    unverified = expand_context(selected, document.snapshot.source, units, tables)
    assert not unverified.complete
    assert unverified.tables == []
    assert unverified.omitted_table_ids == ["rates"]
    tiny = expand_context(
        selected.model_copy(update={"context_tokens": 1}),
        document.snapshot.source,
        units,
        tables,
    )
    assert tiny.omitted_table_ids == ["rates"]
