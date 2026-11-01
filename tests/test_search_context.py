import json

import pytest
import tiktoken

from atrag.ingestion.services.packages import build_package, load_package
from atrag.retrieval.context import assemble_hits
from atrag.schemas.ingestion.structure import EditorialNote
from atrag.schemas.retrieval import Ranking, SearchHit, SearchRequest, SearchResult


@pytest.mark.parametrize("mode", ["python", "json", "json_string"])
def test_search_response_omits_pdf_coordinates(prepared_document, tmp_path, mode):
    data, parsed, chunked, embedded = prepared_document
    document = load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "package")
    )
    fragment = next(f for f in document.fragments if f.tables)
    original_table = fragment.tables[0]
    part = original_table.parts[0].model_copy(
        update={"cell_bboxes": [[(85, 290, 200, 330), None]]}
    )
    table = original_table.model_copy(update={"parts": [part]})
    note = EditorialNote(
        kind="amendment", raw_text="Redação de teste", locators=fragment.locators
    )
    fragment = fragment.model_copy(
        update={"tables": [table], "editorial_notes": [note]}
    )
    result = SearchResult(
        status="ok",
        snapshot_id=document.snapshot.snapshot_id,
        query_language="pt",
        hits=[
            SearchHit(
                fragment_id=fragment.fragment_id,
                score=1,
                fragment=fragment,
                related_units=[document.units[0]],
                related_tables=[table],
            )
        ],
        usage={"requests": 0},
    )
    if mode == "json_string":
        response = json.loads(result.model_dump_json())
    else:
        response = result.model_dump(mode=mode)
    serialized = json.dumps(response, ensure_ascii=False, default=str)
    assert '"bbox":' not in serialized
    assert '"cell_bboxes":' not in serialized
    hit = response["hits"][0]
    assert response["status"] == "ok"
    assert response["snapshot_id"] == document.snapshot.snapshot_id
    assert hit["score"] == 1
    assert hit["fragment"]["text_original"] == fragment.text_original
    assert hit["fragment"]["citation_label"] == fragment.citation_label
    assert hit["fragment"]["locators"][0]["page_number"] == part.page_number
    assert (
        hit["fragment"]["context"][0]["text_original"]
        == fragment.context[0].text_original
    )
    assert hit["fragment"]["editorial_notes"][0]["raw_text"] == note.raw_text
    assert hit["fragment"]["tables"][0]["rows"] == table.rows
    assert hit["related_tables"][0]["header_rows"] == table.header_rows
    assert hit["related_units"][0]["text_original"] == document.units[0].text_original
    assert fragment.locators[0].bbox
    assert fragment.context[0].locators[0].bbox
    assert part.model_dump()["cell_bboxes"] == [[(85, 290, 200, 330), None]]


def test_primary_evidence_survives_oversize_reference_context(
    prepared_document, tmp_path
):
    data, parsed, chunked, embedded = prepared_document
    document = load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "package")
    )
    fragment = document.fragments[0]
    units = {u.legal_unit_id: u for u in document.units}
    target = next(r.target_ids[0] for r in fragment.references if r.target_ids)
    units[target] = units[target].model_copy(
        update={"text_original": "Regra extensa. " * 50000}
    )
    request = SearchRequest(
        query="CIRS 16", snapshot_id=document.snapshot.snapshot_id, context_tokens=6000
    )
    hits, used, limited = assemble_hits(
        [Ranking(fragment_id=fragment.fragment_id, score=1)],
        {fragment.fragment_id: fragment},
        request,
        tiktoken.get_encoding("cl100k_base"),
    )
    assert len(hits) == 1
    assert hits[0].fragment == fragment
    assert target in hits[0].available_reference_ids
    assert not limited
    assert hits[0].related_units == []
    assert hits[0].related_tables == []
    assert 0 < used <= 6000


def test_search_keeps_references_available_without_loading_their_text(
    prepared_document, tmp_path
):
    data, parsed, chunked, embedded = prepared_document
    document = load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "package")
    )
    units = {u.legal_unit_id: u for u in document.units}
    target = next(
        u for u in units.values() if u.kind == "article" and u.article == "68-A"
    )
    root = target.model_copy(update={"parent_id": None})
    note = target.model_copy(
        update={
            "legal_unit_id": target.legal_unit_id + ":note:1",
            "parent_id": root.legal_unit_id,
            "kind": "note",
        }
    )
    units.update({root.legal_unit_id: root, note.legal_unit_id: note})
    request = SearchRequest(
        query="CIRS 16",
        snapshot_id=document.snapshot.snapshot_id,
        limit=1,
        context_tokens=12000,
    )
    rankings = [Ranking(fragment_id=f.fragment_id, score=1) for f in document.fragments]
    hits, used, limited = assemble_hits(
        rankings,
        {f.fragment_id: f for f in document.fragments},
        request,
        tiktoken.get_encoding("cl100k_base"),
    )
    assert len(hits) == 1
    assert not hits[0].related_tables
    assert not hits[0].related_units
    assert hits[0].available_reference_ids == [root.legal_unit_id]
    assert not limited
    assert used > 0


@pytest.mark.parametrize(
    "budget,count,used,limited",
    [(230, 2, 230, False), (229, 1, 88, True), (1, 0, 0, True)],
)
def test_primary_budget_includes_its_boundary(
    prepared_document, tmp_path, budget, count, used, limited
):
    data, parsed, chunked, embedded = prepared_document
    document = load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "package")
    )
    request = SearchRequest(
        query="taxa", snapshot_id=document.snapshot.snapshot_id, context_tokens=budget
    )
    rankings = [
        Ranking(fragment_id=f.fragment_id, score=1) for f in document.fragments[1:]
    ]
    hits, actual, is_limited = assemble_hits(
        rankings,
        {f.fragment_id: f for f in document.fragments},
        request,
        tiktoken.get_encoding("cl100k_base"),
    )
    assert len(hits) == count
    assert actual == used
    assert is_limited is limited


def test_unselected_references_do_not_consume_search_budget(
    prepared_document, tmp_path
):
    data, parsed, chunked, embedded = prepared_document
    document = load_package(
        build_package(parsed, chunked, embedded, data, tmp_path / "package")
    )
    fragment = document.fragments[0]
    request = SearchRequest(
        query="CIRS 16", snapshot_id=document.snapshot.snapshot_id, context_tokens=207
    )
    hits, used, limited = assemble_hits(
        [Ranking(fragment_id=fragment.fragment_id, score=1)],
        {fragment.fragment_id: fragment},
        request,
        tiktoken.get_encoding("cl100k_base"),
    )
    assert used == 109
    assert len(hits) == 1
    assert not limited
    assert hits[0].available_reference_ids == ["pt:cirs:code:article:68-A"]
    assert hits[0].related_units == []
