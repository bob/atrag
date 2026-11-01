import pytest
import tiktoken

from atrag.ingestion.services.packages import build_package, load_package
from atrag.retrieval.context import assemble_hits
from atrag.schemas.retrieval import Ranking, SearchRequest


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
