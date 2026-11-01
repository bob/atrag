"""Parts survive package import and support explicit text retrieval in PostgreSQL."""

import pytest
from fixtures.prepared import prepare_document

from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package
from atrag.migrations import migrate_database
from atrag.retrieval.search import search_tax_documents
from atrag.retrieval.text import load_context, load_neighbor_parts, load_unit_text
from atrag.schemas.retrieval import ContextRequest

pytestmark = pytest.mark.integration


def test_split_unit_search_order_original_and_neighbors(
    empty_corpus_schema, engine, tmp_path
):
    text = "2 - " + "Condição fiscal sem isenção. \n" * 500
    data, parsed, chunked, embedded = prepare_document(tmp_path, second_paragraph=text)
    package = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    migrate_database(engine)
    first = index_package(package, engine, tmp_path / "server")
    second = index_package(package, engine, tmp_path / "server")
    assert first.snapshot_id == second.snapshot_id
    result = search_tax_documents(
        "CIRS 16",
        query_language="pt",
        filters={"paragraph": "2"},
        snapshot_id=first.snapshot_id,
        engine=engine,
        limit=2,
    )
    assert len(result.hits) == 2
    assert [hit.fragment.part_index for hit in result.hits] == [0, 1]
    part = result.hits[0].fragment
    assert part.part_count > 2
    assert not result.unit_coverage[0].complete
    assert result.unit_coverage[0].returned_part_indices == [0, 1]
    original = load_unit_text(
        part.legal_unit_id,
        snapshot_id=first.snapshot_id,
        engine=engine,
        context_tokens=10000,
    )
    assert original.complete
    assert original.text_original == text
    neighbors = load_neighbor_parts(
        part.fragment_id, snapshot_id=first.snapshot_id, engine=engine, after=2
    )
    assert [f.part_index for f in neighbors.parts] == [0, 1, 2]
    assert not neighbors.complete
    assert not neighbors.omitted_part_indices
    assert neighbors.parts[0] == part
    expanded = load_context(
        ContextRequest(
            snapshot_id=first.snapshot_id,
            legal_unit_id="pt:cirs:code:article:16:paragraph:1",
            reference_ids=["pt:cirs:code:article:68-A"],
        ),
        engine=engine,
    )
    assert expanded.complete
    assert [u.legal_unit_id for u in expanded.units] == [
        "pt:cirs:code:article:16:paragraph:1",
        "pt:cirs:code:article:68-A",
    ]
    assert expanded.tables[0].rows == [["1000", "2,5"]]
    assert all(u.snapshot_id == first.snapshot_id for u in expanded.units)
    with pytest.raises(ValueError, match="Snapshot"):
        load_context(
            ContextRequest(snapshot_id="b" * 64, legal_unit_id=part.legal_unit_id),
            engine=engine,
        )
    defaults = load_neighbor_parts(
        result.hits[1].fragment.fragment_id,
        snapshot_id=first.snapshot_id,
        engine=engine,
    )
    assert [f.part_index for f in defaults.parts] == [1, 2]
    limited = load_neighbor_parts(
        part.fragment_id,
        snapshot_id=first.snapshot_id,
        engine=engine,
        after=2,
        context_tokens=1,
    )
    assert limited.parts == []
    assert limited.omitted_part_indices == [0, 1, 2]
    for loader, identity in [
        (load_unit_text, part.legal_unit_id),
        (load_neighbor_parts, part.fragment_id),
    ]:
        with pytest.raises(ValueError):
            loader(identity, snapshot_id="b" * 64, engine=engine)
        with pytest.raises(ValueError):
            loader("absent", snapshot_id=first.snapshot_id, engine=engine)
