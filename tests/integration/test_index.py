"""Corpus ingestion is atomic, idempotent and constrained in PostgreSQL."""

import json

import pytest
from sqlalchemy import event, select, text
from sqlalchemy.exc import IntegrityError

from atrag.db_models import FragmentRow, SnapshotRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package, load_package
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def corpus_schema(empty_corpus_schema, engine):
    migrate_database(engine)


@pytest.fixture
def package(prepared_document, tmp_path):
    data, parsed, chunked, embedded = prepared_document
    return build_package(parsed, chunked, embedded, data, tmp_path / "package")


def test_index_roundtrip_is_complete_idempotent_and_typed(package, engine, tmp_path):
    data = tmp_path / "server-data"
    first = index_package(package, engine, data)
    second = index_package(package, engine, data)
    document = load_package(package)
    assert first.directory != second.directory
    assert first.fragment_count == second.fragment_count == 3
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM source_snapshots")) == 1
        assert connection.scalar(text("SELECT count(*) FROM legal_units")) == len(
            document.units
        )
        assert connection.scalar(text("SELECT count(*) FROM search_fragments")) == 3
        assert connection.scalar(text("SELECT count(*) FROM embeddings")) == 3
        assert connection.scalar(text("SELECT count(*) FROM document_tables")) == 1
        assert connection.scalar(text("SELECT count(*) FROM fragment_units")) >= 3
        snapshot = connection.execute(select(SnapshotRow)).mappings().one()
        assert snapshot["language"] == "pt"
        assert snapshot["ingested_at"].utcoffset() is not None
        assert snapshot["revision_publication_date"].isoformat() == "2026-06-03"
        rows = connection.execute(select(FragmentRow)).mappings().all()
        assert {row["payload"]["fragment_id"] for row in rows} == {
            f.fragment_id for f in document.fragments
        }
        assert all(len(row["embedding"]) == 1536 for row in rows)
        assert (
            connection.scalar(
                text(
                    "SELECT count(*) FROM search_fragments WHERE search_vector @@ plainto_tsquery('portuguese', 'residentes')"
                )
            )
            == 2
        )
        assert (
            connection.scalar(
                text(
                    "SELECT count(*) FROM search_fragments WHERE search_vector @@ plainto_tsquery('portuguese', 'de e os')"
                )
            )
            == 0
        )
        indexes = connection.scalars(
            text("SELECT indexdef FROM pg_indexes WHERE tablename = 'search_fragments'")
        ).all()
        assert any("USING gin (search_vector)" in definition for definition in indexes)
        assert not any("hnsw" in definition.lower() for definition in indexes)
    assert (data / document.snapshot.local_path).read_bytes() == document.files[
        "source.pdf"
    ]
    report = json.loads((second.directory / "manifest.json").read_bytes())
    assert report["status"] == "succeeded"
    assert report["stage"] == "index"


def test_transaction_failure_preserves_previous_searchable_index(
    package, engine, tmp_path
):
    data = tmp_path / "server-data"
    index_package(package, engine, data)

    def reject_insert(connection, cursor, statement, parameters, context, many):
        if statement.startswith("INSERT INTO search_fragments"):
            raise RuntimeError("injected database failure")

    event.listen(engine, "before_cursor_execute", reject_insert)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            index_package(package, engine, data)
    finally:
        event.remove(engine, "before_cursor_execute", reject_insert)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM search_fragments")) == 3
        assert connection.scalar(text("SELECT count(*) FROM legal_units")) == len(
            load_package(package).units
        )
    assert any(
        json.loads(p.read_bytes())["status"] == "failed"
        for p in (data / "runs").glob("*/manifest.json")
    )


@pytest.mark.parametrize(
    "assignment",
    [
        "page_start = 0",
        "legal_status = 'active'",
        "embedding_dimensions = 10",
        "embedding_model = 'other'",
        "legal_unit_id = 'absent'",
    ],
)
def test_database_enforces_domain_constraints(package, engine, tmp_path, assignment):
    index_package(package, engine, tmp_path / "server-data")
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text(f"UPDATE search_fragments SET {assignment}"))


def test_package_is_checked_before_database_is_touched(package, engine, tmp_path):
    (package.parent / "fragments.jsonl").write_text("corrupt")
    with pytest.raises(ValueError, match="checksum"):
        index_package(package, engine, tmp_path / "server-data")
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM source_snapshots")) == 0


def test_reindex_can_change_model_without_mixing_spaces(
    package, prepared_document, engine, tmp_path
):
    from fixtures.prepared import refresh_manifest

    data, parsed, chunked, embedded = prepared_document
    index_package(package, engine, tmp_path / "server-data")
    path = embedded.parent / "embeddings.jsonl"
    path.write_text(
        path.read_text().replace("text-embedding-3-small", "text-embedding-3-large")
    )
    manifest = json.loads(embedded.read_bytes())
    manifest["model"] = "text-embedding-3-large"
    embedded.write_text(json.dumps(manifest))
    refresh_manifest(embedded)
    other = build_package(parsed, chunked, embedded, data, tmp_path / "large")
    index_package(other, engine, tmp_path / "server-data")
    with engine.connect() as connection:
        assert connection.scalars(
            text("SELECT DISTINCT embedding_model FROM search_fragments")
        ).all() == ["text-embedding-3-large"]
        assert connection.scalar(text("SELECT count(*) FROM embeddings")) == 6


def test_conflicting_cached_vector_cannot_change_previous_corpus(
    package, engine, tmp_path, prepared_document
):
    from fixtures.prepared import refresh_manifest

    data, parsed, chunked, embedded = prepared_document
    index_package(package, engine, tmp_path / "server")
    path = embedded.parent / "embeddings.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[0]["embedding"][0] = 2.0
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    refresh_manifest(embedded)
    changed = build_package(parsed, chunked, embedded, data, tmp_path / "changed")
    with pytest.raises(ValueError, match="persistent cache"):
        index_package(changed, engine, tmp_path / "server")
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM search_fragments")) == 3


def test_existing_snapshot_provenance_is_not_overwritten(package, engine, tmp_path):
    data = tmp_path / "server"
    result = index_package(package, engine, data)
    snapshot = data / "raw" / result.snapshot_id / "manifest.json"
    manifest = json.loads(snapshot.read_bytes())
    manifest["source"]["publisher"] = "Changed publisher"
    snapshot.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="provenance"):
        index_package(package, engine, data)


def test_rechunk_replaces_complete_fragment_set(
    package, prepared_document, engine, tmp_path
):
    from fixtures.prepared import digest, refresh_manifest

    from atrag.ingestion.services.chunks import chunk_parsed
    from atrag.schemas.ingestion.fragments import ChunkParameters

    data, parsed, _chunked, embedded = prepared_document
    original = load_package(package)
    index_package(package, engine, tmp_path / "server")
    chunk = chunk_parsed(
        parsed, data, original.snapshot.source, ChunkParameters(target_tokens=701)
    )
    fragments = [
        json.loads(line)
        for line in (chunk.directory / "fragments.jsonl").read_bytes().splitlines()
    ]
    vectors = {v.input_hash: v.model_dump(mode="json") for v in original.embeddings}
    rows = [
        {**vectors[f["embedding_input_hash"]], "fragment_id": f["fragment_id"]}
        for f in fragments
    ]
    (embedded.parent / "embeddings.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    manifest = json.loads(embedded.read_bytes())
    manifest["input_sha256"] = {
        name: digest((chunk.directory / name).read_bytes())
        for name in ("manifest.json", "fragments.jsonl", "report.json")
    }
    embedded.write_text(json.dumps(manifest))
    refresh_manifest(embedded)
    updated = build_package(
        parsed,
        chunk.directory / "manifest.json",
        embedded,
        data,
        tmp_path / "rechunked",
    )
    index_package(updated, engine, tmp_path / "server")
    with engine.connect() as connection:
        ids = set(connection.scalars(select(FragmentRow.fragment_id)))
    assert ids == {f["fragment_id"] for f in fragments}
    assert ids.isdisjoint(f.fragment_id for f in original.fragments)


def test_readers_keep_previous_index_and_writers_serialize(package, engine, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    data = tmp_path / "server"
    index_package(package, engine, data)
    deleting, release, second_started, second_deleted = (
        Event(),
        Event(),
        Event(),
        Event(),
    )

    def pause_writer(connection, cursor, statement, parameters, context, many):
        if not statement.startswith("DELETE FROM search_fragments"):
            return
        if deleting.is_set():
            second_deleted.set()
            return
        deleting.set()
        assert release.wait(5)

    def second_writer():
        second_started.set()
        return index_package(package, engine, data)

    event.listen(engine, "before_cursor_execute", pause_writer)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            first = workers.submit(index_package, package, engine, data)
            assert deleting.wait(5)
            second = workers.submit(second_writer)
            assert second_started.wait(5)
            assert not second_deleted.wait(0.1)
            with engine.connect() as connection:
                assert (
                    connection.scalar(text("SELECT count(*) FROM search_fragments"))
                    == 3
                )
            release.set()
            assert first.result(timeout=5).fragment_count == 3
            assert second.result(timeout=5).fragment_count == 3
    finally:
        release.set()
        event.remove(engine, "before_cursor_execute", pause_writer)
