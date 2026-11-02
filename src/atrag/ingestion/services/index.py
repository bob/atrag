"""Install each immutable prepared version once in a PostgreSQL transaction."""

from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from atrag.db_models import (
    AddressColumns,
    DocumentTableRow,
    FragmentRow,
    FragmentUnitRow,
    LegalUnitRow,
    SnapshotRow,
    SourceDocumentRow,
)
from atrag.ingestion.package_validation import digest
from atrag.ingestion.services.embedding_cache import EMBEDDINGS
from atrag.ingestion.services.packages import load_package
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import _publish_snapshot, read_snapshot
from atrag.schemas.ingestion.embeddings import EmbeddingRecord
from atrag.schemas.ingestion.packages import IndexResult
from atrag.schemas.ingestion.snapshots import SourceSnapshot


def _install_snapshot(document, data_dir):
    raw = data_dir / "raw"
    raw.mkdir(exist_ok=True)
    original = SourceSnapshot.model_validate_json(document.files["snapshot.json"])
    destination = raw / original.snapshot_id
    with TemporaryDirectory(prefix=".import-", dir=raw) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        (staging / "source.pdf").write_bytes(document.files["source.pdf"])
        (staging / "manifest.json").write_bytes(document.files["snapshot.json"])
        _publish_snapshot(staging, destination)
    read_snapshot(destination, original.snapshot_id, original.size_bytes)
    # The first raw manifest is evidence of an import, not the version's metadata.
    # Each prepared package carries its own immutable provenance in PostgreSQL.


def _snapshot_values(document):
    snapshot, manifest = document.snapshot, document.manifest
    values = snapshot.source.model_dump(exclude={"expected_sha256"})
    values.update(
        snapshot_id=snapshot.snapshot_id,
        source_sha256=snapshot.source_sha256,
        preparation_sha256=snapshot.preparation_sha256,
        local_path=str(snapshot.local_path),
        size_bytes=snapshot.size_bytes,
        ingested_at=snapshot.ingested_at,
        indexed_at=datetime.now(UTC),
        embedding_model=manifest.model,
        embedding_dimensions=manifest.dimensions,
        parser_version=manifest.parser_version,
        chunker_version=manifest.chunker_version,
        review_status=document.reports["chunk"]["review_status"],
        reports=document.reports,
    )
    url = snapshot.source.source_url
    values["source_url"] = None if url is None else str(url)
    return values


def _address_values(unit):
    values = {
        key: getattr(unit, key)
        for key in AddressColumns.__annotations__
        if key != "payload"
    }
    return {**values, "payload": unit.model_dump(mode="json")}


def _cache_vectors(connection, document):
    records = {
        vector.input_hash: EmbeddingRecord.model_validate(
            vector.model_dump(include=EmbeddingRecord.model_fields.keys())
        )
        for vector in document.embeddings
    }
    connection.execute(
        insert(EMBEDDINGS).on_conflict_do_nothing(),
        [r.model_dump() for r in records.values()],
    )
    query = select(EMBEDDINGS).where(
        EMBEDDINGS.c.model == document.manifest.model,
        EMBEDDINGS.c.dimensions == document.manifest.dimensions,
        EMBEDDINGS.c.input_hash.in_(records),
    )
    for row in connection.execute(query).mappings():
        record = EmbeddingRecord.model_validate(dict(row))
        if records[record.input_hash] != record:
            raise ValueError("Package vector conflicts with the persistent cache")


def _insert_fragments(connection, document):
    vectors = {v.fragment_id: v for v in document.embeddings}
    rows, links = [], []
    for fragment in document.fragments:
        vector = vectors[fragment.fragment_id]
        rows.append(
            {
                **_address_values(fragment),
                "fragment_id": fragment.fragment_id,
                "document_title": fragment.source.document_title,
                "review_status": fragment.review_status,
                "embedding_model": vector.model,
                "embedding_dimensions": vector.dimensions,
                "embedding_input_hash": vector.input_hash,
                "embedding": vector.embedding,
            }
        )
        for relationship, ids in (
            ("content", fragment.legal_unit_ids),
            ("context", fragment.context_unit_ids),
        ):
            links.extend(
                {
                    "snapshot_id": fragment.snapshot_id,
                    "fragment_id": fragment.fragment_id,
                    "legal_unit_id": identity,
                    "relationship": relationship,
                }
                for identity in set(ids)
            )
    connection.execute(insert(FragmentRow), rows)
    connection.execute(insert(FragmentUnitRow), links)


def _insert_corpus(connection, document):
    snapshot_id = document.snapshot.snapshot_id
    key = int.from_bytes(
        bytes.fromhex(digest(f"atrag:corpus:{snapshot_id}".encode()))[:8], signed=True
    )
    connection.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": key})
    if connection.scalar(
        select(SnapshotRow.snapshot_id).where(SnapshotRow.snapshot_id == snapshot_id)
    ):
        return
    _cache_vectors(connection, document)
    connection.execute(
        insert(SourceDocumentRow)
        .values(source_id=document.snapshot.source.source_id)
        .on_conflict_do_nothing()
    )
    values = _snapshot_values(document)
    connection.execute(insert(SnapshotRow).values(**values))
    connection.execute(
        insert(LegalUnitRow), [_address_values(u) for u in document.units]
    )
    if document.tables:
        connection.execute(
            insert(DocumentTableRow),
            [
                {
                    "snapshot_id": snapshot_id,
                    "table_id": t.table_id,
                    "payload": t.model_dump(mode="json"),
                }
                for t in document.tables
            ],
        )
    _insert_fragments(connection, document)


def index_package(package: Path, engine, data_dir: Path) -> IndexResult:
    with record_run(data_dir, stage="index") as run:
        document = load_package(package)
        run.manifest.update(
            snapshot_id=document.snapshot.snapshot_id,
            fragment_count=len(document.fragments),
            package_sha256=digest(package.read_bytes()),
        )
        _install_snapshot(document, data_dir)
        with engine.begin() as connection:
            _insert_corpus(connection, document)
    return IndexResult(
        document.snapshot.snapshot_id, len(document.fragments), run.directory
    )
