"""The FAQ schema upgrade preserves CIRS and accepts located explanatory units."""

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import insert, select, text
from sqlalchemy.exc import IntegrityError

from atrag.db_models import FragmentRow, LegalUnitRow, SnapshotRow, SourceDocumentRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


def migration(engine, operation, revision):
    config = Config()
    config.set_main_option(
        "script_location", str(Path("src/atrag/migrations").resolve())
    )
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        operation(config, revision)


@pytest.fixture
def legacy_corpus(empty_corpus_schema, engine, prepared_document, tmp_path):
    migration(engine, command.upgrade, "0009_english_default")
    data, parsed, chunked, embedded = prepared_document
    path = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    indexed = index_package(path, engine, tmp_path / "server")
    return indexed.snapshot_id


def test_upgrade_keeps_cirs_bytes_and_default_format(engine, legacy_corpus):
    with engine.connect() as connection:
        before = {
            table: connection.execute(
                text(f"SELECT * FROM {table} ORDER BY 1, 2")
            ).all()
            for table in ("legal_units", "search_fragments")
        }
    migrate_database(engine)
    migrate_database(engine)
    with engine.connect() as connection:
        for table, rows in before.items():
            assert (
                connection.execute(text(f"SELECT * FROM {table} ORDER BY 1, 2")).all()
                == rows
            )
        snapshot = connection.execute(select(SnapshotRow)).mappings().one()
        assert snapshot["source_format"] == "pdf"
        assert snapshot["snapshot_id"] == legacy_corpus
    migration(engine, command.downgrade, "0009_english_default")
    for table in ("legal_units", "search_fragments"):
        for assignment in (
            "document_scope = 'faq'",
            "kind = 'topic'",
            "role = 'at_explanation'",
        ):
            with pytest.raises(IntegrityError), engine.begin() as connection:
                connection.execute(text(f"UPDATE {table} SET {assignment}"))
    migrate_database(engine)
    with engine.connect() as connection:
        assert connection.scalar(select(SnapshotRow.source_format)) == "pdf"


@pytest.fixture
def faq_rows(engine, legacy_corpus):
    migrate_database(engine)
    with engine.begin() as connection:
        snapshot = dict(connection.execute(select(SnapshotRow)).mappings().one())
        snapshot.update(
            snapshot_id="f" * 64,
            source_sha256="f" * 64,
            preparation_sha256=None,
            source_id="pt-at-faq",
            source_format="html_collection",
            document_type="faq",
        )
        connection.execute(insert(SourceDocumentRow).values(source_id="pt-at-faq"))
        connection.execute(insert(SnapshotRow).values(**snapshot))
        for model in (LegalUnitRow, FragmentRow):
            row = dict(connection.execute(select(model).limit(1)).mappings().one())
            row.update(
                snapshot_id=snapshot["snapshot_id"],
                legal_unit_id="pt-at-faq:faqs-00267:3286",
                parent_id=None,
                instrument_id="pt-at-faq",
                code=None,
                document_scope="faq",
                kind="faq_entry",
                role="at_explanation",
                chapter=None,
                section=None,
                article=None,
                paragraph=None,
                subparagraph=None,
                page_start=None,
                page_end=None,
                payload={
                    "location": {
                        "kind": "html",
                        "page_url": "https://example.org/faq",
                        "anchor": "heading20",
                    }
                },
            )
            connection.execute(insert(model).values(**row))
    return snapshot["snapshot_id"]


def test_faq_storage_roundtrips_without_pdf_address(engine, faq_rows):
    with engine.connect() as connection:
        assert (
            connection.scalar(
                select(SnapshotRow.source_format).where(
                    SnapshotRow.snapshot_id == faq_rows
                )
            )
            == "html_collection"
        )
        for model in (LegalUnitRow, FragmentRow):
            row = (
                connection.execute(select(model).where(model.snapshot_id == faq_rows))
                .mappings()
                .one()
            )
            assert row["document_scope"] == "faq"
            assert row["kind"] == "faq_entry"
            assert row["role"] == "at_explanation"
            assert (
                row["page_start"]
                is row["page_end"]
                is row["code"]
                is row["article"]
                is None
            )
            assert row["payload"]["location"]["page_url"] == "https://example.org/faq"


def test_downgrade_refuses_to_discard_html_provenance(engine, faq_rows):
    with pytest.raises(ValueError, match="HTML"):
        migration(engine, command.downgrade, "0009_english_default")
    with engine.connect() as connection:
        assert (
            connection.scalar(
                select(SnapshotRow.snapshot_id).where(
                    SnapshotRow.snapshot_id == faq_rows
                )
            )
            == faq_rows
        )
        assert (
            connection.scalar(text("SELECT version_num FROM alembic_version"))
            == "0010_faq_contracts"
        )


@pytest.mark.parametrize(
    "table, assignment",
    [
        ("source_snapshots", "source_format = 'html'"),
        ("source_snapshots", "source_format = NULL"),
        ("legal_units", "document_scope = 'other'"),
        ("legal_units", "kind = 'other'"),
        ("legal_units", "role = 'other'"),
        ("search_fragments", "document_scope = 'other'"),
        ("search_fragments", "kind = 'other'"),
        ("search_fragments", "role = 'other'"),
    ],
)
def test_upgrade_retains_domain_constraints(engine, legacy_corpus, table, assignment):
    migrate_database(engine)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(text(f"UPDATE {table} SET {assignment}"))
