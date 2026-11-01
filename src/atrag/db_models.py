"""Relational corpus addresses with JSONB evidence and a single vector space."""

from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from atrag.ingestion.services.embedding_cache import EMBEDDINGS


class Base(DeclarativeBase):
    pass


EMBEDDINGS.to_metadata(Base.metadata)


class SnapshotRow(Base):
    __tablename__ = "source_snapshots"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "embedding_model", "embedding_dimensions"),
    )

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(Text, index=True)
    source_sha256: Mapped[str] = mapped_column(String(64))
    document_title: Mapped[str] = mapped_column(Text)
    document_type: Mapped[str] = mapped_column(Text)
    publisher: Mapped[str] = mapped_column(Text)
    language: Mapped[str] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    source_url_status: Mapped[str] = mapped_column(Text)
    local_path: Mapped[str] = mapped_column(Text)
    size_bytes: Mapped[int]
    consolidation_label: Mapped[str | None] = mapped_column(Text)
    publication_date: Mapped[date | None]
    revision_publication_date: Mapped[date | None]
    downloaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    indexed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    embedding_model: Mapped[str] = mapped_column(Text)
    embedding_dimensions: Mapped[int]
    parser_version: Mapped[str] = mapped_column(Text)
    chunker_version: Mapped[str] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(Text)
    reports: Mapped[dict] = mapped_column(JSONB)


class AddressColumns:
    snapshot_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("source_snapshots.snapshot_id"), primary_key=True
    )
    legal_unit_id: Mapped[str] = mapped_column(Text)
    parent_id: Mapped[str | None] = mapped_column(Text)
    instrument_id: Mapped[str] = mapped_column(Text)
    code: Mapped[str | None] = mapped_column(Text)
    document_scope: Mapped[str] = mapped_column(Text)
    kind: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(Text)
    chapter: Mapped[str | None] = mapped_column(Text)
    section: Mapped[str | None] = mapped_column(Text)
    article: Mapped[str | None] = mapped_column(Text)
    paragraph: Mapped[str | None] = mapped_column(Text)
    subparagraph: Mapped[str | None] = mapped_column(Text)
    legal_status: Mapped[str] = mapped_column(Text)
    unit_title: Mapped[str] = mapped_column(Text)
    citation_label: Mapped[str] = mapped_column(Text)
    page_start: Mapped[int]
    page_end: Mapped[int]
    text_original: Mapped[str] = mapped_column(Text)
    text_normalized: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSONB)


class LegalUnitRow(AddressColumns, Base):
    __tablename__ = "legal_units"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "parent_id"],
            ["legal_units.snapshot_id", "legal_units.legal_unit_id"],
            deferrable=True,
            initially="DEFERRED",
        ),
        Index(
            "ix_legal_units_address", "snapshot_id", "code", "document_scope", "article"
        ),
    )
    legal_unit_id: Mapped[str] = mapped_column(Text, primary_key=True)


class DocumentTableRow(Base):
    __tablename__ = "document_tables"
    snapshot_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("source_snapshots.snapshot_id"), primary_key=True
    )
    table_id: Mapped[str] = mapped_column(Text, primary_key=True)
    payload: Mapped[dict] = mapped_column(JSONB)


class FragmentRow(AddressColumns, Base):
    __tablename__ = "search_fragments"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "legal_unit_id"],
            ["legal_units.snapshot_id", "legal_units.legal_unit_id"],
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "embedding_model", "embedding_dimensions"],
            [
                "source_snapshots.snapshot_id",
                "source_snapshots.embedding_model",
                "source_snapshots.embedding_dimensions",
            ],
        ),
        ForeignKeyConstraint(
            ["embedding_model", "embedding_dimensions", "embedding_input_hash"],
            ["embeddings.model", "embeddings.dimensions", "embeddings.input_hash"],
        ),
        Index(
            "ix_search_fragments_address",
            "snapshot_id",
            "code",
            "document_scope",
            "article",
            "legal_status",
        ),
    )
    fragment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_title: Mapped[str] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(Text)
    embedding_model: Mapped[str] = mapped_column(Text)
    embedding_dimensions: Mapped[int]
    embedding_input_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]] = mapped_column(Vector(1536))


class FragmentUnitRow(Base):
    __tablename__ = "fragment_units"
    __table_args__ = (
        ForeignKeyConstraint(
            ["snapshot_id", "fragment_id"],
            ["search_fragments.snapshot_id", "search_fragments.fragment_id"],
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["snapshot_id", "legal_unit_id"],
            ["legal_units.snapshot_id", "legal_units.legal_unit_id"],
        ),
    )
    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    fragment_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    legal_unit_id: Mapped[str] = mapped_column(Text, primary_key=True)
    relationship: Mapped[str] = mapped_column(Text, primary_key=True)
