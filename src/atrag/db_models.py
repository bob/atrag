"""Relational corpus addresses with JSONB evidence and a single vector space."""

from datetime import date, datetime
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from atrag.ingestion.services.embedding_cache import EMBEDDINGS
from atrag.schemas.settings import SupportedLanguage


class Base(DeclarativeBase):
    pass


EMBEDDINGS.to_metadata(Base.metadata)


class ChatUserRow(Base):
    __tablename__ = "chat_users"
    __table_args__ = (UniqueConstraint("issuer", "subject"),)

    id: Mapped[UUID] = mapped_column(primary_key=True)
    issuer: Mapped[str] = mapped_column(Text)
    subject: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text)
    language: Mapped[SupportedLanguage] = mapped_column(Text, server_default="en")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChatConversationRow(Base):
    __tablename__ = "chat_conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("chat_users.id"), index=True)
    title: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ChatTurnRow(Base):
    __tablename__ = "chat_turns"

    # Assigned once by the caller, including retries and resume.
    id: Mapped[UUID] = mapped_column(primary_key=True)
    conversation_id: Mapped[UUID] = mapped_column(
        ForeignKey("chat_conversations.id"), index=True
    )
    # Only provenance migrated from snapshot-bound conversations, not corpus selection.
    legacy_snapshot_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("source_snapshots.snapshot_id")
    )
    corpus_release_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("corpus_releases.release_id")
    )
    retrieval_tokens: Mapped[int] = mapped_column(server_default="0")
    language: Mapped[SupportedLanguage] = mapped_column(Text)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, server_default="")
    sources: Mapped[list[dict]] = mapped_column(
        JSONB, server_default=text("'[]'::jsonb")
    )
    status: Mapped[str] = mapped_column(Text, server_default="pending")
    model_usage: Mapped[dict] = mapped_column(JSONB, server_default=text("'{}'::jsonb"))
    embedding_usage: Mapped[dict] = mapped_column(
        JSONB, server_default=text("'{}'::jsonb")
    )
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class SnapshotRow(Base):
    __tablename__ = "source_snapshots"
    __table_args__ = (
        UniqueConstraint("snapshot_id", "embedding_model", "embedding_dimensions"),
        UniqueConstraint("source_id", "snapshot_id", name="snapshot_source_identity"),
    )

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_id: Mapped[str] = mapped_column(
        Text, ForeignKey("source_documents.source_id"), index=True
    )
    source_sha256: Mapped[str] = mapped_column(String(64))
    preparation_sha256: Mapped[str | None] = mapped_column(String(64))
    document_title: Mapped[str] = mapped_column(Text)
    document_type: Mapped[str] = mapped_column(Text)
    source_format: Mapped[str] = mapped_column(Text, server_default="pdf")
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


class SourceDocumentRow(Base):
    __tablename__ = "source_documents"
    source_id: Mapped[str] = mapped_column(Text, primary_key=True)


class CorpusReleaseRow(Base):
    __tablename__ = "corpus_releases"
    release_id: Mapped[str] = mapped_column(String(64), primary_key=True)


class CorpusMemberRow(Base):
    __tablename__ = "corpus_members"
    __table_args__ = (
        ForeignKeyConstraint(
            ["source_id", "snapshot_id"],
            ["source_snapshots.source_id", "source_snapshots.snapshot_id"],
        ),
    )
    release_id: Mapped[str] = mapped_column(
        String(64),
        ForeignKey("corpus_releases.release_id", deferrable=True, initially="DEFERRED"),
        primary_key=True,
    )
    source_id: Mapped[str] = mapped_column(Text, primary_key=True)
    snapshot_id: Mapped[str] = mapped_column(String(64))


class CorpusStateRow(Base):
    __tablename__ = "corpus_state"
    singleton: Mapped[bool] = mapped_column(primary_key=True)
    release_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("corpus_releases.release_id")
    )


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
    page_start: Mapped[int | None]
    page_end: Mapped[int | None]
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
