"""Detached admin results: users, logical documents, and prepared versions."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True)
class UserSummary:
    id: UUID
    email: str
    created_at: datetime
    last_login_at: datetime | None
    is_active: bool
    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class DocumentVersion:
    source_id: str
    snapshot_id: str
    source_sha256: str
    document_title: str
    source_format: str
    source_url: str | None
    local_path: str
    size_bytes: int
    indexed_at: datetime
    ingested_at: datetime
    published: bool
    available: bool = False


@dataclass(frozen=True)
class DocumentSummary:
    source_id: str
    document_title: str
    indexed_at: datetime
    source_format: str
    size_bytes: int
    published: bool
    versions: tuple[DocumentVersion, ...]


@dataclass(frozen=True)
class Download:
    data: bytes
    filename: str
    mime: str
