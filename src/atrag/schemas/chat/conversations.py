"""Detached values read from storage; no ORM session or UI state escapes."""

from dataclasses import dataclass
from datetime import datetime
from uuid import UUID

from atrag.schemas.settings import SupportedLanguage


@dataclass(frozen=True)
class User:
    id: UUID
    issuer: str
    subject: str
    email: str
    language: SupportedLanguage
    created_at: datetime
    is_active: bool = True
    last_login_at: datetime | None = None


@dataclass(frozen=True)
class Conversation:
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime


@dataclass(frozen=True)
class Turn:
    id: UUID
    conversation_id: UUID
    language: SupportedLanguage
    question: str
    answer: str
    sources: list[dict]
    status: str
    model_usage: dict
    embedding_usage: dict
    error: str | None
    legacy_snapshot_id: str | None
    corpus_release_id: str | None
    retrieval_tokens: int
    created_at: datetime
