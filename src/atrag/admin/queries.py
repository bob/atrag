"""One SQL aggregation per user page, and immutable document provenance."""

from sqlalchemy import BigInteger, func, select

from atrag.db_models import (
    ChatConversationRow as Conversation,
)
from atrag.db_models import (
    ChatTurnRow as Turn,
)
from atrag.db_models import (
    ChatUserRow as User,
)
from atrag.db_models import (
    CorpusMemberRow,
    CorpusStateRow,
    SnapshotRow,
)


def users_query(search: str, active: bool | None, limit: int, offset: int):
    page = select(User).where(User.email.icontains(search, autoescape=True))
    if active is not None:
        page = page.where(User.is_active == active)
    users = page.order_by(User.email, User.id).limit(limit).offset(offset).subquery()
    columns = [
        users.c[name]
        for name in ("id", "email", "created_at", "last_login_at", "is_active")
    ]
    usage = [
        func.coalesce(
            func.sum(Turn.model_usage[field].astext.cast(BigInteger)), 0
        ).label(field)
        for field in ("input_tokens", "output_tokens", "total_tokens")
    ]
    return (
        select(*columns, *usage)
        .outerjoin(Conversation, Conversation.user_id == users.c.id)
        .outerjoin(Turn, Turn.conversation_id == Conversation.id)
        .group_by(*columns)
        .order_by(users.c.email, users.c.id)
    )


def documents_query():
    published = select(CorpusMemberRow.snapshot_id).join(
        CorpusStateRow, CorpusStateRow.release_id == CorpusMemberRow.release_id
    )
    fields = (
        "source_id",
        "snapshot_id",
        "source_sha256",
        "document_title",
        "source_format",
        "source_url",
        "local_path",
        "size_bytes",
        "indexed_at",
        "ingested_at",
    )
    return select(
        *(getattr(SnapshotRow, field) for field in fields),
        SnapshotRow.snapshot_id.in_(published).label("published"),
    ).order_by(
        SnapshotRow.source_id, SnapshotRow.indexed_at.desc(), SnapshotRow.snapshot_id
    )
