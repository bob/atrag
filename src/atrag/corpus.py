"""Internal storage operations; caller's service owns access checks and transactions."""

import hashlib
import json
from dataclasses import dataclass
from uuid import UUID

from pydantic import TypeAdapter
from sqlalchemy import insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atrag.db_models import (
    ChatTurnRow,
    CorpusMemberRow,
    CorpusReleaseRow,
    CorpusStateRow,
    FragmentRow,
    SnapshotRow,
)
from atrag.schemas.materials import SHA256, Nonempty
from atrag.schemas.settings import SupportedLanguage

CHANGES = TypeAdapter(dict[Nonempty, SHA256 | None])


@dataclass(frozen=True)
class CorpusComposition:
    release_id: str
    members: tuple[tuple[str, str], ...]


def resolve_corpus(connection, release_id: str | None = None) -> CorpusComposition:
    if release_id is None:
        release_id = connection.scalar(select(CorpusStateRow.release_id))
    if (
        connection.scalar(
            select(CorpusReleaseRow.release_id).where(
                CorpusReleaseRow.release_id == release_id
            )
        )
        is None
    ):
        raise ValueError("Unknown corpus release")
    rows = connection.execute(
        select(CorpusMemberRow.source_id, CorpusMemberRow.snapshot_id)
        .where(CorpusMemberRow.release_id == release_id)
        .order_by(CorpusMemberRow.source_id)
    )
    return CorpusComposition(
        release_id, tuple((row.source_id, row.snapshot_id) for row in rows)
    )


def _validate_members(connection, members):
    spaces = set()
    for source, snapshot in members.items():
        row = connection.execute(
            select(
                SnapshotRow.source_id,
                SnapshotRow.embedding_model,
                SnapshotRow.embedding_dimensions,
            ).where(SnapshotRow.snapshot_id == snapshot)
        ).one_or_none()
        if row is None:
            raise ValueError("Included version is not indexed and ready")
        if row.source_id != source:
            raise ValueError("Version belongs to another source")
        if not connection.scalar(
            select(FragmentRow.fragment_id)
            .where(FragmentRow.snapshot_id == snapshot)
            .limit(1)
        ):
            raise ValueError("Included version is not indexed and ready")
        spaces.add((row.embedding_model, row.embedding_dimensions))
    if len(spaces) > 1:
        raise ValueError("Corpus has incompatible embedding spaces")


def publish_corpus(engine, changes: dict[str, str | None]) -> CorpusComposition:
    """Merge explicit additions/replacements/removals under one pointer row lock."""
    changes = CHANGES.validate_python(changes)
    with engine.begin() as connection:
        current = connection.scalar(select(CorpusStateRow.release_id).with_for_update())
        members = dict(resolve_corpus(connection, current).members)
        for source, snapshot in changes.items():
            if snapshot is None:
                members.pop(source, None)
            else:
                members[source] = snapshot
        _validate_members(connection, members)
        identity = hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            connection.scalar(
                select(CorpusReleaseRow.release_id).where(
                    CorpusReleaseRow.release_id == identity
                )
            )
            is None
        ):
            if members:
                connection.execute(
                    insert(CorpusMemberRow),
                    [
                        {"release_id": identity, "source_id": s, "snapshot_id": v}
                        for s, v in members.items()
                    ],
                )
            connection.execute(insert(CorpusReleaseRow).values(release_id=identity))
        connection.execute(update(CorpusStateRow).values(release_id=identity))
        return resolve_corpus(connection, identity)


def start_turn(
    connection,
    *,
    turn_id: UUID,
    conversation_id: UUID,
    question: str,
    language: SupportedLanguage,
) -> CorpusComposition:
    """Insert once with the currently visible release; retry keeps original provenance."""
    values = {
        "id": turn_id,
        "conversation_id": conversation_id,
        "question": question,
        "language": language,
    }
    connection.execute(
        pg_insert(ChatTurnRow)
        .values(
            **values,
            corpus_release_id=connection.scalar(select(CorpusStateRow.release_id)),
        )
        .on_conflict_do_nothing(index_elements=["id"])
    )
    row = connection.execute(
        select(
            ChatTurnRow.conversation_id, ChatTurnRow.question, ChatTurnRow.language
        ).where(ChatTurnRow.id == turn_id)
    ).one()
    if (row.conversation_id, row.question, row.language) != (
        conversation_id,
        question,
        language,
    ):
        raise ValueError("Turn UUID is already assigned to another request")
    return resolve_turn_corpus(connection, turn_id)


def resolve_turn_corpus(connection, turn_id: UUID) -> CorpusComposition:
    release = connection.scalar(
        select(ChatTurnRow.corpus_release_id).where(ChatTurnRow.id == turn_id)
    )
    if release is None:
        raise ValueError("Turn has no saved corpus composition")
    return resolve_corpus(connection, release)
