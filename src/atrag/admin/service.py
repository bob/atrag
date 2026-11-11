"""Recheck stored identity, current status and configured role on every call."""

from contextlib import contextmanager
from dataclasses import replace
from uuid import UUID

from sqlalchemy import Engine, select, update

from atrag.admin import queries
from atrag.admin.files import download_source, source_available
from atrag.admin.schemas import DocumentSummary, DocumentVersion, UserSummary
from atrag.chat.db.access import required_row, user_identity
from atrag.db_models import ChatUserRow
from atrag.schemas.chat.conversations import User
from atrag.schemas.settings import Settings


class AdminService:
    def __init__(self, engine: Engine, settings: Settings):
        self.engine = engine
        self.settings = settings

    @contextmanager
    def authorized(self, actor: User):
        with self.engine.begin() as connection:
            row = required_row(
                connection.execute(
                    select(ChatUserRow)
                    .where(user_identity(actor))
                    .with_for_update(read=True)
                )
            )
            if row["email"] not in self.settings.admin_emails:
                raise PermissionError("Access denied")
            yield connection

    def is_admin(self, actor: User) -> bool:
        try:
            with self.authorized(actor):
                return True
        except PermissionError:
            return False

    def list_users(
        self,
        actor: User,
        *,
        search: str = "",
        active: bool | None = None,
        limit: int = 25,
        offset: int = 0,
    ) -> tuple[UserSummary, ...]:
        if not 1 <= limit <= 100 or offset < 0:
            raise ValueError("Invalid pagination")
        with self.authorized(actor) as connection:
            return tuple(
                UserSummary(**row)
                for row in connection.execute(
                    queries.users_query(search, active, limit, offset)
                ).mappings()
            )

    def set_user_active(self, actor: User, user_id: UUID, active: bool) -> None:
        with self.authorized(actor) as connection:
            if type(active) is not bool:
                raise ValueError("Active must be a boolean")
            if actor.id == user_id and not active:
                raise ValueError("Cannot deactivate self")
            required_row(
                connection.execute(
                    update(ChatUserRow)
                    .where(ChatUserRow.id == user_id)
                    .values(is_active=active)
                    .returning(ChatUserRow.id)
                )
            )

    def list_documents(self, actor: User) -> tuple[DocumentSummary, ...]:
        with self.authorized(actor) as connection:
            versions = tuple(
                DocumentVersion(**row)
                for row in connection.execute(queries.documents_query()).mappings()
            )
        groups: dict[str, list[DocumentVersion]] = {}
        for version in versions:
            groups.setdefault(version.source_id, []).append(
                replace(
                    version, available=source_available(self.settings.data_dir, version)
                )
            )
        return tuple(summarize_document(tuple(items)) for items in groups.values())

    def download_source(self, actor: User, snapshot_id: str):
        with self.authorized(actor) as connection:
            row = required_row(
                connection.execute(
                    queries.documents_query().where(
                        queries.SnapshotRow.snapshot_id == snapshot_id
                    )
                )
            )
            return download_source(self.settings.data_dir, DocumentVersion(**row))


def summarize_document(versions: tuple[DocumentVersion, ...]) -> DocumentSummary:
    latest = versions[0]
    sizes = {version.source_sha256: version.size_bytes for version in versions}
    return DocumentSummary(
        source_id=latest.source_id,
        document_title=latest.document_title,
        indexed_at=latest.indexed_at,
        source_format=", ".join(sorted({v.source_format for v in versions})),
        size_bytes=sum(sizes.values()),
        published=any(v.published for v in versions),
        versions=versions,
    )
