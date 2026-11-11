"""A session advisory lock spans streaming; turn writes commit independently."""

from contextlib import contextmanager
from hashlib import blake2b

from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from sqlalchemy import select, update

from atrag.chat.db.access import owned_conversation, required_row
from atrag.chat.db.store import ChatStore
from atrag.db_models import ChatTurnRow
from atrag.schemas.chat.conversations import Turn


@contextmanager
def checkpoint_connection(engine):
    arguments, options = engine.dialect.create_connect_args(engine.url)
    with Connection[DictRow].connect(
        *arguments, **options, autocommit=True, row_factory=dict_row
    ) as connection:
        yield connection


class ExecutionStore(ChatStore):
    @contextmanager
    def conversation_lock(self, user, conversation_id):
        # Check ownership before even opening the session used by the lock.
        self.get_conversation(user, conversation_id)
        key = int.from_bytes(
            blake2b(b"atrag.chat:" + conversation_id.bytes, digest_size=8).digest(),
            signed=True,
        )
        with checkpoint_connection(self.engine) as connection:
            acquired = connection.execute(
                "SELECT pg_try_advisory_lock(%s) AS acquired", (key,)
            ).fetchone()["acquired"]
            try:
                yield acquired
            finally:
                if acquired:
                    connection.execute("SELECT pg_advisory_unlock(%s)", (key,))

    def get_turn(self, user, conversation_id, turn_id):
        self.get_conversation(user, conversation_id)
        with self.engine.connect() as connection:
            row = (
                connection.execute(
                    select(ChatTurnRow).where(
                        ChatTurnRow.id == turn_id,
                        ChatTurnRow.conversation_id == conversation_id,
                    )
                )
                .mappings()
                .one_or_none()
            )
            return Turn(**row) if row is not None else None

    def save_execution(self, user, conversation_id, turn_id, **values):
        with self.engine.begin() as connection:
            # An admitted execution must record its result and usage even if
            # an administrator deactivates the owner while the provider runs.
            required_row(
                connection.execute(
                    owned_conversation(user, conversation_id, require_active=False)
                )
            )
            connection.execute(
                update(ChatTurnRow)
                .where(
                    ChatTurnRow.id == turn_id,
                    ChatTurnRow.conversation_id == conversation_id,
                )
                .values(**values)
            )

    def finish(self, user, conversation_id, turn_id, state):
        self.save_execution(
            user,
            conversation_id,
            turn_id,
            status="completed",
            error=None,
            answer=state.answer,
            model_usage=state.model_usage,
        )
        with self.engine.connect() as connection:
            return Turn(
                **required_row(
                    connection.execute(
                        select(ChatTurnRow).where(
                            ChatTurnRow.id == turn_id,
                            ChatTurnRow.conversation_id == conversation_id,
                        )
                    )
                )
            )
