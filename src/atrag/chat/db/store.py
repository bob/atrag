"""Server-side users and owned conversations; all writes have one transaction."""

from uuid import UUID, uuid4

from sqlalchemy import Engine, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atrag import corpus
from atrag.chat.db.access import owned_conversation, required_row, user_identity
from atrag.db_models import ChatConversationRow, ChatTurnRow, ChatUserRow
from atrag.schemas.chat.conversations import Conversation, Turn, User
from atrag.schemas.chat.identity import GoogleIdentity
from atrag.schemas.settings import SupportedLanguage


class ChatStore:
    """Persist validated chat values; each write uses one transaction."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def save_user(self, identity: GoogleIdentity, *, record_login: bool) -> User:
        statement = pg_insert(ChatUserRow).values(
            id=uuid4(),
            issuer=identity.issuer,
            subject=identity.subject,
            email=identity.email,
        )
        changes = {"email": identity.email}
        if record_login:
            changes["last_login_at"] = func.clock_timestamp()
            statement = statement.values(last_login_at=func.clock_timestamp())
        statement = statement.on_conflict_do_update(
            index_elements=["issuer", "subject"],
            set_=changes,
            where=ChatUserRow.is_active.is_(True),
        ).returning(ChatUserRow)
        with self.engine.begin() as connection:
            return User(**required_row(connection.execute(statement)))

    def get_user(self, user: User) -> User:
        with self.engine.connect() as connection:
            return User(
                **required_row(
                    connection.execute(select(ChatUserRow).where(user_identity(user)))
                )
            )

    def set_language(self, user: User, language: SupportedLanguage) -> User:
        with self.engine.begin() as connection:
            return User(
                **required_row(
                    connection.execute(
                        update(ChatUserRow)
                        .where(user_identity(user))
                        .values(language=language)
                        .returning(ChatUserRow)
                    )
                )
            )

    def create_conversation(self, user: User) -> Conversation:
        with self.engine.begin() as connection:
            required_row(
                connection.execute(select(ChatUserRow.id).where(user_identity(user)))
            )
            return Conversation(
                **connection.execute(
                    insert(ChatConversationRow)
                    .values(id=uuid4(), user_id=user.id, title="")
                    .returning(ChatConversationRow)
                )
                .mappings()
                .one()
            )

    def list_conversations(self, user: User) -> tuple[Conversation, ...]:
        with self.engine.connect() as connection:
            required_row(
                connection.execute(select(ChatUserRow.id).where(user_identity(user)))
            )
            rows = connection.execute(
                select(ChatConversationRow)
                .join(ChatUserRow)
                .where(user_identity(user))
                .order_by(
                    ChatConversationRow.created_at.desc(), ChatConversationRow.id.desc()
                )
            ).mappings()
            return tuple(Conversation(**row) for row in rows)

    def get_conversation(self, user: User, conversation_id: UUID) -> Conversation:
        with self.engine.connect() as connection:
            return Conversation(
                **required_row(
                    connection.execute(owned_conversation(user, conversation_id))
                )
            )

    def load_history(self, user: User, conversation_id: UUID) -> tuple[Turn, ...]:
        with self.engine.connect() as connection:
            required_row(connection.execute(owned_conversation(user, conversation_id)))
            rows = connection.execute(
                select(ChatTurnRow)
                .where(ChatTurnRow.conversation_id == conversation_id)
                .order_by(ChatTurnRow.created_at, ChatTurnRow.id)
            ).mappings()
            return tuple(Turn(**row) for row in rows)

    def start_turn(
        self,
        user: User,
        conversation_id: UUID,
        turn_id: UUID,
        question: str,
        language: SupportedLanguage,
    ) -> corpus.CorpusComposition:
        with self.engine.begin() as connection:
            required_row(
                connection.execute(
                    owned_conversation(user, conversation_id).with_for_update(
                        of=ChatConversationRow
                    )
                )
            )
            has_history = (
                connection.scalar(
                    select(ChatTurnRow.id)
                    .where(ChatTurnRow.conversation_id == conversation_id)
                    .limit(1)
                )
                is not None
            )
            composition = corpus.start_turn(
                connection,
                turn_id=turn_id,
                conversation_id=conversation_id,
                question=question,
                language=language,
            )
            if not has_history:
                connection.execute(
                    update(ChatConversationRow)
                    .where(ChatConversationRow.id == conversation_id)
                    .values(title=" ".join(question.split())[:80])
                )
            return composition

    def resolve_turn_corpus(
        self, user: User, conversation_id: UUID, turn_id: UUID
    ) -> corpus.CorpusComposition:
        with self.engine.connect() as connection:
            required_row(connection.execute(owned_conversation(user, conversation_id)))
            required_row(
                connection.execute(
                    select(ChatTurnRow.id).where(
                        ChatTurnRow.id == turn_id,
                        ChatTurnRow.conversation_id == conversation_id,
                    )
                )
            )
            return corpus.resolve_turn_corpus(connection, turn_id)
