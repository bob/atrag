"""Server-side users and owned conversations; all writes have one transaction."""

from collections.abc import Mapping
from uuid import UUID, uuid4

from pydantic import TypeAdapter
from sqlalchemy import Engine, and_, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from atrag import corpus
from atrag.chat.identity import parse_google_identity
from atrag.chat.models import Conversation, Turn, User
from atrag.db_models import ChatConversationRow, ChatTurnRow, ChatUserRow
from atrag.schemas.materials import Nonempty
from atrag.schemas.settings import SupportedLanguage

LANGUAGE = TypeAdapter(SupportedLanguage)
QUESTION = TypeAdapter(Nonempty)


def _identity(user: User):
    return and_(
        ChatUserRow.id == user.id,
        ChatUserRow.issuer == user.issuer,
        ChatUserRow.subject == user.subject,
    )


def _required(result):
    row = result.mappings().one_or_none()
    if row is None:
        raise PermissionError("Access denied")
    return row


def _conversation(user: User, conversation_id: UUID):
    return (
        select(ChatConversationRow)
        .join(ChatUserRow)
        .where(ChatConversationRow.id == conversation_id, _identity(user))
    )


class ChatService:
    """User arguments come from authenticate(), never from URL or form fields."""

    def __init__(self, engine: Engine):
        self.engine = engine

    def authenticate(
        self, claims: Mapping[str, object], allowed_emails: object
    ) -> User:
        identity = parse_google_identity(claims, allowed_emails)
        statement = pg_insert(ChatUserRow).values(
            id=uuid4(),
            issuer=identity.issuer,
            subject=identity.subject,
            email=identity.email,
        )
        statement = statement.on_conflict_do_update(
            index_elements=["issuer", "subject"], set_={"email": identity.email}
        ).returning(ChatUserRow)
        with self.engine.begin() as connection:
            return User(**connection.execute(statement).mappings().one())

    def get_user(self, user: User) -> User:
        with self.engine.connect() as connection:
            return User(
                **_required(
                    connection.execute(select(ChatUserRow).where(_identity(user)))
                )
            )

    def set_language(self, user: User, language: SupportedLanguage) -> User:
        language = LANGUAGE.validate_python(language)
        with self.engine.begin() as connection:
            return User(
                **_required(
                    connection.execute(
                        update(ChatUserRow)
                        .where(_identity(user))
                        .values(language=language)
                        .returning(ChatUserRow)
                    )
                )
            )

    def create_conversation(self, user: User) -> Conversation:
        with self.engine.begin() as connection:
            _required(connection.execute(select(ChatUserRow.id).where(_identity(user))))
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
            rows = connection.execute(
                select(ChatConversationRow)
                .join(ChatUserRow)
                .where(_identity(user))
                .order_by(
                    ChatConversationRow.created_at.desc(), ChatConversationRow.id.desc()
                )
            ).mappings()
            return tuple(Conversation(**row) for row in rows)

    def get_conversation(self, user: User, conversation_id: UUID) -> Conversation:
        with self.engine.connect() as connection:
            return Conversation(
                **_required(connection.execute(_conversation(user, conversation_id)))
            )

    def load_history(self, user: User, conversation_id: UUID) -> tuple[Turn, ...]:
        with self.engine.connect() as connection:
            _required(connection.execute(_conversation(user, conversation_id)))
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
        text: str,
        language: SupportedLanguage,
    ) -> corpus.CorpusComposition:
        question = QUESTION.validate_python(text, strict=True)
        language = LANGUAGE.validate_python(language)
        with self.engine.begin() as connection:
            _required(
                connection.execute(
                    _conversation(user, conversation_id).with_for_update(
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
            _required(connection.execute(_conversation(user, conversation_id)))
            _required(
                connection.execute(
                    select(ChatTurnRow.id).where(
                        ChatTurnRow.id == turn_id,
                        ChatTurnRow.conversation_id == conversation_id,
                    )
                )
            )
            return corpus.resolve_turn_corpus(connection, turn_id)
