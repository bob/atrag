"""Authenticate and validate chat use cases before entering persistence."""

from collections.abc import Mapping
from uuid import UUID

from pydantic import TypeAdapter
from sqlalchemy import Engine

from atrag import corpus
from atrag.chat.core.identity import parse_google_identity
from atrag.chat.db.store import ChatStore
from atrag.schemas.chat.conversations import Conversation, Turn, User
from atrag.schemas.materials import Nonempty
from atrag.schemas.settings import SupportedLanguage

LANGUAGE = TypeAdapter(SupportedLanguage)
QUESTION = TypeAdapter(Nonempty)


class ChatService:
    """User arguments come from authenticate(), never from URL or form fields."""

    def __init__(self, engine: Engine):
        self.engine = engine
        self.store = ChatStore(engine)

    def authenticate(
        self,
        claims: Mapping[str, object],
        allowed_emails: object,
        *,
        check_allowed_emails: object = True,
        record_login: bool = False,
    ) -> User:
        identity = parse_google_identity(
            claims, allowed_emails, check_allowed_emails=check_allowed_emails
        )
        return self.store.save_user(identity, record_login=record_login)

    def get_user(self, user: User) -> User:
        return self.store.get_user(user)

    def set_language(self, user: User, language: SupportedLanguage) -> User:
        return self.store.set_language(user, LANGUAGE.validate_python(language))

    def create_conversation(self, user: User) -> Conversation:
        return self.store.create_conversation(user)

    def list_conversations(self, user: User) -> tuple[Conversation, ...]:
        return self.store.list_conversations(user)

    def get_conversation(self, user: User, conversation_id: UUID) -> Conversation:
        return self.store.get_conversation(user, conversation_id)

    def load_history(self, user: User, conversation_id: UUID) -> tuple[Turn, ...]:
        return self.store.load_history(user, conversation_id)

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
        return self.store.start_turn(user, conversation_id, turn_id, question, language)

    def resolve_turn_corpus(
        self, user: User, conversation_id: UUID, turn_id: UUID
    ) -> corpus.CorpusComposition:
        return self.store.resolve_turn_corpus(user, conversation_id, turn_id)
