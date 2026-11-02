"""Service access checks precede every conversation and corpus operation."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import insert, select, update

from atrag.chat.services.chat import ChatService
from atrag.corpus import publish_corpus
from atrag.db_models import ChatConversationRow, ChatTurnRow, ChatUserRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


@pytest.fixture
def service(empty_corpus_schema, engine):
    migrate_database(engine)
    return ChatService(engine)


def login(service, subject="owner", email="owner@example.com", **changes):
    return service.authenticate(
        {
            "iss": "https://accounts.google.com",
            "sub": subject,
            "email": email,
            "email_verified": True,
            **changes,
        },
        ["owner@example.com", "other@example.com"],
    )


def test_identity_is_not_email_and_preferences_survive_reconnection(service, engine):
    owner = login(service)
    assert owner.language == "ru"
    assert service.get_user(owner) == owner
    for language in ("en", "pt", "ru", "pt"):
        owner = service.set_language(owner, language)
        assert owner.language == language
    engine.dispose()
    renewed = login(
        ChatService(engine), email="OTHER@example.com", iss="accounts.google.com"
    )
    assert renewed.id == owner.id
    assert renewed.language == "pt"
    assert renewed.email == "other@example.com"
    other = login(service, subject="Other", email="other@example.com")
    assert other.id != owner.id
    assert service.list_conversations(other) == ()
    with engine.connect() as connection:
        assert len(connection.execute(select(ChatUserRow)).all()) == 2


def test_concurrent_login_is_idempotent(service):
    with ThreadPoolExecutor(max_workers=2) as workers:
        users = list(workers.map(lambda _: login(service), range(2)))
    assert users[0] == users[1]


def test_concurrent_turn_retry_keeps_one_question_and_title(service):
    owner = login(service)
    conversation = service.create_conversation(owner)
    turn = uuid4()
    with ThreadPoolExecutor(max_workers=2) as workers:
        compositions = list(
            workers.map(
                lambda _: service.start_turn(
                    owner, conversation.id, turn, "First?", "en"
                ),
                range(2),
            )
        )
    assert compositions[0] == compositions[1]
    assert [item.id for item in service.load_history(owner, conversation.id)] == [turn]
    assert service.get_conversation(owner, conversation.id).title == "First?"


def test_denied_login_does_not_create_or_change_user(service, engine):
    owner = login(service)
    for changes in ({"email_verified": False}, {"email": "forbidden@example.com"}):
        with pytest.raises(PermissionError):
            login(service, **changes)
    with engine.connect() as connection:
        assert len(connection.execute(select(ChatUserRow)).all()) == 1
    assert service.get_user(owner) == owner


def test_empty_conversations_and_first_question_title(service, engine):
    owner = login(service)
    first = service.create_conversation(owner)
    assert first.user_id == owner.id
    assert first.title == ""
    assert service.get_conversation(owner, first.id) == first
    assert service.load_history(owner, first.id) == ()
    question = "  Кто\n считается   резидентом? " + "д" * 100
    turn = uuid4()
    composition = service.start_turn(owner, first.id, turn, question, "ru")
    assert composition.members == ()
    assert (
        service.get_conversation(owner, first.id).title
        == " ".join(question.split())[:80]
    )
    assert service.start_turn(owner, first.id, turn, question, "ru") == composition
    second = service.create_conversation(owner)
    assert service.list_conversations(owner) == (
        second,
        service.get_conversation(owner, first.id),
    )
    service.start_turn(owner, first.id, uuid4(), "Другой вопрос", "pt")
    assert (
        service.get_conversation(owner, first.id).title
        == " ".join(question.split())[:80]
    )
    assert len(service.load_history(owner, first.id)) == 2


def test_history_keeps_original_messages_sources_usage_and_status(service, engine):
    owner = login(service)
    conversation = service.create_conversation(owner)
    turn = uuid4()
    saved = service.start_turn(owner, conversation.id, turn, "Residência?", "pt")
    sources = [{"id": "1", "text": "São residentes", "snapshot_id": "old"}]
    with engine.begin() as connection:
        connection.execute(
            update(ChatTurnRow)
            .where(ChatTurnRow.id == turn)
            .values(
                answer="Sim [1]",
                sources=sources,
                status="completed",
                model_usage={"output_tokens": 30},
                embedding_usage={"input_tokens": 10},
            )
        )
    service.set_language(owner, "en")
    service.start_turn(owner, conversation.id, uuid4(), "Next?", "en")
    engine.dispose()
    history = ChatService(engine).load_history(owner, conversation.id)
    assert [item.question for item in history] == ["Residência?", "Next?"]
    item = history[0]
    assert (item.id, item.conversation_id, item.language) == (
        turn,
        conversation.id,
        "pt",
    )
    assert (item.answer, item.sources, item.status) == ("Sim [1]", sources, "completed")
    assert item.model_usage == {"output_tokens": 30}
    assert item.embedding_usage == {"input_tokens": 10}
    assert item.error is None and item.legacy_snapshot_id is None
    assert item.corpus_release_id == saved.release_id
    assert item.created_at.utcoffset() is not None
    assert service.resolve_turn_corpus(owner, conversation.id, turn) == saved


def test_foreign_missing_and_forged_owners_cannot_read_or_write(service, engine):
    owner = login(service)
    other = login(service, "other", "other@example.com")
    conversation = service.create_conversation(owner)
    turn = uuid4()
    service.start_turn(owner, conversation.id, turn, "Private", "en")
    for outsider in (
        other,
        replace(other, id=owner.id),
        replace(owner, issuer="https://evil.example"),
        replace(owner, subject="OWNER"),
    ):
        for operation in (
            lambda outsider=outsider: service.get_conversation(
                outsider, conversation.id
            ),
            lambda outsider=outsider: service.load_history(outsider, conversation.id),
            lambda outsider=outsider: service.start_turn(
                outsider, conversation.id, uuid4(), "Attack", "en"
            ),
            lambda outsider=outsider: service.resolve_turn_corpus(
                outsider, conversation.id, turn
            ),
        ):
            with pytest.raises(PermissionError, match="Access denied"):
                operation()
        assert service.list_conversations(outsider) == ()
    forged = replace(other, id=owner.id)
    for operation in (
        lambda: service.get_user(forged),
        lambda: service.set_language(forged, "pt"),
        lambda: service.create_conversation(forged),
    ):
        with pytest.raises(PermissionError, match="Access denied"):
            operation()
    missing = uuid4()
    for operation in (
        lambda: service.get_conversation(owner, missing),
        lambda: service.load_history(owner, missing),
        lambda: service.start_turn(owner, missing, uuid4(), "Missing", "en"),
        lambda: service.resolve_turn_corpus(owner, conversation.id, missing),
    ):
        with pytest.raises(PermissionError, match="Access denied"):
            operation()
    assert service.get_user(owner).language == "ru"
    assert [row.question for row in service.load_history(owner, conversation.id)] == [
        "Private"
    ]
    with engine.connect() as connection:
        assert len(connection.execute(select(ChatConversationRow)).all()) == 1


def test_turn_id_must_belong_to_requested_owned_conversation(service):
    owner = login(service)
    first, second = (
        service.create_conversation(owner),
        service.create_conversation(owner),
    )
    turn = uuid4()
    service.start_turn(owner, first.id, turn, "Original", "ru")
    with pytest.raises(PermissionError):
        service.resolve_turn_corpus(owner, second.id, turn)
    for conversation, question, language in (
        (second.id, "Original", "ru"),
        (first.id, "Changed", "ru"),
        (first.id, "Original", "pt"),
    ):
        with pytest.raises(ValueError, match="already assigned"):
            service.start_turn(owner, conversation, turn, question, language)
    assert service.get_conversation(owner, second.id).title == ""
    assert service.load_history(owner, second.id) == ()


def test_service_pins_corpus_and_checks_access_before_resolving_it(
    service, engine, prepared_document, tmp_path
):
    owner = login(service)
    other = login(service, "other", "other@example.com")
    conversation = service.create_conversation(owner)
    turn = uuid4()
    empty = service.start_turn(owner, conversation.id, turn, "Before", "en")
    data, parsed, chunked, embedded = prepared_document
    package = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    snapshot = index_package(package, engine, tmp_path / "server").snapshot_id
    published = publish_corpus(engine, {"pt-at-cirs": snapshot})
    assert published != empty
    assert service.start_turn(owner, conversation.id, turn, "Before", "en") == empty
    assert service.resolve_turn_corpus(owner, conversation.id, turn) == empty
    assert (
        service.start_turn(owner, conversation.id, uuid4(), "After", "en") == published
    )
    # A turn with no saved composition proves authorization precedes resolution.
    legacy = uuid4()
    with engine.begin() as connection:
        connection.execute(
            insert(ChatTurnRow).values(
                id=legacy,
                conversation_id=conversation.id,
                question="Legacy",
                language="ru",
            )
        )
    with pytest.raises(PermissionError):
        service.resolve_turn_corpus(other, conversation.id, legacy)
    with pytest.raises(ValueError, match="no saved corpus"):
        service.resolve_turn_corpus(owner, conversation.id, legacy)


@pytest.mark.parametrize("language", ["de", "", None])
def test_invalid_language_rejected_without_changing_preferences(service, language):
    owner = login(service)
    with pytest.raises(ValueError):
        service.set_language(owner, language)
    assert service.get_user(owner).language == "ru"


@pytest.mark.parametrize("question", ["", " \n ", None, 123, b"Question?"])
def test_invalid_question_rejected_without_creating_turn(service, question):
    owner = login(service)
    conversation = service.create_conversation(owner)
    with pytest.raises(ValueError):
        service.start_turn(owner, conversation.id, uuid4(), question, "ru")
    assert service.load_history(owner, conversation.id) == ()
    assert service.get_conversation(owner, conversation.id).title == ""


@pytest.mark.parametrize("language", ["de", "", None])
def test_invalid_turn_language_does_not_add_question(service, language):
    owner = login(service)
    conversation = service.create_conversation(owner)
    with pytest.raises(ValueError):
        service.start_turn(owner, conversation.id, uuid4(), "Question?", language)
    assert service.load_history(owner, conversation.id) == ()
