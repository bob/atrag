"""Admin use cases use real PostgreSQL and the same public chat boundary."""

from contextlib import contextmanager
from dataclasses import replace
from uuid import uuid4

import pytest
from sqlalchemy import insert, select, update

from atrag.admin.service import AdminService
from atrag.chat.db.execution import ExecutionStore
from atrag.chat.services.chat import ChatService
from atrag.chat.services.turns import TurnService
from atrag.db_models import ChatConversationRow, ChatTurnRow, ChatUserRow
from atrag.migrations import migrate_database
from atrag.schemas.chat.execution import ExecutionState, TextDelta
from atrag.schemas.settings import Settings

pytestmark = pytest.mark.integration


def login(service, subject="admin", email="admin@example.com", **kwargs):
    return service.authenticate(
        {
            "iss": "accounts.google.com",
            "sub": subject,
            "email": email,
            "email_verified": True,
        },
        ["admin@example.com", "reader@example.com"],
        **kwargs,
    )


@pytest.fixture
def administration(empty_corpus_schema, engine, tmp_path):
    migrate_database(engine)
    chat = ChatService(engine)
    actor = login(chat)
    reader = login(chat, "reader", "reader@example.com")
    settings = Settings(admin_emails="admin@example.com", data_dir=tmp_path)
    return AdminService(engine, settings), chat, actor, reader


def test_admin_access_is_rechecked_from_storage(administration, engine):
    admin, chat, actor, reader = administration
    assert admin.is_admin(actor)
    assert not admin.is_admin(reader)
    for denied in (
        reader,
        replace(actor, subject="forged"),
        replace(reader, email=actor.email),
    ):
        with pytest.raises(PermissionError):
            admin.list_users(denied)
        with pytest.raises(PermissionError):
            admin.list_documents(denied)
        with pytest.raises(PermissionError):
            admin.set_user_active(denied, reader.id, False)
    with pytest.raises(ValueError, match="self"):
        admin.set_user_active(actor, actor.id, False)
    with engine.begin() as connection:
        connection.execute(
            update(ChatUserRow)
            .where(ChatUserRow.id == actor.id)
            .values(is_active=False)
        )
    assert not admin.is_admin(actor)
    with pytest.raises(PermissionError):
        admin.list_users(actor)
    with pytest.raises(PermissionError):
        login(chat)


def test_blocked_session_loses_access_and_can_be_reactivated(administration):
    admin, chat, actor, reader = administration
    conversation = chat.create_conversation(reader)
    turn_id = uuid4()
    chat.start_turn(reader, conversation.id, turn_id, "Saved question", "en")
    admin.set_user_active(actor, reader.id, False)
    admin.set_user_active(actor, reader.id, False)
    for operation in (
        lambda: chat.get_user(reader),
        lambda: login(chat, "reader", reader.email),
        lambda: chat.set_language(reader, "pt"),
        lambda: chat.create_conversation(reader),
        lambda: chat.list_conversations(reader),
        lambda: chat.get_conversation(reader, conversation.id),
        lambda: chat.load_history(reader, conversation.id),
        lambda: chat.start_turn(reader, conversation.id, uuid4(), "New", "en"),
        lambda: chat.resolve_turn_corpus(reader, conversation.id, turn_id),
        lambda: ExecutionStore(chat.engine).get_turn(reader, conversation.id, turn_id),
    ):
        with pytest.raises(PermissionError):
            operation()
    admin.set_user_active(actor, reader.id, True)
    assert chat.load_history(reader, conversation.id)[0].question == "Saved question"
    assert login(chat, "reader", reader.email).id == reader.id


def test_login_time_is_separate_from_regular_authorization(administration):
    _, chat, actor, _ = administration
    assert actor.last_login_at is None
    session = login(chat, record_login=True)
    assert session.last_login_at is not None
    assert login(chat).last_login_at == session.last_login_at
    assert login(chat, record_login=True).last_login_at > session.last_login_at


@pytest.mark.parametrize("limit,offset", [(0, 0), (101, 0), (1, -1)])
def test_user_pagination_rejects_invalid_bounds(administration, limit, offset):
    admin, _, actor, _ = administration
    with pytest.raises(ValueError):
        admin.list_users(actor, limit=limit, offset=offset)


def test_role_never_bypasses_admission_or_verified_google_identity(administration):
    admin, chat, actor, _ = administration
    assert len(admin.list_users(actor, limit=100, offset=0)) == 2
    admin.set_user_active(actor, actor.id, True)
    with pytest.raises(ValueError):
        admin.set_user_active(actor, actor.id, "true")
    claims = {
        "iss": actor.issuer,
        "sub": actor.subject,
        "email": actor.email,
        "email_verified": True,
    }
    with pytest.raises(PermissionError):
        chat.authenticate(claims, [])
    with pytest.raises(PermissionError):
        chat.authenticate(
            {**claims, "email_verified": False}, [], check_allowed_emails=False
        )
    assert not AdminService(chat.engine, Settings()).is_admin(actor)


def test_usage_is_aggregated_once_with_search_status_and_pagination(
    administration, engine
):
    admin, chat, actor, reader = administration
    for status, count in (("completed", 10), ("failed", 20), ("interrupted", 30)):
        conversation = chat.create_conversation(reader)
        with engine.begin() as connection:
            connection.execute(
                insert(ChatTurnRow).values(
                    id=uuid4(),
                    conversation_id=conversation.id,
                    question=status,
                    language="en",
                    status=status,
                    model_usage={
                        "requests": 2,
                        "input_tokens": count,
                        "output_tokens": 3,
                        "total_tokens": count + 3,
                    },
                    embedding_usage={"input_tokens": 999},
                )
            )
    rows = admin.list_users(actor, search="READER@", limit=1)
    assert len(rows) == 1
    assert (rows[0].input_tokens, rows[0].output_tokens, rows[0].total_tokens) == (
        60,
        9,
        69,
    )
    assert admin.list_users(actor, search="READER@", limit=1, offset=1) == ()
    assert admin.list_users(actor, search="%") == ()
    assert admin.list_users(actor, search="admin")[0].total_tokens == 0
    assert admin.list_users(actor, active=False) == ()
    admin.set_user_active(actor, reader.id, False)
    assert admin.list_users(actor, active=False)[0].id == reader.id
    assert len(admin.list_users(actor, active=True)) == 1
    with pytest.raises(PermissionError):
        admin.set_user_active(actor, uuid4(), True)
    with engine.connect() as connection:
        assert (
            connection.scalar(select(ChatConversationRow.user_id).limit(1)) == reader.id
        )


@pytest.mark.parametrize("failure", [False, True])
def test_running_request_records_usage_after_deactivation(
    administration, engine, failure
):
    admin, chat, actor, reader = administration
    conversation = chat.create_conversation(reader)
    turn_id = uuid4()
    usage = {"requests": 2, "input_tokens": 10, "output_tokens": 4, "total_tokens": 14}

    class Session:
        def stream(self, turn):
            admin.set_user_active(actor, reader.id, False)
            yield TextDelta("Saved response")
            if failure:
                raise RuntimeError("Provider failure")

        def state(self):
            return ExecutionState(
                str(turn_id), False, "Saved response", "Saved response", usage
            )

    class Backend:
        @contextmanager
        def open(self, *args):
            yield Session()

    turns = TurnService(engine, Backend())
    events = list(turns.stream_turn(reader, conversation.id, turn_id, "Question", "en"))
    assert events[-1].kind == ("error" if failure else "completed")
    with engine.connect() as connection:
        saved = connection.execute(select(ChatTurnRow)).mappings().one()
        assert saved["status"] == ("failed" if failure else "completed")
        assert saved["answer"] == "Saved response"
        assert saved["model_usage"] == usage
    for resume in (False, True):
        with pytest.raises(PermissionError):
            list(
                turns.stream_turn(
                    reader, conversation.id, turn_id, "Question", "en", resume=resume
                )
            )
    assert admin.list_users(actor, search="reader")[0].total_tokens == 14
