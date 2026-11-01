"""Chat schema and explicit checkpoint migrations against the reused test database."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import MetaData, Table, insert, null, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from atrag.db_models import ChatConversationRow, ChatTurnRow, ChatUserRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def empty_chat_database(empty_corpus_schema, engine):
    def clear_checkpoints():
        with engine.begin() as connection:
            connection.execute(
                text(
                    "DROP TABLE IF EXISTS checkpoint_writes, checkpoint_blobs, "
                    "checkpoints, checkpoint_migrations"
                )
            )

    clear_checkpoints()
    yield
    clear_checkpoints()


@pytest.fixture
def stored_chat(engine, prepared_document, tmp_path):
    migrate_database(engine)
    data, parsed, chunked, embedded = prepared_document
    package = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    snapshot = index_package(package, engine, tmp_path / "indexed").snapshot_id
    tables = {
        name: Table(f"chat_{name}", MetaData(), autoload_with=engine)
        for name in ("users", "conversations", "turns")
    }
    user_id, conversation_id, turn_id = uuid4(), uuid4(), uuid4()
    with engine.begin() as connection:
        connection.execute(
            insert(tables["users"]),
            {
                "id": user_id,
                "issuer": "https://accounts.google.com",
                "subject": "google-subject",
                "email": "owner@example.com",
            },
        )
        connection.execute(
            insert(tables["conversations"]),
            {
                "id": conversation_id,
                "user_id": user_id,
                "snapshot_id": snapshot,
                "title": "Резидентство",
            },
        )
        connection.execute(
            insert(tables["turns"]),
            {
                "id": turn_id,
                "conversation_id": conversation_id,
                "language": "ru",
                "question": "Кто считается резидентом?",
            },
        )
    return tables, user_id, conversation_id, turn_id, snapshot


def test_chat_data_survives_repeat_migration_and_new_connections(engine, stored_chat):
    tables, user_id, conversation_id, turn_id, snapshot = stored_chat
    sources = [{"id": "1", "text": "São residentes", "page": 48}]
    with engine.begin() as connection:
        connection.execute(
            update(tables["turns"]).where(tables["turns"].c.id == turn_id),
            {
                "answer": "См. [1]",
                "sources": sources,
                "status": "completed",
                "model_usage": {"input_tokens": 100, "output_tokens": 30},
                "embedding_usage": {"input_tokens": 10},
            },
        )
    migrate_database(engine)
    migrate_database(engine)
    engine.dispose()
    with engine.connect() as connection:
        user = connection.execute(select(tables["users"])).mappings().one()
        conversation = (
            connection.execute(select(tables["conversations"])).mappings().one()
        )
        turn = connection.execute(select(tables["turns"])).mappings().one()
        assert user["id"] == user_id
        assert (user["issuer"], user["subject"], user["email"], user["language"]) == (
            "https://accounts.google.com",
            "google-subject",
            "owner@example.com",
            "ru",
        )
        assert conversation["id"] == conversation_id
        assert conversation["user_id"] == user_id
        assert conversation["snapshot_id"] == snapshot
        assert conversation["title"] == "Резидентство"
        assert turn["id"] == turn_id
        assert turn["conversation_id"] == conversation_id
        assert (turn["question"], turn["language"]) == (
            "Кто считается резидентом?",
            "ru",
        )
        assert (turn["answer"], turn["status"], turn["sources"]) == (
            "См. [1]",
            "completed",
            sources,
        )
        assert turn["model_usage"] == {"input_tokens": 100, "output_tokens": 30}
        assert turn["embedding_usage"] == {"input_tokens": 10}
        assert turn["error"] is None
        assert all(
            row["created_at"].utcoffset() is not None
            for row in (user, conversation, turn)
        )
    with Session(engine) as session:
        user_row = session.get(ChatUserRow, user_id)
        conversation_row = session.get(ChatConversationRow, conversation_id)
        turn_row = session.get(ChatTurnRow, turn_id)
        assert user_row is not None and user_row.email == "owner@example.com"
        assert conversation_row is not None and conversation_row.snapshot_id == snapshot
        assert turn_row is not None and turn_row.sources == sources


@pytest.mark.parametrize(
    ("table", "values"),
    [
        ("users", {"language": "fr"}),
        ("users", {"issuer": " "}),
        ("users", {"subject": " "}),
        ("users", {"email": " "}),
        ("conversations", {"user_id": uuid4()}),
        ("conversations", {"snapshot_id": "f" * 64}),
        ("turns", {"conversation_id": uuid4()}),
        ("turns", {"language": "fr"}),
        ("turns", {"question": " "}),
        ("turns", {"status": "unknown"}),
        ("turns", {"sources": {}}),
        ("turns", {"sources": None}),
        ("turns", {"model_usage": []}),
        ("turns", {"model_usage": None}),
        ("turns", {"embedding_usage": []}),
        ("turns", {"embedding_usage": None}),
    ],
)
def test_chat_schema_rejects_invalid_values(engine, stored_chat, table, values):
    tables, *_ = stored_chat
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(update(tables[table]).values(**values))


@pytest.mark.parametrize(
    ("table", "columns"),
    [
        ("users", ("id", "issuer", "subject", "email", "language", "created_at")),
        ("conversations", ("id", "user_id", "snapshot_id", "title", "created_at")),
        (
            "turns",
            (
                "id",
                "conversation_id",
                "language",
                "question",
                "answer",
                "sources",
                "status",
                "model_usage",
                "embedding_usage",
                "created_at",
            ),
        ),
    ],
)
def test_chat_required_values_cannot_be_null(engine, stored_chat, table, columns):
    tables, *_ = stored_chat
    for column in columns:
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(update(tables[table]).values({column: null()}))


@pytest.mark.parametrize("table", ["users", "conversations", "turns"])
def test_chat_ids_cannot_be_reused(engine, stored_chat, table):
    tables, *_ = stored_chat
    with engine.connect() as connection:
        values = dict(connection.execute(select(tables[table])).mappings().one())
    if table == "users":
        values["subject"] = "different-identity-same-id"
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(tables[table]), values)


def test_identity_and_turn_ids_are_unique(engine, stored_chat):
    tables, _, _, turn_id, _ = stored_chat
    with engine.connect() as connection:
        user = dict(connection.execute(select(tables["users"])).mappings().one())
        turn = dict(connection.execute(select(tables["turns"])).mappings().one())
        assert (
            turn["status"],
            turn["answer"],
            turn["sources"],
            turn["model_usage"],
            turn["embedding_usage"],
        ) == ("pending", "", [], {}, {})
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(tables["users"]), user | {"id": uuid4()})
    with engine.begin() as connection:
        connection.execute(
            insert(tables["users"]), user | {"id": uuid4(), "subject": "other-subject"}
        )
        connection.execute(
            insert(tables["users"]), user | {"id": uuid4(), "issuer": "other-issuer"}
        )
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(insert(tables["turns"]), turn)
    with engine.connect() as connection:
        assert connection.scalar(select(tables["turns"].c.id)) == turn_id


def test_chat_downgrade_and_upgrade_preserve_corpus(engine, stored_chat):
    *_, snapshot = stored_chat
    config = Config()
    config.set_main_option(
        "script_location", str(Path("src/atrag/migrations").resolve())
    )
    with engine.begin() as connection:
        before = connection.execute(
            text(
                "SELECT fragment_id, embedding::text, payload FROM search_fragments ORDER BY fragment_id"
            )
        ).all()
        config.attributes["connection"] = connection
        command.downgrade(config, "0003_vector_search")
        assert connection.scalar(text("SELECT to_regclass('chat_users')")) is None
        assert (
            connection.scalar(text("SELECT to_regclass('chat_conversations')")) is None
        )
        assert connection.scalar(text("SELECT to_regclass('chat_turns')")) is None
    migrate_database(engine)
    with engine.connect() as connection:
        assert (
            connection.scalar(text("SELECT snapshot_id FROM source_snapshots"))
            == snapshot
        )
        assert (
            connection.execute(
                text(
                    "SELECT fragment_id, embedding::text, payload FROM search_fragments ORDER BY fragment_id"
                )
            ).all()
            == before
        )


def test_migration_initializes_durable_checkpoints(engine, database_dsn):
    migrate_database(engine)
    thread_id = str(uuid4())
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id, "checkpoint_ns": ""}
    }
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"question": "CIRS 16"}
    checkpoint["channel_versions"] = {"question": "1"}
    with PostgresSaver.from_conn_string(database_dsn) as saver:
        saved = saver.put(
            config,
            checkpoint,
            {"source": "input", "step": 0, "parents": {}},
            {"question": "1"},
        )
    migrate_database(engine)
    with PostgresSaver.from_conn_string(database_dsn) as saver:
        restored = saver.get_tuple(saved)
        assert restored is not None
        assert restored.checkpoint["channel_values"] == {"question": "CIRS 16"}
        assert restored.config["configurable"]["thread_id"] == thread_id
        saver.delete_thread(thread_id)


def test_checkpoint_failure_is_retryable_and_releases_the_migration_lock(
    engine, monkeypatch, database_dsn
):
    import psycopg

    def fail_setup(self):
        raise psycopg.OperationalError("checkpoint setup interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(PostgresSaver, "setup", fail_setup)
        with pytest.raises(psycopg.OperationalError, match="interrupted"):
            migrate_database(engine)
    with psycopg.connect(database_dsn, autocommit=True) as connection:
        assert connection.execute(
            "SELECT version_num FROM alembic_version"
        ).fetchone() == ("0004_chat",)
        assert connection.execute(
            "SELECT pg_try_advisory_lock(6174726167)"
        ).fetchone() == (True,)
        connection.execute("SELECT pg_advisory_unlock(6174726167)")
    migrate_database(engine)
    with PostgresSaver.from_conn_string(database_dsn) as saver:
        assert saver.get_tuple({"configurable": {"thread_id": str(uuid4())}}) is None


def test_concurrent_migrations_are_serialized(engine, database_dsn):
    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(migrate_database, engine) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    with PostgresSaver.from_conn_string(database_dsn) as saver:
        assert saver.get_tuple({"configurable": {"thread_id": str(uuid4())}}) is None
