"""Upgrade existing accounts without changing ownership, usage or history."""

from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


def test_admin_upgrade_preserves_registered_users_and_history(
    empty_corpus_schema, engine
):
    config = Config()
    config.set_main_option(
        "script_location", str(Path("src/atrag/migrations").resolve())
    )
    user_id, conversation_id, turn_id = uuid4(), uuid4(), uuid4()
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "0010_faq_contracts")
        connection.execute(
            text(
                "INSERT INTO chat_users (id,issuer,subject,email) VALUES (:id,'accounts.google.com','old','old@example.com')"
            ),
            {"id": user_id},
        )
        connection.execute(
            text(
                "INSERT INTO chat_conversations (id,user_id,title) VALUES (:id,:user,'History')"
            ),
            {"id": conversation_id, "user": user_id},
        )
        connection.execute(
            text(
                "INSERT INTO chat_turns (id,conversation_id,language,question,answer,model_usage) VALUES (:id,:conversation,'en','Question','Answer',CAST(:usage AS jsonb))"
            ),
            {
                "id": turn_id,
                "conversation": conversation_id,
                "usage": '{"total_tokens":12}',
            },
        )
        original = connection.execute(text("SELECT * FROM chat_users")).mappings().one()
        history = connection.execute(text("SELECT * FROM chat_turns")).mappings().one()
    migrate_database(engine)
    migrate_database(engine)
    with engine.connect() as connection:
        user = dict(
            connection.execute(text("SELECT * FROM chat_users")).mappings().one()
        )
        assert user.pop("is_active") is True
        assert user.pop("last_login_at") is None
        assert user == original
        assert (
            connection.execute(text("SELECT * FROM chat_turns")).mappings().one()
            == history
        )
