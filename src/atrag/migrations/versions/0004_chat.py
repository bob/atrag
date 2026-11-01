"""Persist OIDC identities, snapshot-bound conversations and durable turn IDs.

Revision ID: 0004_chat
Revises: 0003_vector_search
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004_chat"
down_revision = "0003_vector_search"
branch_labels = None
depends_on = None


def _created_at() -> sa.Column:
    return sa.Column(
        "created_at",
        sa.DateTime(timezone=True),
        server_default=sa.func.now(),
        nullable=False,
    )


def _users() -> None:
    op.create_table(
        "chat_users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("issuer", sa.Text(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("language", sa.Text(), server_default="ru", nullable=False),
        _created_at(),
        sa.UniqueConstraint("issuer", "subject", name="chat_user_identity"),
        sa.CheckConstraint("btrim(issuer) <> ''", name="chat_user_issuer"),
        sa.CheckConstraint("btrim(subject) <> ''", name="chat_user_subject"),
        sa.CheckConstraint("btrim(email) <> ''", name="chat_user_email"),
        sa.CheckConstraint("language IN ('ru', 'en', 'pt')", name="chat_user_language"),
    )


def _conversations() -> None:
    op.create_table(
        "chat_conversations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("chat_users.id"), nullable=False),
        sa.Column(
            "snapshot_id",
            sa.String(64),
            sa.ForeignKey("source_snapshots.snapshot_id"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        _created_at(),
    )
    op.create_index("ix_chat_conversations_user_id", "chat_conversations", ["user_id"])


def _turns() -> None:
    op.create_table(
        "chat_turns",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.Uuid(),
            sa.ForeignKey("chat_conversations.id"),
            nullable=False,
        ),
        sa.Column("language", sa.Text(), nullable=False),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), server_default="", nullable=False),
        sa.Column(
            "sources", JSONB(), server_default=sa.text("'[]'::jsonb"), nullable=False
        ),
        sa.Column("status", sa.Text(), server_default="pending", nullable=False),
        sa.Column(
            "model_usage",
            JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "embedding_usage",
            JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", sa.Text(), nullable=True),
        _created_at(),
        sa.CheckConstraint("language IN ('ru', 'en', 'pt')", name="chat_turn_language"),
        sa.CheckConstraint("btrim(question) <> ''", name="chat_turn_question"),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'interrupted', 'failed', 'completed')",
            name="chat_turn_status",
        ),
        sa.CheckConstraint("jsonb_typeof(sources) = 'array'", name="chat_turn_sources"),
        sa.CheckConstraint(
            "jsonb_typeof(model_usage) = 'object'", name="chat_turn_model_usage"
        ),
        sa.CheckConstraint(
            "jsonb_typeof(embedding_usage) = 'object'", name="chat_turn_embedding_usage"
        ),
    )
    op.create_index("ix_chat_turns_conversation_id", "chat_turns", ["conversation_id"])


def upgrade() -> None:
    _users()
    _conversations()
    _turns()


def downgrade() -> None:
    op.drop_index("ix_chat_turns_conversation_id", table_name="chat_turns")
    op.drop_table("chat_turns")
    op.drop_index("ix_chat_conversations_user_id", table_name="chat_conversations")
    op.drop_table("chat_conversations")
    op.drop_table("chat_users")
