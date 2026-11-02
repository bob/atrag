"""Move the old conversation snapshot to each existing turn's provenance.

Revision ID: 0005_chat_provenance
Revises: 0004_chat
"""

import sqlalchemy as sa
from alembic import op

revision = "0005_chat_provenance"
down_revision = "0004_chat"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_turns",
        sa.Column(
            "legacy_snapshot_id",
            sa.String(64),
            sa.ForeignKey(
                "source_snapshots.snapshot_id", name="chat_turn_legacy_snapshot"
            ),
            nullable=True,
        ),
    )
    op.execute(
        "UPDATE chat_turns AS turn "
        "SET legacy_snapshot_id = conversation.snapshot_id "
        "FROM chat_conversations AS conversation "
        "WHERE turn.conversation_id = conversation.id"
    )
    op.drop_column("chat_conversations", "snapshot_id")


def downgrade() -> None:
    # Refuse to invent a snapshot for new, empty or multi-version conversations.
    incompatible = op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS ("
            "SELECT conversation.id FROM chat_conversations AS conversation "
            "LEFT JOIN chat_turns AS turn ON turn.conversation_id = conversation.id "
            "GROUP BY conversation.id "
            "HAVING count(DISTINCT turn.legacy_snapshot_id) <> 1 "
            "OR count(turn.legacy_snapshot_id) <> count(turn.id))"
        )
    )
    if incompatible:
        raise ValueError(
            "Cannot restore snapshot-bound conversations: every conversation must "
            "have turns with one known legacy snapshot."
        )
    op.add_column(
        "chat_conversations",
        sa.Column(
            "snapshot_id",
            sa.String(64),
            sa.ForeignKey(
                "source_snapshots.snapshot_id",
                name="chat_conversations_snapshot_id_fkey",
            ),
            nullable=True,
        ),
    )
    op.execute(
        "UPDATE chat_conversations AS conversation "
        "SET snapshot_id = turn.legacy_snapshot_id "
        "FROM chat_turns AS turn WHERE turn.conversation_id = conversation.id"
    )
    op.alter_column("chat_conversations", "snapshot_id", nullable=False)
    op.drop_column("chat_turns", "legacy_snapshot_id")
