"""Allow Ukrainian user preferences and conversation turns."""

from alembic import op

revision = "0008_ukrainian_language"
down_revision = "0007_rag_evidence"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table, constraint in (
        ("chat_users", "chat_user_language"),
        ("chat_turns", "chat_turn_language"),
    ):
        op.drop_constraint(constraint, table, type_="check")
        op.create_check_constraint(
            constraint, table, "language IN ('pt', 'en', 'uk', 'ru')"
        )


def downgrade() -> None:
    # Refuse to discard Ukrainian preferences or relabel saved conversation turns.
    for table, constraint in (
        ("chat_users", "chat_user_language"),
        ("chat_turns", "chat_turn_language"),
    ):
        op.drop_constraint(constraint, table, type_="check")
        op.create_check_constraint(constraint, table, "language IN ('ru', 'en', 'pt')")
