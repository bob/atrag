"""Default new user profiles to English."""

from alembic import op

revision = "0009_english_default"
down_revision = "0008_ukrainian_language"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.alter_column("chat_users", "language", server_default="en")


def downgrade() -> None:
    op.alter_column("chat_users", "language", server_default="ru")
