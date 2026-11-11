"""Preserve accounts and history while adding access status and login time."""

import sqlalchemy as sa
from alembic import op

revision = "0011_admin_users"
down_revision = "0010_faq_contracts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_users",
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "chat_users",
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("chat_users", "last_login_at")
    op.drop_column("chat_users", "is_active")
