"""Preserve paid vectors independently of corpus indexing.

Revision ID: 0001_embedding_cache
Revises: None
"""

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision = "0001_embedding_cache"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "embeddings",
        sa.Column("model", sa.Text(), primary_key=True),
        sa.Column("dimensions", sa.Integer(), primary_key=True),
        sa.Column("input_hash", sa.String(64), primary_key=True),
        # Mixed spaces can be cached; a later search index must select one space.
        sa.Column("embedding", Vector(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "(model = 'text-embedding-3-small' AND dimensions BETWEEN 1 AND 1536) OR "
            "(model = 'text-embedding-3-large' AND dimensions BETWEEN 1 AND 3072)",
            name="embedding_space",
        ),
        sa.CheckConstraint("input_hash ~ '^[0-9a-f]{64}$'", name="embedding_hash"),
        sa.CheckConstraint(
            "vector_dims(embedding) = dimensions", name="embedding_dimensions"
        ),
        sa.CheckConstraint("vector_norm(embedding) > 0", name="embedding_nonzero"),
    )


def downgrade() -> None:
    op.drop_table("embeddings")
