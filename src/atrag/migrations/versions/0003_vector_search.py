"""Remove the obsolete lexical search column and index from the MVP schema.

Revision ID: 0003_vector_search
Revises: 0002_corpus
"""

from alembic import op

revision = "0003_vector_search"
down_revision = "0002_corpus"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_search_fragments_fts", table_name="search_fragments")
    op.drop_column("search_fragments", "search_vector")


def downgrade() -> None:
    # Restore the historical 0002 schema when rolling back that application.
    op.execute("""
        ALTER TABLE search_fragments ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            setweight(to_tsvector('portuguese', unit_title), 'A') ||
            setweight(to_tsvector('portuguese', document_title), 'B') ||
            setweight(to_tsvector('portuguese', text_normalized), 'D')
        ) STORED NOT NULL
    """)
    op.create_index(
        "ix_search_fragments_fts",
        "search_fragments",
        ["search_vector"],
        postgresql_using="gin",
    )
