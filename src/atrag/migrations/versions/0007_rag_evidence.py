"""Persist shared retrieval budgets and allow located material without PDF pages."""

from alembic import op

revision = "0007_rag_evidence"
down_revision = "0006_corpus_publication"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("""
        ALTER TABLE chat_turns ADD COLUMN retrieval_tokens integer NOT NULL DEFAULT 0
            CHECK (retrieval_tokens BETWEEN 0 AND 24000);
        ALTER TABLE legal_units ALTER COLUMN page_start DROP NOT NULL;
        ALTER TABLE legal_units ALTER COLUMN page_end DROP NOT NULL;
        ALTER TABLE search_fragments ALTER COLUMN page_start DROP NOT NULL;
        ALTER TABLE search_fragments ALTER COLUMN page_end DROP NOT NULL;
    """)
    for table in ("legal_units", "search_fragments"):
        op.execute(f"""
            ALTER TABLE {table} ADD CONSTRAINT located_material CHECK (
                (page_start IS NOT NULL AND page_end IS NOT NULL) OR
                (page_start IS NULL AND page_end IS NULL AND
                 COALESCE(payload->'location'->>'kind' IN ('html', 'docx'), false))
            )
        """)


def downgrade() -> None:
    # SET NOT NULL refuses a downgrade that would erase non-PDF provenance.
    for table in ("legal_units", "search_fragments"):
        op.execute(f"""
            ALTER TABLE {table} ALTER COLUMN page_start SET NOT NULL;
            ALTER TABLE {table} ALTER COLUMN page_end SET NOT NULL;
            ALTER TABLE {table} DROP CONSTRAINT located_material;
        """)
    op.drop_column("chat_turns", "retrieval_tokens")
