"""Atomic snapshot corpus with exact vectors and Portuguese weighted FTS.

Revision ID: 0002_corpus
Revises: 0001_embedding_cache
"""

from alembic import op

revision = "0002_corpus"
down_revision = "0001_embedding_cache"
branch_labels = None
depends_on = None


def _address_columns():
    return """
        snapshot_id varchar(64) NOT NULL REFERENCES source_snapshots(snapshot_id),
        legal_unit_id text NOT NULL,
        parent_id text,
        instrument_id text NOT NULL,
        code text,
        document_scope text NOT NULL CHECK (document_scope IN ('code', 'approving_decree', 'preamble')),
        kind text NOT NULL CHECK (kind IN ('document', 'chapter', 'section', 'subsection', 'article', 'paragraph', 'subparagraph', 'item', 'note')),
        role text NOT NULL CHECK (role IN ('norm', 'approving_decree', 'code', 'preamble', 'editorial_note', 'transitional', 'closing')),
        chapter text, section text, article text, paragraph text, subparagraph text,
        legal_status text NOT NULL CHECK (legal_status IN ('unknown', 'repealed')),
        unit_title text NOT NULL,
        citation_label text NOT NULL,
        page_start integer NOT NULL CHECK (page_start > 0),
        page_end integer NOT NULL CHECK (page_end >= page_start),
        text_original text NOT NULL,
        text_normalized text NOT NULL,
        payload jsonb NOT NULL,
    """


def upgrade() -> None:
    op.execute("""
        CREATE TABLE source_snapshots (
            snapshot_id varchar(64) PRIMARY KEY CHECK (snapshot_id ~ '^[0-9a-f]{64}$'),
            source_id text NOT NULL,
            source_sha256 varchar(64) NOT NULL CHECK (source_sha256 = snapshot_id),
            document_title text NOT NULL,
            document_type text NOT NULL CHECK (document_type IN ('law', 'at_instruction', 'individual_ruling', 'faq')),
            publisher text NOT NULL,
            language text NOT NULL CHECK (language IN ('pt', 'ru', 'en')),
            source_url text,
            source_url_status text NOT NULL CHECK (source_url_status IN ('unknown', 'candidate', 'verified')),
            local_path text NOT NULL,
            size_bytes integer NOT NULL CHECK (size_bytes > 0),
            consolidation_label text,
            publication_date date,
            revision_publication_date date,
            downloaded_at timestamptz,
            ingested_at timestamptz NOT NULL,
            indexed_at timestamptz NOT NULL,
            embedding_model text NOT NULL CHECK (embedding_model IN ('text-embedding-3-small', 'text-embedding-3-large')),
            embedding_dimensions integer NOT NULL CHECK (embedding_dimensions = 1536),
            parser_version text NOT NULL,
            chunker_version text NOT NULL,
            review_status text NOT NULL CHECK (review_status IN ('ready', 'needs_review')),
            reports jsonb NOT NULL,
            CHECK (source_url_status = 'unknown' OR source_url IS NOT NULL),
            UNIQUE (snapshot_id, embedding_model, embedding_dimensions)
        )
    """)
    op.create_index("ix_source_snapshots_source_id", "source_snapshots", ["source_id"])
    op.execute(
        "CREATE TABLE legal_units ("
        + _address_columns()
        + """
        PRIMARY KEY (snapshot_id, legal_unit_id),
        FOREIGN KEY (snapshot_id, parent_id) REFERENCES legal_units(snapshot_id, legal_unit_id)
            DEFERRABLE INITIALLY DEFERRED
    )"""
    )
    op.create_index(
        "ix_legal_units_address",
        "legal_units",
        ["snapshot_id", "code", "document_scope", "article"],
    )
    op.execute("""
        CREATE TABLE document_tables (
            snapshot_id varchar(64) NOT NULL REFERENCES source_snapshots(snapshot_id),
            table_id text NOT NULL,
            payload jsonb NOT NULL,
            PRIMARY KEY (snapshot_id, table_id)
        )
    """)
    op.execute(
        "CREATE TABLE search_fragments ("
        + _address_columns()
        + """
        fragment_id varchar(64) NOT NULL CHECK (fragment_id ~ '^[0-9a-f]{64}$'),
        document_title text NOT NULL,
        review_status text NOT NULL CHECK (review_status IN ('ready', 'needs_review')),
        embedding_model text NOT NULL,
        embedding_dimensions integer NOT NULL,
        embedding_input_hash varchar(64) NOT NULL,
        embedding vector(1536) NOT NULL CHECK (vector_norm(embedding) > 0),
        search_vector tsvector GENERATED ALWAYS AS (
            setweight(to_tsvector('portuguese', unit_title), 'A') ||
            setweight(to_tsvector('portuguese', document_title), 'B') ||
            setweight(to_tsvector('portuguese', text_normalized), 'D')
        ) STORED NOT NULL,
        PRIMARY KEY (snapshot_id, fragment_id),
        FOREIGN KEY (snapshot_id, legal_unit_id) REFERENCES legal_units(snapshot_id, legal_unit_id),
        FOREIGN KEY (snapshot_id, embedding_model, embedding_dimensions)
            REFERENCES source_snapshots(snapshot_id, embedding_model, embedding_dimensions),
        FOREIGN KEY (embedding_model, embedding_dimensions, embedding_input_hash)
            REFERENCES embeddings(model, dimensions, input_hash)
    )"""
    )
    op.create_index(
        "ix_search_fragments_address",
        "search_fragments",
        ["snapshot_id", "code", "document_scope", "article", "legal_status"],
    )
    op.create_index(
        "ix_search_fragments_fts",
        "search_fragments",
        ["search_vector"],
        postgresql_using="gin",
    )
    op.execute("""
        CREATE TABLE fragment_units (
            snapshot_id varchar(64) NOT NULL,
            fragment_id varchar(64) NOT NULL,
            legal_unit_id text NOT NULL,
            relationship text NOT NULL CHECK (relationship IN ('content', 'context')),
            PRIMARY KEY (snapshot_id, fragment_id, legal_unit_id, relationship),
            FOREIGN KEY (snapshot_id, fragment_id) REFERENCES search_fragments(snapshot_id, fragment_id) ON DELETE CASCADE,
            FOREIGN KEY (snapshot_id, legal_unit_id) REFERENCES legal_units(snapshot_id, legal_unit_id)
        )
    """)


def downgrade() -> None:
    op.drop_table("fragment_units")
    op.drop_table("search_fragments")
    op.drop_table("document_tables")
    op.drop_table("legal_units")
    op.drop_table("source_snapshots")
