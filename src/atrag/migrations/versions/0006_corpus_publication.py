"""Immutable prepared versions and explicitly published corpus compositions.

Revision ID: 0006_corpus_publication
Revises: 0005_chat_provenance
"""

import sqlalchemy as sa
from alembic import op

revision = "0006_corpus_publication"
down_revision = "0005_chat_provenance"
branch_labels = None
depends_on = None

EMPTY_RELEASE = "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"
MATERIAL_TABLES = (
    "source_snapshots",
    "legal_units",
    "search_fragments",
    "document_tables",
    "fragment_units",
)


def upgrade() -> None:
    op.execute("""
        ALTER TABLE source_snapshots DROP CONSTRAINT source_snapshots_check;
        ALTER TABLE source_snapshots ADD COLUMN preparation_sha256 varchar(64)
            CHECK (preparation_sha256 ~ '^[0-9a-f]{64}$' AND preparation_sha256 = snapshot_id);
        ALTER TABLE source_snapshots ADD CONSTRAINT source_sha256_format
            CHECK (source_sha256 ~ '^[0-9a-f]{64}$');
        CREATE TABLE source_documents (source_id text PRIMARY KEY CHECK (btrim(source_id) <> ''));
        INSERT INTO source_documents SELECT DISTINCT source_id FROM source_snapshots;
        ALTER TABLE source_snapshots ADD CONSTRAINT snapshot_source
            FOREIGN KEY (source_id) REFERENCES source_documents(source_id);
        ALTER TABLE source_snapshots ADD CONSTRAINT snapshot_source_identity UNIQUE (source_id, snapshot_id);
        CREATE TABLE corpus_releases (
            release_id varchar(64) PRIMARY KEY CHECK (release_id ~ '^[0-9a-f]{64}$')
        );
        CREATE TABLE corpus_members (
            release_id varchar(64) NOT NULL REFERENCES corpus_releases(release_id) DEFERRABLE INITIALLY DEFERRED,
            source_id text NOT NULL,
            snapshot_id varchar(64) NOT NULL,
            PRIMARY KEY (release_id, source_id),
            FOREIGN KEY (source_id, snapshot_id) REFERENCES source_snapshots(source_id, snapshot_id)
        );
        CREATE TABLE corpus_state (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            release_id varchar(64) NOT NULL REFERENCES corpus_releases(release_id)
        );
    """)
    op.execute(
        sa.text("INSERT INTO corpus_releases VALUES (:id)").bindparams(id=EMPTY_RELEASE)
    )
    op.execute(
        sa.text("INSERT INTO corpus_state VALUES (true, :id)").bindparams(
            id=EMPTY_RELEASE
        )
    )
    op.add_column(
        "chat_turns",
        sa.Column(
            "corpus_release_id",
            sa.String(64),
            sa.ForeignKey("corpus_releases.release_id", name="turn_corpus_release"),
        ),
    )
    op.execute("""
        CREATE OR REPLACE FUNCTION protect_corpus_composition() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'INSERT' AND TG_TABLE_NAME = 'corpus_members' THEN
                IF NOT EXISTS (SELECT FROM corpus_releases WHERE release_id = NEW.release_id) THEN
                    RETURN NEW;
                END IF;
            END IF;
            RAISE EXCEPTION 'Published corpus compositions are immutable' USING ERRCODE = '23514';
        END $$;
        CREATE TRIGGER immutable_release BEFORE UPDATE OR DELETE ON corpus_releases
            FOR EACH ROW EXECUTE FUNCTION protect_corpus_composition();
        CREATE TRIGGER immutable_members BEFORE INSERT OR UPDATE OR DELETE ON corpus_members
            FOR EACH ROW EXECUTE FUNCTION protect_corpus_composition();
        CREATE OR REPLACE FUNCTION protect_published_material() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP <> 'INSERT' AND (EXISTS (SELECT FROM corpus_members WHERE snapshot_id = OLD.snapshot_id) OR EXISTS (SELECT FROM chat_turns WHERE legacy_snapshot_id = OLD.snapshot_id)) THEN
                RAISE EXCEPTION 'Published materials are immutable' USING ERRCODE = '23514';
            END IF;
            IF TG_OP <> 'DELETE' AND (EXISTS (SELECT FROM corpus_members WHERE snapshot_id = NEW.snapshot_id) OR EXISTS (SELECT FROM chat_turns WHERE legacy_snapshot_id = NEW.snapshot_id)) THEN
                RAISE EXCEPTION 'Published materials are immutable' USING ERRCODE = '23514';
            END IF;
            IF TG_OP = 'DELETE' THEN RETURN OLD; END IF;
            RETURN NEW;
        END $$;
        CREATE OR REPLACE FUNCTION protect_turn_corpus() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF NEW.corpus_release_id IS DISTINCT FROM OLD.corpus_release_id THEN
                RAISE EXCEPTION 'A turn keeps its original corpus' USING ERRCODE = '23514';
            END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER immutable_turn_corpus BEFORE UPDATE ON chat_turns
            FOR EACH ROW EXECUTE FUNCTION protect_turn_corpus();
    """)
    for table in MATERIAL_TABLES:
        op.execute(
            f"CREATE TRIGGER immutable_published_material BEFORE INSERT OR UPDATE OR DELETE ON {table} FOR EACH ROW EXECUTE FUNCTION protect_published_material()"
        )


def downgrade() -> None:
    # Do not erase a turn's saved composition or force prepared IDs back to PDF hashes.
    if op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS (SELECT FROM chat_turns WHERE corpus_release_id IS NOT NULL) OR EXISTS (SELECT FROM source_snapshots WHERE snapshot_id <> source_sha256)"
        )
    ):
        raise ValueError("Cannot discard prepared versions or turn corpus provenance")
    op.execute("DROP TRIGGER immutable_turn_corpus ON chat_turns")
    op.drop_column("chat_turns", "corpus_release_id")
    op.drop_table("corpus_state")
    op.drop_table("corpus_members")
    op.drop_table("corpus_releases")
    for table in MATERIAL_TABLES:
        op.execute(f"DROP TRIGGER immutable_published_material ON {table}")
    op.execute("""
        DROP FUNCTION protect_turn_corpus();
        DROP FUNCTION protect_corpus_composition();
        DROP FUNCTION protect_published_material();
        ALTER TABLE source_snapshots DROP CONSTRAINT snapshot_source;
        ALTER TABLE source_snapshots DROP CONSTRAINT snapshot_source_identity;
        ALTER TABLE source_snapshots DROP CONSTRAINT source_sha256_format;
        ALTER TABLE source_snapshots DROP COLUMN preparation_sha256;
        ALTER TABLE source_snapshots ADD CONSTRAINT source_snapshots_check CHECK (source_sha256 = snapshot_id);
    """)
    op.drop_table("source_documents")
