"""Publication selects complete immutable materials; turns keep their own release."""

from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from sqlalchemy import insert, select, text
from sqlalchemy.exc import IntegrityError

from atrag.corpus import publish_corpus, resolve_corpus, resolve_turn_corpus, start_turn
from atrag.db_models import ChatConversationRow, ChatUserRow, SnapshotRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package, load_package
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def schema(empty_corpus_schema, engine):
    migrate_database(engine)


@pytest.fixture
def material(prepared_document, tmp_path, engine):
    data, parsed, chunked, embedded = prepared_document
    package = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    indexed = index_package(package, engine, tmp_path / "server")
    return package, indexed.snapshot_id, load_package(package).snapshot.source.source_id


def test_empty_corpus_is_explicit_and_turn_is_pinned(engine):
    with engine.begin() as connection:
        initial = resolve_corpus(connection)
        assert initial.members == ()
        user, conversation, turn = uuid4(), uuid4(), uuid4()
        connection.execute(
            insert(ChatUserRow).values(id=user, issuer="g", subject="s", email="a@b")
        )
        connection.execute(
            insert(ChatConversationRow).values(id=conversation, user_id=user, title="")
        )
        start_turn(
            connection,
            turn_id=turn,
            conversation_id=conversation,
            question="Question?",
            language="en",
        )
    with engine.connect() as connection:
        assert resolve_turn_corpus(connection, turn) == initial
    assert publish_corpus(engine, {}) == initial


def test_publication_replacement_removal_and_old_turn(
    engine, material, tmp_path, prepared_document
):
    package, first, source = material
    # A second document is independent even if its bytes are the same.
    from fixtures.prepared import prepare_document

    data2, parsed2, chunked2, embedded2 = prepare_document(
        tmp_path / "second", source_id="second"
    )
    second_package = build_package(
        parsed2, chunked2, embedded2, data2, tmp_path / "second-package"
    )
    second = index_package(second_package, engine, tmp_path / "server2").snapshot_id
    release = publish_corpus(engine, {"second": second, source: first})
    assert dict(release.members) == {source: first, "second": second}
    assert publish_corpus(engine, {"second": second, source: first}) == release
    with engine.begin() as connection:
        user, conversation, turn = uuid4(), uuid4(), uuid4()
        connection.execute(
            insert(ChatUserRow).values(id=user, issuer="g", subject="s", email="a@b")
        )
        connection.execute(
            insert(ChatConversationRow).values(id=conversation, user_id=user, title="")
        )
        start_turn(
            connection,
            turn_id=turn,
            conversation_id=conversation,
            question="Question?",
            language="en",
        )
    # Same PDF, changed preparation parameters; both versions survive.
    from fixtures.prepared import prepare_revision

    newer_package = prepare_revision(prepared_document, tmp_path / "revision")
    newer = index_package(newer_package, engine, tmp_path / "server").snapshot_id
    assert newer != first
    updated = publish_corpus(engine, {source: newer})
    assert dict(updated.members) == {source: newer, "second": second}
    index_package(package, engine, tmp_path / "server")
    with engine.begin() as connection:
        assert resolve_corpus(connection) == updated
        assert resolve_turn_corpus(connection, turn) == release
        start_turn(
            connection,
            turn_id=turn,
            conversation_id=conversation,
            question="Question?",
            language="en",
        )
        assert resolve_turn_corpus(connection, turn) == release
        next_turn = uuid4()
        start_turn(
            connection,
            turn_id=next_turn,
            conversation_id=conversation,
            question="Next?",
            language="en",
        )
        assert resolve_turn_corpus(connection, next_turn) == updated
    removed = publish_corpus(engine, {source: None})
    assert dict(removed.members) == {"second": second}
    with engine.connect() as connection:
        assert resolve_corpus(connection, release.release_id) == release
        assert (
            connection.scalar(
                select(SnapshotRow.snapshot_id).where(SnapshotRow.snapshot_id == first)
            )
            == first
        )
    for statement in (
        "UPDATE source_snapshots SET document_title = 'changed' WHERE snapshot_id = :id",
        "DELETE FROM source_snapshots WHERE snapshot_id = :id",
        "UPDATE legal_units SET text_original = 'changed' WHERE snapshot_id = :id",
        "DELETE FROM corpus_members WHERE release_id = :release",
        "UPDATE corpus_releases SET release_id = :id WHERE release_id = :release",
    ):
        with pytest.raises(IntegrityError), engine.begin() as connection:
            connection.execute(
                text(statement), {"id": first, "release": release.release_id}
            )


def test_failed_and_competing_publications(engine, material):
    _, snapshot, source = material
    with pytest.raises(ValueError, match="ready"):
        publish_corpus(engine, {source: "f" * 64})
    with pytest.raises(ValueError, match="source"):
        publish_corpus(engine, {"wrong": snapshot})
    with engine.connect() as connection:
        assert resolve_corpus(connection).members == ()
    # Both writers start from the single locked pointer and merge their changes.
    with ThreadPoolExecutor(max_workers=2) as workers:
        results = list(
            workers.map(lambda _: publish_corpus(engine, {source: snapshot}), range(2))
        )
    assert results[0] == results[1]


def test_publication_failure_rolls_back_whole_composition(engine, material):
    from sqlalchemy import event

    _, snapshot, source = material
    with engine.connect() as connection:
        before = resolve_corpus(connection)

    def fail_pointer(connection, cursor, statement, parameters, context, many):
        if statement.startswith("UPDATE corpus_state"):
            with engine.connect() as reader:
                assert resolve_corpus(reader) == before
            raise RuntimeError("publication failure")

    event.listen(engine, "before_cursor_execute", fail_pointer)
    try:
        with pytest.raises(RuntimeError, match="publication failure"):
            publish_corpus(engine, {source: snapshot})
    finally:
        event.remove(engine, "before_cursor_execute", fail_pointer)
    with engine.connect() as connection:
        assert resolve_corpus(connection) == before
        assert connection.scalar(text("SELECT count(*) FROM corpus_releases")) == 1
        assert connection.scalar(text("SELECT count(*) FROM corpus_members")) == 0


def test_unready_and_incompatible_materials_cannot_be_published(
    engine, material, tmp_path
):
    import json

    from fixtures.prepared import prepare_document, refresh_manifest

    _, snapshot, source = material
    publish_corpus(engine, {source: snapshot})
    data, parsed, chunked, embedded = prepare_document(
        tmp_path / "other", source_id="other"
    )
    vectors = embedded.parent / "embeddings.jsonl"
    vectors.write_text(
        vectors.read_text().replace("text-embedding-3-small", "text-embedding-3-large")
    )
    manifest = json.loads(embedded.read_bytes())
    manifest["model"] = "text-embedding-3-large"
    embedded.write_text(json.dumps(manifest))
    refresh_manifest(embedded)
    package = build_package(parsed, chunked, embedded, data, tmp_path / "large")
    other = index_package(package, engine, tmp_path / "server2").snapshot_id
    with pytest.raises(ValueError, match="embedding spaces"):
        publish_corpus(engine, {"other": other})
    # A partially present version is not ready, even though its header exists.
    with engine.begin() as connection:
        connection.execute(
            text("DELETE FROM search_fragments WHERE snapshot_id = :id"), {"id": other}
        )
    with pytest.raises(ValueError, match="ready"):
        publish_corpus(engine, {"other": other})
    with engine.connect() as connection:
        assert dict(resolve_corpus(connection).members) == {source: snapshot}


def test_unknown_release_and_turn_are_not_silently_rebound(engine):
    with engine.connect() as connection:
        with pytest.raises(ValueError, match="Unknown"):
            resolve_corpus(connection, "f" * 64)
        with pytest.raises(ValueError, match="no saved"):
            resolve_turn_corpus(connection, uuid4())


def test_competing_additions_preserve_both_documents(engine, material, tmp_path):
    from threading import Barrier

    from fixtures.prepared import prepare_document

    _, first, source = material
    data, parsed, chunked, embedded = prepare_document(
        tmp_path / "second", source_id="second"
    )
    package = build_package(parsed, chunked, embedded, data, tmp_path / "pkg2")
    second = index_package(package, engine, tmp_path / "server2").snapshot_id
    barrier = Barrier(2)

    def publish(changes):
        barrier.wait(timeout=5)
        return publish_corpus(engine, changes)

    with ThreadPoolExecutor(max_workers=2) as workers:
        list(workers.map(publish, [{source: first}, {"second": second}]))
    with engine.connect() as connection:
        assert dict(resolve_corpus(connection).members) == {
            source: first,
            "second": second,
        }


def test_turn_uuid_cannot_be_reassigned_or_change_release(engine):
    from sqlalchemy import update

    from atrag.db_models import ChatTurnRow

    user, conversation, turn = uuid4(), uuid4(), uuid4()
    with engine.begin() as connection:
        connection.execute(
            insert(ChatUserRow).values(id=user, issuer="g", subject="s", email="a@b")
        )
        connection.execute(
            insert(ChatConversationRow).values(id=conversation, user_id=user, title="")
        )
        composition = start_turn(
            connection,
            turn_id=turn,
            conversation_id=conversation,
            question="First?",
            language="en",
        )
    with (
        pytest.raises(ValueError, match="another request"),
        engine.begin() as connection,
    ):
        start_turn(
            connection,
            turn_id=turn,
            conversation_id=conversation,
            question="Other?",
            language="en",
        )
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            update(ChatTurnRow)
            .where(ChatTurnRow.id == turn)
            .values(corpus_release_id=None)
        )
    with engine.connect() as connection:
        assert resolve_turn_corpus(connection, turn) == composition


def test_cli_publishes_and_reads_saved_release(
    engine, material, tmp_path, database_dsn, monkeypatch, capsys
):
    import json

    from atrag.cli import main

    _, snapshot, source = material
    monkeypatch.setenv("DATABASE_URL", database_dsn)
    changes = tmp_path / "changes.json"
    changes.write_text(json.dumps({source: snapshot}))
    main(["corpus", "publish", "--changes", str(changes)])
    output = json.loads(capsys.readouterr().out)
    assert output["members"] == {source: snapshot}
    main(["corpus", "show", "--release", output["release_id"]])
    assert json.loads(capsys.readouterr().out) == output


def test_downgrade_cannot_erase_prepared_versions_or_pinned_turns(engine, material):
    from pathlib import Path

    from alembic import command
    from alembic.config import Config

    config = Config()
    config.set_main_option(
        "script_location", str(Path("src/atrag/migrations").resolve())
    )
    with (
        pytest.raises(ValueError, match="Cannot discard"),
        engine.begin() as connection,
    ):
        config.attributes["connection"] = connection
        command.downgrade(config, "0005_chat_provenance")
    with engine.connect() as connection:
        assert (
            connection.scalar(text("SELECT version_num FROM alembic_version"))
            == "0007_rag_evidence"
        )


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO corpus_members VALUES (:release, :source, :snapshot)",
        "INSERT INTO legal_units SELECT snapshot_id, legal_unit_id || '_new', parent_id, instrument_id, code, document_scope, kind, role, chapter, section, article, paragraph, subparagraph, legal_status, unit_title, citation_label, page_start, page_end, text_original, text_normalized, payload FROM legal_units WHERE snapshot_id = :snapshot LIMIT 1",
        "UPDATE corpus_members SET snapshot_id = :snapshot WHERE release_id = :release",
    ],
)
def test_published_members_and_blocks_cannot_be_extended_or_rewritten(
    engine, material, statement
):
    _, snapshot, source = material
    release = publish_corpus(engine, {source: snapshot})
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(statement),
            {"release": release.release_id, "source": source, "snapshot": snapshot},
        )


def test_migration_keeps_legacy_cirs_and_protects_its_saved_sources(engine, material):
    from fixtures.prepared import index_legacy_document

    from atrag.db_models import ChatTurnRow

    package, _, _ = material
    legacy = index_legacy_document(package, engine)
    user, conversation, turn = uuid4(), uuid4(), uuid4()
    with engine.begin() as connection:
        connection.execute(
            insert(ChatUserRow).values(id=user, issuer="g", subject="s", email="a@b")
        )
        connection.execute(
            insert(ChatConversationRow).values(id=conversation, user_id=user, title="")
        )
        connection.execute(
            insert(ChatTurnRow).values(
                id=turn,
                conversation_id=conversation,
                question="Old?",
                language="en",
                legacy_snapshot_id=legacy,
            )
        )
    migrate_database(engine)
    with engine.connect() as connection:
        assert (
            connection.scalar(
                select(SnapshotRow.snapshot_id).where(SnapshotRow.snapshot_id == legacy)
            )
            == legacy
        )
        assert resolve_corpus(connection).members == ()
        with pytest.raises(ValueError, match="no saved"):
            resolve_turn_corpus(connection, turn)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "UPDATE legal_units SET text_original = 'changed' WHERE snapshot_id = :id"
            ),
            {"id": legacy},
        )
