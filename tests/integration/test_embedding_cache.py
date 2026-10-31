"""Real PostgreSQL cache/migrations; no model is invoked."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

from atrag.ingestion.services.embedding_cache import PostgresEmbeddingCache
from atrag.migrations import migrate_database
from atrag.schemas.ingestion.embeddings import EmbeddingRecord, EmbeddingSpace

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def clean_cache_schema(engine):
    # Always execute the migration being tested, including in mutation workers.
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS embeddings"))
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
    yield
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS embeddings"))
        connection.execute(text("DROP TABLE IF EXISTS alembic_version"))


def test_migration_is_idempotent_and_cache_survives_reopening(engine):
    migrate_database(engine)
    migrate_database(engine)
    space = EmbeddingSpace(model="text-embedding-3-small", dimensions=2)
    row = EmbeddingRecord(
        **space.model_dump(), input_hash="a" * 64, embedding=[0.1, 0.2]
    )
    second = row.model_copy(update={"input_hash": "e" * 64})
    with engine.begin() as connection:
        connection.execute(text("TRUNCATE embeddings"))
    with PostgresEmbeddingCache(engine).writer(space) as cache:
        assert cache.get([row.input_hash]) == {}
        cache.save([row, second])
        cache.save([row.model_copy(update={"embedding": [0.5, 1]})])
    with PostgresEmbeddingCache(engine).writer(space) as cache:
        assert cache.get([row.input_hash]) == {row.input_hash: row}
    other = space.model_copy(update={"model": "text-embedding-3-large"})
    with PostgresEmbeddingCache(engine).writer(other) as cache:
        assert cache.get([row.input_hash]) == {}
        cache.save([row.model_copy(update={"model": other.model})])
    with PostgresEmbeddingCache(engine).writer(
        space.model_copy(update={"dimensions": 3})
    ) as cache:
        assert cache.get([row.input_hash]) == {}
        cache.save(
            [
                EmbeddingRecord(
                    model=space.model,
                    dimensions=3,
                    input_hash=row.input_hash,
                    embedding=[1, 1, 1],
                )
            ]
        )
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM embeddings")) == 4
        assert (
            connection.scalar(
                text("SELECT created_at FROM embeddings LIMIT 1")
            ).utcoffset()
            is not None
        )
        assert (
            connection.scalar(
                text(
                    "SELECT vector_dims(embedding) FROM embeddings WHERE dimensions = 2 LIMIT 1"
                )
            )
            == 2
        )


def test_failed_cache_batch_rolls_back_and_releases_writer(engine):
    migrate_database(engine)
    space = EmbeddingSpace(model="text-embedding-3-small", dimensions=2)
    row = EmbeddingRecord(**space.model_dump(), input_hash="b" * 64, embedding=[1, 1])
    with PostgresEmbeddingCache(engine).writer(space) as cache:
        with pytest.raises(ValueError, match="space"):
            cache.save([row.model_copy(update={"model": "text-embedding-3-large"})])
        assert cache.get([row.input_hash]) == {}
    with (
        pytest.raises(IntegrityError),
        PostgresEmbeddingCache(engine).writer(space) as cache,
    ):
        cache.save(
            [row, row.model_copy(update={"input_hash": "c" * 64, "embedding": [1]})]
        )
    with PostgresEmbeddingCache(engine).writer(space) as cache:
        assert cache.get([row.input_hash]) == {}
        cache.save([row])


def test_concurrent_writer_waits_then_observes_committed_cache(engine):
    migrate_database(engine)
    space = EmbeddingSpace(model="text-embedding-3-small", dimensions=2)
    row = EmbeddingRecord(**space.model_dump(), input_hash="d" * 64, embedding=[1, 1])
    entering, acquired = Event(), Event()

    def second_writer():
        entering.set()
        with PostgresEmbeddingCache(engine).writer(space) as cache:
            acquired.set()
            return cache.get([row.input_hash])

    with ThreadPoolExecutor(max_workers=1) as workers:
        with PostgresEmbeddingCache(engine).writer(space) as cache:
            future = workers.submit(second_writer)
            assert entering.wait(5)
            assert not acquired.wait(0.1)
            cache.save([row])
        assert future.result(timeout=5) == {row.input_hash: row}


def test_cache_migration_can_be_reversed_without_removing_vector(engine):
    migrate_database(engine)
    config = Config()
    config.set_main_option(
        "script_location", str(Path("src/atrag/migrations").resolve())
    )
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
        assert connection.scalar(text("SELECT to_regclass('embeddings')")) is None
        assert connection.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname='vector'")
        )
    migrate_database(engine)


@pytest.mark.parametrize("values", ["'[1,1]', NULL", "NULL, now()"])
def test_cache_vector_and_creation_time_cannot_be_null(engine, values):
    migrate_database(engine)
    with pytest.raises(IntegrityError), engine.begin() as connection:
        connection.execute(
            text(
                "INSERT INTO embeddings(model, dimensions, input_hash, embedding, created_at) "
                f"VALUES ('text-embedding-3-small', 2, :hash, {values})"
            ),
            {"hash": "f" * 64},
        )
