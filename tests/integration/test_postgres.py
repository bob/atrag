"""Run only against a disposable database named atrag_test_* with vector enabled."""

import os

import psycopg
import pytest
from langgraph.checkpoint.postgres import PostgresSaver
from pgvector.sqlalchemy import Vector
from psycopg.conninfo import conninfo_to_dict
from psycopg_pool import ConnectionPool
from sqlalchemy import Column, Integer, MetaData, Table, select, text

from atrag.db import create_db_engine
from atrag.schemas.settings import Settings

pytestmark = pytest.mark.integration


@pytest.fixture
def database_dsn():
    dsn = os.environ.get("ATRAG_TEST_DATABASE_URL")
    if not dsn:
        pytest.skip("Set ATRAG_TEST_DATABASE_URL to a disposable local database")
    if not conninfo_to_dict(dsn)["dbname"].startswith("atrag_test_"):
        pytest.fail("Integration tests require an atrag_test_* database")
    return dsn


@pytest.fixture
def engine(database_dsn):
    engine = create_db_engine(Settings(database_url=database_dsn))
    yield engine
    engine.dispose()


def test_sqlalchemy_psycopg_vector_roundtrip_and_rollback(engine):
    items = Table(
        "vector_probe",
        MetaData(),
        Column("id", Integer, primary_key=True),
        Column("embedding", Vector(3)),
        prefixes=["TEMPORARY"],
    )
    with engine.connect() as connection:
        transaction = connection.begin()
        items.create(connection)
        connection.execute(
            items.insert(),
            [
                {"id": 1, "embedding": [1, 0, 0]},
                {"id": 2, "embedding": [0, 1, 0]},
            ],
        )
        nearest = select(items.c.id).order_by(
            items.c.embedding.cosine_distance([1, 0, 0])
        )
        assert connection.scalars(nearest).all() == [1, 2]
        assert connection.scalar(
            text("SELECT extversion FROM pg_extension WHERE extname='vector'")
        )
        transaction.rollback()
        assert (
            connection.scalar(text("SELECT to_regclass('pg_temp.vector_probe')"))
            is None
        )


def test_pool_recovers_after_its_idle_connection_is_terminated(engine, database_dsn):
    with engine.connect() as connection:
        old_pid = connection.scalar(text("SELECT pg_backend_pid()"))
    with psycopg.connect(database_dsn, autocommit=True) as control:
        assert control.execute(
            "SELECT pg_terminate_backend(%s)", (old_pid,)
        ).fetchone()[0]
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT 1")) == 1
        assert connection.scalar(text("SELECT pg_backend_pid()")) != old_pid


def test_checkpointer_uses_separate_psycopg_pool(database_dsn):
    with ConnectionPool(
        database_dsn,
        min_size=1,
        max_size=2,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        pool.wait(timeout=5)
        saver = PostgresSaver(pool)
        saver.setup()
        assert saver.get_tuple({"configurable": {"thread_id": "step-1-probe"}}) is None
