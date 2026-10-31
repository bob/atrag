"""Connections only to an explicitly selected disposable PostgreSQL database."""

import os

import pytest
from psycopg.conninfo import conninfo_to_dict

from atrag.db import create_db_engine
from atrag.schemas.settings import Settings


@pytest.fixture
def empty_corpus_schema(engine):
    from sqlalchemy import text

    from atrag.db_models import Base

    def clear():
        with engine.begin() as connection:
            Base.metadata.drop_all(connection)
            connection.execute(text("DROP TABLE IF EXISTS alembic_version"))

    clear()
    yield
    clear()


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
