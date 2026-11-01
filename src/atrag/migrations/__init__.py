"""Run packaged Alembic migrations with an already configured connection."""

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from time import sleep

from alembic import command
from alembic.config import Config
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.rows import DictRow, dict_row
from sqlalchemy import Engine

MIGRATION_LOCK = 6174726167


def migrate_database(engine: Engine) -> None:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent))
    with _migration_connection(engine) as checkpoint_connection:
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        # CONCURRENTLY needs autocommit, after the Alembic transaction commits.
        PostgresSaver(checkpoint_connection).setup()


@contextmanager
def _migration_connection(engine: Engine) -> Generator[Connection[DictRow]]:
    arguments, options = engine.dialect.create_connect_args(engine.url)
    with Connection[DictRow].connect(
        *arguments, **options, autocommit=True, row_factory=dict_row
    ) as connection:
        # A blocking lock query holds a snapshot that CONCURRENTLY may wait for.
        # Poll in autocommit so waiting migrators leave no transaction open.
        while connection.execute(
            "SELECT pg_try_advisory_lock(%s) AS acquired", (MIGRATION_LOCK,)
        ).fetchone() != {"acquired": True}:
            sleep(0.05)
        try:
            yield connection
        finally:
            connection.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK,))
