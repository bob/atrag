"""Run packaged Alembic migrations with an already configured connection."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text


def migrate_database(engine: Engine) -> None:
    config = Config()
    config.set_main_option("script_location", str(Path(__file__).parent))
    with engine.begin() as connection:
        # Serialize concurrent migrations independently of embedding writers.
        connection.execute(text("SELECT pg_advisory_xact_lock(6174726167)"))
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
