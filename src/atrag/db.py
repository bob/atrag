"""Synchronous SQLAlchemy connection setup from the canonical PostgreSQL DSN."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url

from atrag.schemas.settings import Settings


def create_db_engine(settings: Settings) -> Engine:
    if settings.database_url is None:
        raise ValueError("DATABASE_URL is required for database operations")
    url = make_url(settings.database_url.get_secret_value()).set(
        drivername="postgresql+psycopg"
    )
    return create_engine(url, pool_pre_ping=True, hide_parameters=True)
