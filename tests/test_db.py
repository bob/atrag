import pytest

from atrag.db import create_db_engine
from atrag.settings import Settings


def test_engine_preserves_dsn_options_and_selects_psycopg():
    dsn = "postgresql://user:p%40ss@localhost:5432/atrag?sslmode=require"
    settings = Settings(database_url=dsn)
    engine = create_db_engine(settings)
    try:
        assert engine.url.drivername == "postgresql+psycopg"
        assert engine.url.password == "p@ss"
        assert engine.url.query["sslmode"] == "require"
        assert settings.database_url.get_secret_value() == dsn
        assert "p@ss" not in repr(engine)
        assert engine.hide_parameters is True
    finally:
        engine.dispose()


def test_database_is_required_only_for_database_operations():
    with pytest.raises(ValueError, match="DATABASE_URL"):
        create_db_engine(Settings())
