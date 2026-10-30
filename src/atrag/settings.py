"""The single boundary for .env and process environment configuration."""

import os
from pathlib import Path

from dotenv.main import DotEnv

from atrag.schemas.settings import PROJECT_ROOT, Settings


def _dotenv_values(path: Path) -> dict[str, str | None]:
    if os.environ.get("PYTHON_DOTENV_DISABLED", "").casefold() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
    }:
        return {}
    # Keep load_dotenv(override=False) expansion without writing to os.environ.
    return DotEnv(path, override=False, encoding="utf-8").dict()


def load_settings(project_root: Path = PROJECT_ROOT) -> Settings:
    root = project_root.resolve()
    environment = _dotenv_values(root / ".env") | dict(os.environ)
    values = {
        field.alias: environment[field.alias]
        for field in Settings.model_fields.values()
        if environment.get(field.alias) is not None
    }
    data_dir = values.get("ATRAG_DATA_DIR", "data")
    if not data_dir.strip():
        raise ValueError("ATRAG_DATA_DIR must not be empty")
    values["ATRAG_DATA_DIR"] = (root / data_dir).resolve()
    return Settings.model_validate(values)
