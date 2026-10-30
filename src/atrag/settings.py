"""The single boundary for .env and process environment configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

from atrag.schemas.settings import PROJECT_ROOT, Settings


def load_settings(project_root: Path = PROJECT_ROOT) -> Settings:
    root = project_root.resolve()
    load_dotenv(dotenv_path=root / ".env", override=False)
    values = {
        field.alias: os.environ[field.alias]
        for field in Settings.model_fields.values()
        if field.alias in os.environ
    }
    data_dir = values.get("ATRAG_DATA_DIR", "data")
    if not data_dir.strip():
        raise ValueError("ATRAG_DATA_DIR must not be empty")
    values["ATRAG_DATA_DIR"] = (root / data_dir).resolve()
    return Settings.model_validate(values)
