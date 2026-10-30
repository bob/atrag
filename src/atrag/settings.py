"""The single boundary for .env and process environment configuration."""

import os
from pathlib import Path
from typing import Literal, Self

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    SecretStr,
    field_validator,
    model_validator,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SupportedLanguage = Literal["ru", "en", "pt"]
EMBEDDING_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


class Settings(BaseModel):
    model_config = ConfigDict(
        frozen=True, populate_by_name=True, hide_input_in_errors=True
    )

    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    embedding_model: Literal["text-embedding-3-small", "text-embedding-3-large"] = (
        Field(default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL")
    )
    embedding_dimensions: PositiveInt = Field(
        default=1536, alias="OPENAI_EMBEDDING_DIMENSIONS"
    )
    database_url: SecretStr | None = Field(default=None, alias="DATABASE_URL")
    data_dir: Path = Field(default=PROJECT_ROOT / "data", alias="ATRAG_DATA_DIR")
    embedding_batch_size: PositiveInt = Field(
        default=32, alias="ATRAG_EMBEDDING_BATCH_SIZE"
    )
    chunk_target_tokens: PositiveInt = Field(
        default=700, alias="ATRAG_CHUNK_TARGET_TOKENS"
    )
    chunk_max_tokens: PositiveInt = Field(default=1200, alias="ATRAG_CHUNK_MAX_TOKENS")
    rrf_k: PositiveInt = Field(default=60, alias="ATRAG_RRF_K")

    @field_validator("openai_api_key", "database_url", mode="before")
    @classmethod
    def empty_secret(cls, value):
        return None if value == "" else value

    @field_validator("database_url")
    @classmethod
    def canonical_database_url(cls, value: SecretStr | None) -> SecretStr | None:
        if value is None:
            return None
        try:
            url = make_url(value.get_secret_value())
        except ValueError, ArgumentError:
            raise ValueError("DATABASE_URL must be a valid PostgreSQL URL") from None
        if url.drivername != "postgresql" or not url.database:
            raise ValueError("DATABASE_URL must use postgresql:// and name a database")
        return value

    @model_validator(mode="after")
    def consistent_limits(self) -> Self:
        if self.chunk_target_tokens > self.chunk_max_tokens:
            raise ValueError("Chunk target must not exceed chunk maximum")
        if self.embedding_dimensions > EMBEDDING_DIMENSIONS[self.embedding_model]:
            raise ValueError(
                "Embedding dimensions exceed the selected model's dimensions"
            )
        return self


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
