"""Embedding identity, validated vectors and the completed chunk boundary."""

from dataclasses import dataclass
from pathlib import Path
from struct import pack, unpack
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator

from atrag.schemas.ingestion.extraction import Artifact
from atrag.schemas.settings import EmbeddingModel, validate_embedding_dimensions

# pgvector stores float32; reject overflow before sending a vector to PostgreSQL.
VectorValue = Annotated[
    float,
    Field(
        strict=True,
        allow_inf_nan=False,
        le=3.4028234663852886e38,
        ge=-3.4028234663852886e38,
    ),
]


class EmbeddingSpace(Artifact):
    model: EmbeddingModel
    dimensions: PositiveInt

    @model_validator(mode="after")
    def supported_dimensions(self) -> Self:
        validate_embedding_dimensions(self.model, self.dimensions)
        return self


class EmbeddingRecord(EmbeddingSpace):
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding: list[VectorValue]

    @field_validator("embedding")
    @classmethod
    def float32_values(cls, values: list[float]) -> list[float]:
        return [unpack("f", pack("f", value))[0] for value in values]

    @model_validator(mode="after")
    def usable_vector(self) -> Self:
        if len(self.embedding) != self.dimensions or not any(self.embedding):
            raise ValueError(
                "Embedding must have the expected dimensions and nonzero norm"
            )
        return self


class FragmentEmbedding(EmbeddingRecord):
    schema_version: Literal[1] = 1
    snapshot_id: str
    fragment_id: str


class ChunkManifest(BaseModel):
    stage: Literal["chunk"]
    status: Literal["succeeded"]
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    fragment_count: PositiveInt
    output_sha256: dict[str, str]


class APIUsage(BaseModel):
    prompt_tokens: int = Field(ge=0, strict=True)
    total_tokens: int = Field(ge=0, strict=True)


class APIEmbedding(BaseModel):
    index: int = Field(ge=0, strict=True)
    embedding: list[VectorValue]


class APIResponse(BaseModel):
    model: str
    data: list[APIEmbedding]


@dataclass(frozen=True)
class EmbeddingInput:
    input_hash: str
    text: str
    token_count: int


@dataclass(frozen=True)
class EmbeddingRun:
    run_id: UUID
    directory: Path
    fragment_count: int
