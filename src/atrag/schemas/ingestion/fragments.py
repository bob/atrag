"""Searchable text with explicit legal context and reproducible chunk limits."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, Field, PositiveInt, model_validator

from atrag.schemas.ingestion.extraction import Artifact, ExtractedTable
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import LegalUnit, SourceLocator

CHUNKER_VERSION = "1"


class ChunkParameters(Artifact):
    target_tokens: PositiveInt = 700
    max_tokens: PositiveInt = 1200
    token_encoding: Literal["cl100k_base"] = "cl100k_base"

    @model_validator(mode="after")
    def consistent_limits(self) -> Self:
        if self.target_tokens > self.max_tokens:
            raise ValueError("Chunk target must not exceed chunk maximum")
        return self


class FragmentContext(Artifact):
    legal_unit_id: str
    text_original: str
    locators: list[SourceLocator]


class SearchFragment(LegalUnit):
    fragment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunker_version: str = CHUNKER_VERSION
    parser_version: str
    parameters: ChunkParameters
    source: SourceDefinition
    legal_unit_ids: list[str]
    context_unit_ids: list[str]
    context: list[FragmentContext]
    embedding_text: str
    embedding_input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    token_count: PositiveInt
    tables: list[ExtractedTable]
    review_status: Literal["ready", "needs_review"]


class FragmentOmission(Artifact):
    legal_unit_id: str
    reason: Literal["over_token_limit", "unverified_table"]
    token_count: int
    table_ids: list[str]


class ChunkResult(Artifact):
    fragments: list[SearchFragment]
    omissions: list[FragmentOmission]


class ParsedManifest(BaseModel):
    stage: Literal["parse"]
    status: Literal["succeeded"]
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_version: str
    output_sha256: dict[str, str]


class ParsedReport(BaseModel):
    excluded_block_count: int = Field(ge=0)
    extraction_issues: list[dict]


@dataclass(frozen=True)
class ChunkRun:
    run_id: UUID
    directory: Path
    fragment_count: int
    review_status: str
