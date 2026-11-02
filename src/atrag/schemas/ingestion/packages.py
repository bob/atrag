"""Portable, checked input to corpus indexing; one embedding space per snapshot."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from atrag.schemas.ingestion.embeddings import EmbeddingSpace, FragmentEmbedding
from atrag.schemas.ingestion.extraction import ExtractedTable
from atrag.schemas.ingestion.fragments import SearchFragment
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.ingestion.structure import LegalUnit

SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class PackageManifest(EmbeddingSpace):
    schema_version: Literal[1, 2] = 2
    dimensions: Literal[1536] = 1536
    snapshot_id: SHA256
    raw_snapshot_id: SHA256 | None = None
    preparation_version: Literal["1"] = "1"
    parser_version: str
    chunker_version: str
    fragment_count: int = Field(gt=0)
    files: dict[str, SHA256]

    @field_validator("files")
    @classmethod
    def relative_paths(cls, files):
        for name in files:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError("Package file path must be relative and contained")
            if str(path) != name or name == ".":
                raise ValueError("Package file path must be canonical")
        return files


class CompletedStage(BaseModel):
    stage: Literal["parse", "chunk", "embed"]
    status: Literal["succeeded"]
    snapshot_id: SHA256
    input_sha256: dict[str, str] = Field(default_factory=dict)
    output_sha256: dict[str, str]
    parser_version: str | None = None
    chunker_version: str | None = None
    model: str | None = None
    dimensions: int | None = None
    fragment_count: int | None = None


class ChunkQuality(BaseModel):
    snapshot_id: SHA256
    fragment_count: int = Field(gt=0)
    review_status: Literal["ready", "needs_review"]
    extraction_issues: list[dict] = Field(default_factory=list)
    unresolved_references: list[dict] = Field(default_factory=list)
    omissions: list[dict] = Field(default_factory=list)
    uncovered_unit_ids: list[str] = Field(default_factory=list)
    table_issues: list[str] = Field(default_factory=list)
    structure_issues: list[dict] = Field(default_factory=list)


@dataclass(frozen=True)
class PreparedDocument:
    manifest: PackageManifest
    snapshot: SourceSnapshot
    units: list[LegalUnit]
    tables: list[ExtractedTable]
    fragments: list[SearchFragment]
    embeddings: list[FragmentEmbedding]
    reports: dict[str, dict]
    files: dict[str, bytes]


@dataclass(frozen=True)
class IndexResult:
    snapshot_id: str
    fragment_count: int
    directory: Path
