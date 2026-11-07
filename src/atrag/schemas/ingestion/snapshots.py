"""Persisted shapes for immutable source snapshots."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PositiveInt,
    model_validator,
)

from atrag.schemas.ingestion.collections import HTMLCollection
from atrag.schemas.ingestion.sources import SourceDefinition, SourceFormat


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    schema_version: Literal[1, 2] = 1
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    local_path: Path
    size_bytes: PositiveInt
    ingested_at: AwareDatetime
    source: SourceDefinition
    collection: HTMLCollection | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        if self.snapshot_id != (self.preparation_sha256 or self.source_sha256):
            raise ValueError(
                "Snapshot identity must equal the source checksum or preparation checksum"
            )
        expected = (
            Path("raw")
            / self.source_sha256
            / self.source.source_format.snapshot_filename
        )
        if self.local_path != expected:
            raise ValueError("Snapshot path must match its content address")
        return self

    @model_validator(mode="after")
    def consistent_format(self) -> Self:
        if self.source.source_format == SourceFormat.PDF:
            if self.collection is not None:
                raise ValueError("PDF snapshot cannot contain an HTML collection")
        else:
            self._check_collection()
        return self

    def _check_collection(self) -> None:
        if self.schema_version != 2 or self.collection is None:
            raise ValueError("HTML snapshot requires a version 2 collection manifest")
        if (
            self.collection.source_sha256 != self.source_sha256
            or self.collection.size_bytes != self.size_bytes
            or self.collection.catalog_url != self.source.source_url
        ):
            raise ValueError("HTML collection does not match snapshot provenance")


@dataclass(frozen=True)
class SnapshotImport:
    run_id: UUID
    snapshot: SourceSnapshot
    reused: bool
