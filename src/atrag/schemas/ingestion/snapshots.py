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

from atrag.schemas.ingestion.sources import SourceDefinition


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    schema_version: Literal[1] = 1
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    local_path: Path
    size_bytes: PositiveInt
    ingested_at: AwareDatetime
    source: SourceDefinition

    @model_validator(mode="after")
    def consistent_identity(self) -> Self:
        if self.snapshot_id != self.source_sha256:
            raise ValueError("Snapshot identity must equal the source checksum")
        if self.local_path != Path("raw") / self.snapshot_id / "source.pdf":
            raise ValueError("Snapshot path must match its content address")
        return self


@dataclass(frozen=True)
class SnapshotImport:
    run_id: UUID
    snapshot: SourceSnapshot
    reused: bool
