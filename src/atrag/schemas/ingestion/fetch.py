"""Single-owner crawl checkpoints, separate from complete immutable collections."""

from dataclasses import dataclass
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from atrag.schemas.ingestion.collections import HTMLPage
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.ingestion.sources import SourceDefinition


class FetchState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    source: SourceDefinition
    pages: list[HTMLPage] = Field(default_factory=list)
    failures: dict[str, str] = Field(default_factory=dict)


@dataclass(frozen=True)
class CollectionFetch:
    run_id: UUID
    fetch_id: UUID
    snapshot: SourceSnapshot
    reused: bool
