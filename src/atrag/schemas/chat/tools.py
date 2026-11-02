"""Model-controlled arguments contain addresses and narrowing filters only."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from atrag.schemas.materials import SHA256, Nonempty
from atrag.schemas.retrieval import SearchFilters


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    context_tokens: int = Field(default=6000, ge=1, le=6000, strict=True)


class BlockAddress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["unit", "fragment"]
    source_id: Nonempty
    snapshot_id: SHA256
    block_id: Nonempty


class CitationAddress(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    kind: Literal["citation"]
    turn_id: UUID
    citation_id: int = Field(ge=1, strict=True)


Address = Annotated[BlockAddress | CitationAddress, Field(discriminator="kind")]


class SearchArguments(Arguments):
    query: str = Field(min_length=1, max_length=32000)
    limit: int = Field(default=5, ge=1, le=50, strict=True)
    source_ids: list[Nonempty] | None = None
    filters: SearchFilters = Field(default_factory=SearchFilters)


class UnitArguments(Arguments):
    address: Address
    start: int = Field(default=0, ge=0, strict=True)


class NeighborArguments(Arguments):
    address: Address
    before: int = Field(default=0, ge=0, le=50, strict=True)
    after: int = Field(default=1, ge=0, le=50, strict=True)


class ContextArguments(Arguments):
    address: Address
    reference_ids: list[Nonempty] = Field(default_factory=list)
    note_ids: list[Nonempty] = Field(default_factory=list)
    dependency_edges: list[tuple[Nonempty, Nonempty]] = Field(default_factory=list)
