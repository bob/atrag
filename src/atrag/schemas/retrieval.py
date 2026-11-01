"""Language-independent source filters and original Portuguese search evidence."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
)

from atrag.schemas.ingestion.extraction import ExtractedTable
from atrag.schemas.ingestion.fragments import SearchFragment
from atrag.schemas.ingestion.packages import SHA256
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import LegalUnit, SourceLocator
from atrag.schemas.settings import SupportedLanguage


def _unique_page_locators(locators):
    unique = {}
    for locator in locators:
        labels = locator.get("printed_page_labels")
        key = (locator.get("page_number"), None if labels is None else tuple(labels))
        unique.setdefault(key, locator)
    return list(unique.values())


def _compact_pdf_locations(value):
    if isinstance(value, dict):
        compact = {
            key: _compact_pdf_locations(item)
            for key, item in value.items()
            if key not in {"bbox", "cell_bboxes"}
        }
        if "locators" in compact:
            compact["locators"] = _unique_page_locators(compact["locators"])
        return compact
    if isinstance(value, list):
        return [_compact_pdf_locations(item) for item in value]
    return value


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    code: str | None = None
    instrument_id: str | None = None
    document_scope: Literal["code", "approving_decree", "preamble"] | None = None
    article: str | None = None
    paragraph: str | None = None
    subparagraph: str | None = None
    legal_status: Literal["unknown", "repealed"] | None = None


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(min_length=1, max_length=32000)
    snapshot_id: SHA256
    limit: int = Field(default=5, ge=1, le=50, strict=True)
    query_language: SupportedLanguage | None = None
    context_tokens: int = Field(default=6000, gt=0, strict=True)
    filters: SearchFilters = Field(default_factory=SearchFilters)


class Ranking(BaseModel):
    fragment_id: str
    score: float
    vector_rank: int | None = None


class SearchHit(Ranking):
    fragment: SearchFragment
    related_units: list[LegalUnit]
    related_tables: list[ExtractedTable]
    available_reference_ids: list[str] = Field(default_factory=list)


class UnitCoverage(BaseModel):
    legal_unit_id: str
    part_count: int
    returned_part_indices: list[int]
    source_ranges: list[tuple[int, int]]
    complete: bool


class SearchResult(BaseModel):
    status: Literal["ok", "no_match"]
    snapshot_id: str
    query_language: SupportedLanguage
    document_title: str | None = None
    consolidation_label: str | None = None
    hits: list[SearchHit] = Field(default_factory=list)
    reason: str | None = None
    # These describe returned evidence; complete corpus diagnostics stay stored.
    extraction_issues: list[dict] = Field(default_factory=list)
    unresolved_references: list[dict] = Field(default_factory=list)
    omissions: list[dict] = Field(default_factory=list)
    uncovered_unit_ids: list[str] = Field(default_factory=list)
    table_issues: list[str] = Field(default_factory=list)
    structure_issues: list[dict] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    context_tokens_used: int = 0
    unit_coverage: list[UnitCoverage] = Field(default_factory=list)
    usage: dict

    @model_serializer(mode="wrap")
    def serialize_response(self, handler: SerializerFunctionWrapHandler) -> dict:
        # Keep geometry in source artifacts, outside the response sent to an LLM.
        return _compact_pdf_locations(handler(self))


class UnitText(BaseModel):
    snapshot_id: str
    legal_unit_id: str
    citation_label: str
    role: str
    legal_status: str
    source: SourceDefinition
    locators: list[SourceLocator]
    source_start: int
    source_end: int
    source_length: int
    text_original: str
    complete: bool
    next_start: int | None
    available_note_ids: list[str]
    available_reference_ids: list[str]
    context_tokens_used: int


class NeighborParts(BaseModel):
    snapshot_id: str
    legal_unit_id: str
    parts: list[SearchFragment]
    omitted_part_indices: list[int]
    complete: bool
    context_tokens_used: int


class ContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: SHA256
    legal_unit_id: str
    reference_ids: list[str] = Field(default_factory=list)
    note_ids: list[str] = Field(default_factory=list)
    # Caller-reviewed dependencies; an edge must also exist in the source graph.
    dependency_edges: list[tuple[str, str]] = Field(default_factory=list)
    context_tokens: int = Field(default=6000, gt=0, strict=True)


class ExpandedContext(BaseModel):
    snapshot_id: str
    legal_unit_id: str
    units: list[UnitText]
    tables: list[ExtractedTable]
    omitted_unit_ids: list[str]
    omitted_table_ids: list[str]
    complete: bool
    context_tokens_used: int
