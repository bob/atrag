"""Snapshot-specific, manually reviewed retrieval expectations."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from atrag.schemas.ingestion.packages import SHA256
from atrag.schemas.retrieval import ContextRequest, SearchFilters
from atrag.schemas.settings import SupportedLanguage


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    query: str = Field(min_length=1)
    language: SupportedLanguage
    detect_language: bool = False
    filters: SearchFilters = Field(default_factory=SearchFilters)
    kind: Literal["exact", "semantic"]
    expected_status: Literal["ok", "no_match"] = "ok"
    expected_reason: str | None = None
    expected_unit_ids: list[str] = Field(default_factory=list)
    expected_quotes: dict[str, list[str]] = Field(default_factory=dict)
    required_context_ids: list[str] = Field(default_factory=list)
    required_table_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_expectation(self) -> Self:
        if not set(self.expected_quotes) <= set(self.expected_unit_ids):
            raise ValueError("Search quotes need an expected target unit")
        if self.expected_status == "ok":
            if not self.expected_unit_ids or self.expected_reason is not None:
                raise ValueError("Positive cases need target units and no reason")
        elif (
            not self.expected_reason
            or self.expected_unit_ids
            or self.required_context_ids
            or self.required_table_ids
        ):
            raise ValueError("No-match cases need a reason and no evidence")
        return self


class ExpansionCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[a-zA-Z0-9_-]+$")
    language: SupportedLanguage
    kind: Literal["original", "references", "explanation", "note"]
    request: ContextRequest
    required_unit_ids: list[str] = Field(min_length=1)
    required_table_ids: list[str] = Field(default_factory=list)


class TableExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    header_rows: list[list[str | None]]
    rows: list[list[str | None]]
    pages: list[int] = Field(min_length=1)


class UnitCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contains: list[str] = Field(default_factory=list)
    legal_status: Literal["unknown", "repealed"] | None = None
    role: str | None = None


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1, 2] = 1
    snapshot_id: SHA256
    review_note: str = ""
    unit_pages: dict[str, tuple[int, int]]
    tables: dict[str, TableExpectation] = Field(default_factory=dict)
    unit_checks: dict[str, UnitCheck] = Field(default_factory=dict)
    cases: list[EvaluationCase] = Field(min_length=1)
    expansion_cases: list[ExpansionCase] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        ids = [case.case_id for case in [*self.cases, *self.expansion_cases]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate benchmark case IDs")
        return self
