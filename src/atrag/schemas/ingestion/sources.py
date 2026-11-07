"""Source registration and provenance supplied to snapshot import."""

from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    model_validator,
)

from atrag.schemas.settings import SupportedLanguage


class SourceFormat(StrEnum):
    PDF = "pdf"
    HTML_COLLECTION = "html_collection"

    @property
    def snapshot_filename(self) -> str:
        return {self.PDF: "source.pdf", self.HTML_COLLECTION: "collection.json"}[self]


class SourceDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    source_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    document_type: Literal["law", "at_instruction", "individual_ruling", "faq"]
    # Legacy PDF serialization is part of prepared identities.
    source_format: SourceFormat = Field(
        default=SourceFormat.PDF, exclude_if=lambda v: v == SourceFormat.PDF
    )
    publisher: str = Field(min_length=1)
    language: SupportedLanguage
    local_path: Path
    source_url: HttpUrl | None
    source_url_status: Literal["unknown", "candidate", "verified"]
    consolidation_label: str | None = None
    publication_date: date | None = None
    revision_publication_date: date | None = None
    downloaded_at: AwareDatetime | None = None
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def url_evidence(self) -> Self:
        if self.source_url_status != "unknown" and self.source_url is None:
            raise ValueError("A candidate or verified source URL is required")
        return self
