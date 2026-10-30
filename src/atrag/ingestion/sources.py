"""Source registration and provenance supplied to snapshot import."""

from datetime import date
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

from atrag.settings import SupportedLanguage


class SourceDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    source_id: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    document_type: Literal["law", "at_instruction", "individual_ruling", "faq"]
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


def load_source_registry(path: Path) -> SourceDefinition:
    return SourceDefinition.model_validate_json(path.read_text(encoding="utf-8"))
