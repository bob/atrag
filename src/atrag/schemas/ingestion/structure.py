"""Located legal addresses, independent of legal versions and applicability."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import Field, PositiveInt

from atrag.schemas.ingestion.extraction import Artifact, BBox, ExtractedTable
from atrag.schemas.materials import Location


class SourceLocator(Artifact):
    page_number: PositiveInt
    printed_page_labels: list[str]
    bbox: BBox


class EditorialNote(Artifact):
    kind: Literal["amendment", "addition", "repeal"]
    raw_text: str
    locators: list[SourceLocator]


class LegalReference(Artifact):
    raw_text: str
    kind: Literal["article", "paragraph", "subparagraph"]
    status: Literal["resolved", "unresolved"]
    target_ids: list[str]
    reason: str | None = None


class LegalUnit(Artifact):
    schema_version: Literal[1] = 1
    legal_unit_id: str
    parent_id: str | None
    instrument_id: str
    code: Literal["CIRS"] | None
    document_scope: Literal["approving_decree", "preamble", "code"]
    kind: Literal[
        "document",
        "chapter",
        "section",
        "subsection",
        "article",
        "paragraph",
        "subparagraph",
        "item",
        "note",
    ]
    role: Literal[
        "norm",
        "approving_decree",
        "code",
        "preamble",
        "editorial_note",
        "transitional",
        "closing",
    ]
    chapter: str | None = None
    section: str | None = None
    article: str | None = None
    paragraph: str | None = None
    subparagraph: str | None = None
    label: str | None = None
    unit_title: str = ""
    citation_label: str
    # A repeal marker is evidence of a printed status, never of its period.
    legal_status: Literal["unknown", "repealed"] = "unknown"
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_start: PositiveInt | None = None
    page_end: PositiveInt | None = None
    # Omitted for legacy PDF artifacts so their content hashes remain valid.
    location: Location | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    printed_page_labels: list[str]
    locators: list[SourceLocator]
    text_original: str
    text_normalized: str
    editorial_notes: list[EditorialNote]
    references: list[LegalReference] = Field(default_factory=list)
    table_ids: list[str]
    issues: list[str]


class StructureResult(Artifact):
    units: list[LegalUnit]
    tables: list[ExtractedTable]
    excluded_block_count: int
    extraction_issues: list[dict]


@dataclass(frozen=True)
class StructureRun:
    run_id: UUID
    directory: Path
    unit_count: int
    review_status: str
