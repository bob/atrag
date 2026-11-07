"""Located legal addresses, independent of legal versions and applicability."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

from pydantic import Field, PositiveInt, model_validator

from atrag.schemas.ingestion.extraction import Artifact, BBox, ExtractedTable
from atrag.schemas.ingestion.faq import FAQEntry
from atrag.schemas.materials import HTMLLocation, Location


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
    code: Literal["CIRS"] | None = None
    document_scope: Literal["approving_decree", "preamble", "code", "faq"]
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
        "topic",
        "faq_entry",
    ]
    role: Literal[
        "norm",
        "approving_decree",
        "code",
        "preamble",
        "editorial_note",
        "transitional",
        "closing",
        "at_explanation",
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
    faq: FAQEntry | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def consistent_faq(self) -> Self:
        if self.document_scope == "faq":
            location = self._check_faq_address()
            self._check_faq_entry(location)
        elif (
            self.faq is not None
            or self.kind in ("topic", "faq_entry")
            or self.role == "at_explanation"
        ):
            raise ValueError("FAQ metadata, kinds and role require FAQ scope")
        return self

    def _check_faq_address(self) -> HTMLLocation:
        address = (
            self.code,
            self.chapter,
            self.section,
            self.article,
            self.paragraph,
            self.subparagraph,
        )
        pdf = (
            self.page_start,
            self.page_end,
            *self.locators,
            *self.printed_page_labels,
        )
        if any(value is not None for value in (*address, *pdf)):
            raise ValueError("FAQ cannot have a legal address or PDF coordinates")
        if self.role != "at_explanation" or self.kind not in (
            "document",
            "topic",
            "faq_entry",
        ):
            raise ValueError("FAQ requires explanatory units")
        if self.legal_status != "unknown" or self.editorial_notes:
            raise ValueError(
                "FAQ cannot have a normative status or editorial amendments"
            )
        if (
            not isinstance(self.location, HTMLLocation)
            or self.location.page_url is None
        ):
            raise ValueError("FAQ requires an HTML location with a page URL")
        return self.location

    def _check_faq_entry(self, location: HTMLLocation) -> None:
        if self.kind != "faq_entry":
            if self.faq is not None:
                raise ValueError("Only FAQ entries carry question and answer metadata")
            return
        if self.faq is None:
            raise ValueError("FAQ entry requires original question and answer metadata")
        if self.legal_unit_id != f"{self.instrument_id}:{self.faq.entry_id}":
            raise ValueError("FAQ unit identity must match its source and entry")
        if (location.page_url, location.anchor) != (
            self.faq.page_url,
            self.faq.anchor,
        ):
            raise ValueError(
                "FAQ entry and location must refer to the same page and anchor"
            )


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
