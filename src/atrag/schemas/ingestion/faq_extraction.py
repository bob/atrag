"""Inspect FAQ extraction without inventing PDF or legal addresses."""

from typing import Literal

from atrag.schemas.ingestion.extraction import Artifact, ExtractedTable, HTMLTablePart
from atrag.schemas.ingestion.faq import FAQEntry


class FAQListItem(Artifact):
    text: str
    ordered: bool
    depth: int


class ExtractedFAQ(Artifact):
    entry: FAQEntry
    question_normalized: str
    answer_normalized: str
    table_ids: tuple[str, ...] = ()
    list_items: tuple[FAQListItem, ...] = ()
    references: tuple[str, ...] = ()


class FAQIssue(Artifact):
    kind: str
    ordinal: int | None = None
    detail: str


class FAQPage(Artifact):
    page_url: str
    detected_count: int
    answer_count: int = 0
    detected_table_count: int = 0
    faqs: tuple[ExtractedFAQ, ...] = ()
    tables: tuple[ExtractedTable[HTMLTablePart], ...] = ()
    issues: tuple[FAQIssue, ...] = ()
    rejected_ordinals: tuple[int, ...] = ()

    @property
    def review_status(self) -> Literal["verified", "needs_review"]:
        return "needs_review" if self.issues else "verified"
