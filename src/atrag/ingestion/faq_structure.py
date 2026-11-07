"""Adapt checked FAQ records to the common material tree, preserving answers."""

from typing import Literal, TypedDict

from atrag.ingestion.extraction import normalize_text
from atrag.schemas.ingestion.structure import (
    EditorialNote,
    LegalUnit,
    SourceLocator,
    StructureResult,
)
from atrag.schemas.materials import HTMLLocation


class _FAQFields(TypedDict):
    instrument_id: str
    document_scope: Literal["faq"]
    role: Literal["at_explanation"]
    snapshot_id: str
    printed_page_labels: list[str]
    locators: list[SourceLocator]
    editorial_notes: list[EditorialNote]
    issues: list[str]


def faq_structure(records, tables, snapshot_id, source):
    units = []
    topics = {}
    for record in records:
        entry = record.entry
        topic_id = f"{source.source_id}:{entry.entry_id.rsplit(':', 1)[0]}"
        location = HTMLLocation(
            kind="html",
            page_url=entry.page_url,
            anchor=entry.anchor,
            section=" > ".join(entry.topic_path),
        )
        common: _FAQFields = {
            "instrument_id": source.source_id,
            "document_scope": "faq",
            "role": "at_explanation",
            "snapshot_id": snapshot_id,
            "printed_page_labels": [],
            "locators": [],
            "editorial_notes": [],
            "issues": [],
        }
        if topic_id not in topics:
            topic = LegalUnit(
                **common,
                legal_unit_id=topic_id,
                parent_id=None,
                kind="topic",
                citation_label=location.section or source.document_title,
                location=location,
                text_original="",
                text_normalized="",
                table_ids=[],
            )
            topics[topic_id] = topic
            units.append(topic)
        units.append(
            LegalUnit(
                **common,
                legal_unit_id=f"{source.source_id}:{entry.entry_id}",
                parent_id=topic_id,
                kind="faq_entry",
                citation_label=f"FAQ AT {entry.faq_id}",
                location=location,
                text_original=entry.answer,
                text_normalized=normalize_text(entry.answer),
                table_ids=list(record.table_ids),
                faq=entry,
            )
        )
    return StructureResult(
        units=units, tables=tables, excluded_block_count=0, extraction_issues=[]
    )
