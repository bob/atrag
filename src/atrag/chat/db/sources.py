"""Build citations from retrieved source slices, never from model-authored text."""

from sqlalchemy import select

from atrag.db_models import LegalUnitRow, SnapshotRow
from atrag.retrieval.diagnostics import scope_diagnostics
from atrag.retrieval.parts import available_notes, available_references
from atrag.retrieval.text import _source, _units
from atrag.schemas.chat.tools import BlockAddress
from atrag.schemas.ingestion.structure import LegalUnit
from atrag.schemas.retrieval import _compact_pdf_locations


def source_record(connection, address: BlockAddress, text, *, complete, start, end):
    unit = LegalUnit.model_validate(
        connection.scalar(
            select(LegalUnitRow.payload).where(
                LegalUnitRow.snapshot_id == address.snapshot_id,
                LegalUnitRow.legal_unit_id == text.legal_unit_id,
            )
        )
    )
    source = _source(connection, address.snapshot_id)
    evidence = text if isinstance(text, LegalUnit) else unit
    units = _units(connection, address.snapshot_id)
    reports = connection.scalar(
        select(SnapshotRow.reports).where(
            SnapshotRow.snapshot_id == address.snapshot_id
        )
    )
    locations = (
        [unit.location.model_dump(mode="json")]
        if unit.location is not None
        else [
            {"kind": "pdf", "page": page}
            for page in sorted({l.page_number for l in unit.locators})
        ]
    )
    return _compact_pdf_locations(
        {
            "address": address.model_dump(mode="json"),
            "legal_unit_id": unit.legal_unit_id,
            "citation_label": unit.citation_label,
            "document_title": source.document_title,
            "language": source.language,
            "consolidation_label": source.consolidation_label,
            "publication_date": source.model_dump(mode="json")["publication_date"],
            "revision_publication_date": source.model_dump(mode="json")[
                "revision_publication_date"
            ],
            "locations": locations,
            "legal_status": unit.legal_status,
            "role": unit.role,
            "text_original": text.text_original,
            "tables": [
                table.model_dump(mode="json") for table in getattr(text, "tables", [])
            ],
            "repeated_intro": getattr(text, "repeated_intro", ""),
            "source_start": start,
            "source_end": end,
            "source_length": len(unit.text_original),
            "complete": complete,
            "editorial_notes": [
                n.model_dump(mode="json") for n in evidence.editorial_notes
            ],
            "issues": evidence.issues,
            "available_note_ids": available_notes(unit, units),
            "available_reference_ids": available_references(unit, units),
            "diagnostics": scope_diagnostics(
                {unit.legal_unit_id},
                {l.page_number for l in unit.locators},
                set(unit.table_ids),
                reports["chunk"],
            ),
        }
    )
