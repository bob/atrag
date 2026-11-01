"""Load explicit source text or neighboring parts from one indexed snapshot."""

from sqlalchemy import select

from atrag.db_models import DocumentTableRow, FragmentRow, LegalUnitRow, SnapshotRow
from atrag.retrieval.expansion import expand_context
from atrag.retrieval.parts import neighbor_parts, unit_text
from atrag.schemas.ingestion.extraction import ExtractedTable
from atrag.schemas.ingestion.fragments import SearchFragment
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import LegalUnit
from atrag.schemas.retrieval import ContextRequest


def _units(connection, snapshot_id):
    rows = connection.scalars(
        select(LegalUnitRow.payload).where(LegalUnitRow.snapshot_id == snapshot_id)
    )
    return {u.legal_unit_id: u for u in map(LegalUnit.model_validate, rows)}


def _source(connection, snapshot_id):
    snapshot = (
        connection.execute(
            select(SnapshotRow).where(SnapshotRow.snapshot_id == snapshot_id)
        )
        .mappings()
        .one_or_none()
    )
    if snapshot is None:
        raise ValueError("Snapshot is not indexed")
    return SourceDefinition.model_validate(
        {key: snapshot[key] for key in SourceDefinition.model_fields if key in snapshot}
    )


def load_unit_text(legal_unit_id, *, snapshot_id, engine, start=0, context_tokens=6000):
    with (
        engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        ) as connection,
        connection.begin(),
    ):
        source = _source(connection, snapshot_id)
        units = _units(connection, snapshot_id)
        if legal_unit_id not in units:
            raise ValueError("Legal unit is absent from the selected snapshot")
        return unit_text(
            units[legal_unit_id],
            source,
            units,
            start=start,
            context_tokens=context_tokens,
        )


def load_context(request, *, engine):
    request = ContextRequest.model_validate(request)
    with (
        engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        ) as connection,
        connection.begin(),
    ):
        source = _source(connection, request.snapshot_id)
        units = _units(connection, request.snapshot_id)
        rows = connection.scalars(
            select(DocumentTableRow.payload).where(
                DocumentTableRow.snapshot_id == request.snapshot_id
            )
        )
        tables = {t.table_id: t for t in map(ExtractedTable.model_validate, rows)}
        return expand_context(request, source, units, tables)


def load_neighbor_parts(
    fragment_id, *, snapshot_id, engine, before=0, after=1, context_tokens=6000
):
    with (
        engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        ) as connection,
        connection.begin(),
    ):
        payload = connection.scalar(
            select(FragmentRow.payload).where(
                FragmentRow.snapshot_id == snapshot_id,
                FragmentRow.fragment_id == fragment_id,
            )
        )
        if payload is None:
            raise ValueError("Fragment is absent from the selected snapshot")
        part = SearchFragment.model_validate(payload)
        rows = connection.scalars(
            select(FragmentRow.payload)
            .where(
                FragmentRow.snapshot_id == snapshot_id,
                FragmentRow.legal_unit_id == part.legal_unit_id,
            )
            .order_by(FragmentRow.payload["part_index"].as_integer())
        )
        parts = list(map(SearchFragment.model_validate, rows))
        unit = connection.scalar(
            select(LegalUnitRow.payload).where(
                LegalUnitRow.snapshot_id == snapshot_id,
                LegalUnitRow.legal_unit_id == part.legal_unit_id,
            )
        )
        return neighbor_parts(
            LegalUnit.model_validate(unit),
            parts,
            part.part_index,
            before=before,
            after=after,
            context_tokens=context_tokens,
        )
