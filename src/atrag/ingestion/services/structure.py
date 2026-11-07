"""Read a completed extraction and publish a separate local structure run."""

import hashlib
from pathlib import Path

from atrag.ingestion.parse_cirs import PARSER_VERSION, parse_cirs
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import read_snapshot
from atrag.schemas.ingestion.extraction import (
    ExtractedPage,
    ExtractedTable,
    ExtractionManifest,
    ExtractionTables,
)
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import StructureRun


def _check_tables(pages: list[ExtractedPage], tables: list[ExtractedTable]):
    by_id = {t.table_id: t for t in tables}
    parts = {p.part_id: p for t in tables for p in t.parts}
    located = {p.part_id: p for page in pages for p in page.tables}
    if len(by_id) != len(tables) or len(parts) != sum(len(t.parts) for t in tables):
        raise ValueError("Duplicate table identity")
    if parts != located:
        raise ValueError("Table locations differ between extraction artifacts")
    if len(located) != sum(len(page.tables) for page in pages):
        raise ValueError("Duplicate table location")
    for page in pages:
        for part in page.tables:
            if part.page_number != page.page_number or part.table_id not in by_id:
                raise ValueError("Invalid table locator")


def _load_inputs(raw: dict[str, bytes], data_dir: Path, source: SourceDefinition):
    parent = ExtractionManifest.model_validate_json(raw["manifest.json"])
    if (
        source.expected_sha256 is not None
        and source.expected_sha256 != parent.snapshot_id
    ):
        raise ValueError("Extraction does not match the registry")
    snapshot_dir = data_dir / "raw" / parent.snapshot_id
    snapshot = SourceSnapshot.model_validate_json(
        (snapshot_dir / "manifest.json").read_bytes()
    )
    read_snapshot(snapshot_dir, parent.snapshot_id, snapshot.size_bytes)
    pages = [
        ExtractedPage.model_validate_json(line)
        for line in raw["pages.jsonl"].splitlines()
    ]
    if [p.page_number for p in pages] != list(range(1, parent.page_count + 1)):
        raise ValueError("Expected every physical page in order")
    if any(p.snapshot_id != parent.snapshot_id for p in pages):
        raise ValueError("Extraction pages belong to another snapshot")
    tables = ExtractionTables.model_validate_json(raw["tables.json"]).tables
    _check_tables(pages, tables)
    return pages, tables


def _report(result) -> dict:
    units = result.units
    return {
        "schema_version": 1,
        "parser_version": PARSER_VERSION,
        "snapshot_id": units[0].snapshot_id,
        "unit_count": len(units),
        "article_count": sum(u.kind == "article" for u in units),
        "excluded_block_count": result.excluded_block_count,
        "repealed_unit_count": sum(u.legal_status == "repealed" for u in units),
        "unresolved_reference_count": sum(
            r.status == "unresolved" for u in units for r in u.references
        ),
        "review_status": "needs_review",
        "validity_status": "unknown",
        "extraction_issues": result.extraction_issues,
        "issues": [
            {"legal_unit_id": u.legal_unit_id, "issues": u.issues}
            for u in units
            if u.issues
        ],
    }


def _save(directory: Path, result):
    write_jsonl(directory / "legal_units.jsonl", result.units)
    write_json(
        directory / "tables.json",
        {"tables": [t.model_dump(mode="json") for t in result.tables]},
    )
    report = _report(result)
    write_json(directory / "report.json", report)
    return report


def parse_extraction(
    extracted: Path, data_dir: Path, source: SourceDefinition
) -> StructureRun:
    from atrag.schemas.ingestion.sources import SourceFormat

    if source.source_format == SourceFormat.HTML_COLLECTION:
        from atrag.ingestion.services.faq_structure import parse_faq_extraction

        return parse_faq_extraction(extracted, data_dir, source)
    with record_run(
        data_dir,
        stage="parse",
        parser_version=PARSER_VERSION,
        source=source.model_dump(mode="json"),
    ) as run:
        directory, manifest = run.directory, run.manifest
        raw = {
            name: (extracted.parent / name).read_bytes()
            for name in ("pages.jsonl", "tables.json")
        }
        raw["manifest.json"] = extracted.read_bytes()
        manifest["input_sha256"] = {
            name: hashlib.sha256(content).hexdigest() for name, content in raw.items()
        }
        pages, tables = _load_inputs(raw, data_dir, source)
        result = parse_cirs(pages, tables)
        report = _save(directory, result)
        manifest["output_sha256"] = {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("legal_units.jsonl", "tables.json", "report.json")
        }
        manifest.update(**report)
    return StructureRun(
        run.run_id, directory, len(result.units), report["review_status"]
    )
