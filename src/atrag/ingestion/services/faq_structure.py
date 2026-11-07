"""Check extracted FAQ artifacts once before entering the shared pipeline."""

import json

from atrag.ingestion.faq_structure import faq_structure
from atrag.ingestion.package_validation import digest
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import read_snapshot
from atrag.schemas.ingestion.extraction import MaterialTables
from atrag.schemas.ingestion.faq_extraction import ExtractedFAQ, VerifiedFAQExtraction
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.ingestion.structure import StructureRun


def _load_inputs(raw, data_dir, source):
    parent = VerifiedFAQExtraction.model_validate_json(raw["manifest.json"])
    for name in ("faqs.jsonl", "tables.json", "report.json"):
        if parent.output_sha256.get(name) != digest(raw[name]):
            raise ValueError("FAQ extraction artifact checksum mismatch")
    directory = data_dir / "raw" / parent.snapshot_id
    stored = SourceSnapshot.model_validate_json(
        (directory / "manifest.json").read_bytes()
    )
    snapshot = read_snapshot(directory, parent.snapshot_id, stored.size_bytes)
    if (
        source.expected_sha256 is not None
        and source.expected_sha256 != parent.snapshot_id
    ):
        raise ValueError("FAQ extraction does not match the registry")
    ignored = {"local_path", "expected_sha256"}
    if snapshot.source.model_dump(exclude=ignored) != source.model_dump(
        exclude=ignored
    ):
        raise ValueError("FAQ extraction source differs from snapshot")
    records = [
        ExtractedFAQ.model_validate_json(line)
        for line in raw["faqs.jsonl"].splitlines()
    ]
    if len(records) != parent.question_count or len(
        {r.entry.entry_id for r in records}
    ) != len(records):
        raise ValueError("FAQ extraction count or identity mismatch")
    return parent, records


def parse_faq_extraction(extracted, data_dir, source):
    with record_run(
        data_dir, stage="parse", source=source.model_dump(mode="json")
    ) as run:
        raw = {
            name: (extracted.parent / name).read_bytes()
            for name in ("faqs.jsonl", "tables.json", "report.json", "manifest.json")
        }
        parent, records = _load_inputs(raw, data_dir, source)
        snapshot_id = parent.snapshot_id
        result = faq_structure(
            records,
            MaterialTables.model_validate_json(raw["tables.json"]).tables,
            snapshot_id,
            source,
        )
        write_jsonl(run.directory / "legal_units.jsonl", result.units)
        write_json(run.directory / "tables.json", json.loads(raw["tables.json"]))
        report = {
            "schema_version": 1,
            "snapshot_id": snapshot_id,
            "parser_version": parent.parser_version,
            "unit_count": len(result.units),
            "excluded_block_count": 0,
            "extraction_issues": [],
            "review_status": "verified",
            "faq_count": len(records),
        }
        write_json(run.directory / "report.json", report)
        run.manifest.update(
            **report,
            input_sha256={name: digest(content) for name, content in raw.items()},
            output_sha256={
                name: digest((run.directory / name).read_bytes())
                for name in ("legal_units.jsonl", "tables.json", "report.json")
            },
        )
    return StructureRun(run.run_id, run.directory, len(result.units), "verified")
