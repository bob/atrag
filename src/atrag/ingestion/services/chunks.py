"""Validate a completed parse and publish isolated, deterministic fragments."""

import hashlib
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import tiktoken

from atrag.ingestion.chunk import chunk_structure
from atrag.ingestion.part_validation import reassemble_unit
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import read_snapshot
from atrag.schemas.ingestion.extraction import ExtractionTables
from atrag.schemas.ingestion.fragments import (
    CHUNKER_VERSION,
    ChunkParameters,
    ChunkRun,
    ParsedManifest,
    ParsedReport,
)
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import LegalUnit, StructureResult

ARTIFACTS = ("legal_units.jsonl", "tables.json", "report.json")


def _digest(content):
    return hashlib.sha256(content).hexdigest()


def _load_inputs(raw, data_dir, source):
    parent = ParsedManifest.model_validate_json(raw["manifest.json"])
    for name in ARTIFACTS:
        if parent.output_sha256.get(name) != _digest(raw[name]):
            raise ValueError("Parsed artifact checksum mismatch")
    if (
        source.expected_sha256 is not None
        and source.expected_sha256 != parent.snapshot_id
    ):
        raise ValueError("Parsed snapshot does not match the registry")
    snapshot_dir = data_dir / "raw" / parent.snapshot_id
    snapshot = SourceSnapshot.model_validate_json(
        (snapshot_dir / "manifest.json").read_bytes()
    )
    read_snapshot(snapshot_dir, parent.snapshot_id, snapshot.size_bytes)
    units = [
        LegalUnit.model_validate_json(line)
        for line in raw["legal_units.jsonl"].splitlines()
    ]
    if any(u.snapshot_id != parent.snapshot_id for u in units):
        raise ValueError("Legal units belong to another snapshot")
    report = ParsedReport.model_validate_json(raw["report.json"])
    result = StructureResult(
        units=units,
        tables=ExtractionTables.model_validate_json(raw["tables.json"]).tables,
        excluded_block_count=report.excluded_block_count,
        extraction_issues=report.extraction_issues,
    )
    return parent, snapshot, result


def _descendant_coverage(structure, output):
    covered = {f.legal_unit_id for f in output.fragments}
    children = defaultdict(list)
    for unit in structure.units:
        if unit.kind != "note":
            children[unit.parent_id].append(unit.legal_unit_id)
            if unit.parent_id in covered:
                covered.add(unit.legal_unit_id)
    return covered, children


def _coverage(structure, output):
    covered, children = _descendant_coverage(structure, output)
    headers = {c.legal_unit_id for f in output.fragments for c in f.context}
    for unit in reversed(structure.units):
        if (
            unit.legal_unit_id in headers
            and set(children[unit.legal_unit_id]) <= covered
        ):
            covered.add(unit.legal_unit_id)
    required = [
        u for u in structure.units if u.kind not in ("chapter", "section", "subsection")
    ]
    return [
        u.legal_unit_id
        for u in required
        if u.legal_unit_id not in covered and u.role != "code"
    ]


def _review_status(output, uncovered, extraction_issues, table_issues):
    needs_review = (
        not output.fragments
        or uncovered
        or extraction_issues
        or table_issues
        or any(f.issues for f in output.fragments)
    )
    return "needs_review" if needs_review else "ready"


def _report(structure, output):
    lengths = [f.token_count for f in output.fragments]
    uncovered = _coverage(structure, output)
    table_issues = [
        t.table_id for t in structure.tables if t.review_status != "verified"
    ]
    return {
        "schema_version": 1,
        "chunker_version": CHUNKER_VERSION,
        "snapshot_id": next(iter(structure.units)).snapshot_id,
        "fragment_count": len(output.fragments),
        "article_count": sum(u.kind == "article" for u in structure.units),
        "paragraph_count": sum(u.kind == "paragraph" for u in structure.units),
        "token_counts": lengths,
        "source_part_coverage": _source_part_coverage(structure, output),
        "uncovered_unit_ids": uncovered,
        "omissions": [o.model_dump(mode="json") for o in output.omissions],
        "extraction_issues": structure.extraction_issues,
        "structure_issues": [
            {"legal_unit_id": u.legal_unit_id, "issues": u.issues}
            for u in structure.units
            if u.issues
        ],
        "unresolved_references": [
            {"legal_unit_id": u.legal_unit_id, "reference": r.model_dump(mode="json")}
            for u in structure.units
            for r in u.references
            if r.status == "unresolved"
        ],
        "table_issues": table_issues,
        "excluded_structural_block_count": structure.excluded_block_count,
        "review_status": _review_status(
            output, uncovered, structure.extraction_issues, table_issues
        ),
        "validity_status": "unknown",
    }


def _source_part_coverage(structure, output):
    groups = defaultdict(list)
    units = {u.legal_unit_id: u for u in structure.units}
    for part in output.fragments:
        groups[part.legal_unit_id].append(part)
    coverage = []
    for identity, parts in groups.items():
        reassemble_unit(units[identity], parts)
        coverage.append(
            {
                "legal_unit_id": identity,
                "complete": True,
                "ranges": [[p.source_start, p.source_end] for p in parts],
            }
        )
    return coverage


def chunk_parsed(
    parsed: Path, data_dir: Path, source: SourceDefinition, parameters: ChunkParameters
) -> ChunkRun:
    with record_run(
        data_dir,
        stage="chunk",
        chunker_version=CHUNKER_VERSION,
        parameters=parameters.model_dump(),
    ) as run:
        raw = {name: (parsed.parent / name).read_bytes() for name in ARTIFACTS}
        raw["manifest.json"] = parsed.read_bytes()
        run.manifest["input_sha256"] = {
            name: _digest(content) for name, content in raw.items()
        }
        parent, _snapshot, structure = _load_inputs(raw, data_dir, source)
        encoding = tiktoken.get_encoding(parameters.token_encoding)
        result = chunk_structure(
            structure,
            source,
            parser_version=parent.parser_version,
            parameters=parameters,
            count_tokens=lambda text: len(encoding.encode_ordinary(text)),
        )
        write_jsonl(run.directory / "fragments.jsonl", result.fragments)
        report = _report(structure, result)
        write_json(run.directory / "report.json", report)
        run.manifest.update(
            **report,
            parser_version=parent.parser_version,
            tokenizer_version=version("tiktoken"),
            output_sha256={
                name: _digest((run.directory / name).read_bytes())
                for name in ("fragments.jsonl", "report.json")
            },
        )
    return ChunkRun(
        run.run_id, run.directory, len(result.fragments), report["review_status"]
    )
