import hashlib
import json
from pathlib import Path

import pytest

from atrag.cli import main
from atrag.ingestion.parse_cirs import parse_cirs
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.chunks import chunk_parsed
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.schemas.ingestion.extraction import ExtractedPage, TextBlock
from atrag.schemas.ingestion.fragments import ChunkParameters


@pytest.fixture
def parsed(tmp_path, monkeypatch):
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(b"%PDF-1.7\nChunk test\n%%EOF\n")
    source = load_source_registry(Path("data/sources/cirs.json")).model_copy(
        update={"local_path": pdf, "expected_sha256": None}
    )
    data = tmp_path / "data"
    snapshot = import_snapshot(source, data).snapshot
    texts = [source.document_title, "Artigo 1.º", "Título", "1 - Não se aplica."]
    blocks = [
        TextBlock(
            text=t,
            bbox=(260, 100 + i * 20, 335, 110 + i * 20),
            font_names=["Arial-BoldMT" if i < 3 else "ArialMT"],
        )
        for i, t in enumerate(texts)
    ]
    result = parse_cirs(
        [
            ExtractedPage(
                snapshot_id=snapshot.snapshot_id,
                page_number=1,
                width=595,
                height=842,
                raw_text="\n".join(texts),
                blocks=blocks,
                tables=[],
                images=[],
            )
        ],
        [],
    )
    directory = tmp_path / "parsed"
    directory.mkdir()
    write_jsonl(directory / "legal_units.jsonl", result.units)
    write_json(directory / "tables.json", {"tables": []})
    write_json(
        directory / "report.json",
        {
            "excluded_block_count": 0,
            "extraction_issues": [{"page_number": 1, "issues": ["image_needs_review"]}],
        },
    )
    write_json(
        directory / "manifest.json",
        {
            "stage": "parse",
            "status": "succeeded",
            "parser_version": "2",
            "snapshot_id": snapshot.snapshot_id,
            "output_sha256": {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ("legal_units.jsonl", "tables.json", "report.json")
            },
        },
    )
    registry = tmp_path / "registry.json"
    registry.write_text(source.model_dump_json())
    monkeypatch.setenv("ATRAG_DATA_DIR", str(data))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DATABASE_URL", "")
    return data, directory / "manifest.json", source, registry


def test_chunk_run_is_reproducible_and_preserves_input(parsed):
    data, parent, source, _ = parsed
    before = {p: p.read_bytes() for p in parent.parent.iterdir()}
    first = chunk_parsed(parent, data, source, ChunkParameters())
    second = chunk_parsed(parent, data, source, ChunkParameters())
    assert first.run_id != second.run_id
    assert first.fragment_count == 1
    for name in ("fragments.jsonl", "report.json"):
        assert (first.directory / name).read_bytes() == (
            second.directory / name
        ).read_bytes()
    report = json.loads((first.directory / "report.json").read_text())
    assert report["article_count"] == report["paragraph_count"] == 1
    assert report["schema_version"] == 1
    assert report["uncovered_unit_ids"] == []
    assert report["extraction_issues"][0]["page_number"] == 1
    assert report["review_status"] == "needs_review"
    manifest = json.loads((first.directory / "manifest.json").read_text())
    assert (manifest["stage"], manifest["status"], manifest["chunker_version"]) == (
        "chunk",
        "succeeded",
        "1",
    )
    assert manifest["tokenizer_version"]
    assert manifest["parameters"]["max_tokens"] == 1200
    assert (
        manifest["output_sha256"]["fragments.jsonl"]
        == hashlib.sha256(
            (first.directory / "fragments.jsonl").read_bytes()
        ).hexdigest()
    )
    assert (
        manifest["input_sha256"]["manifest.json"]
        == hashlib.sha256(before[parent]).hexdigest()
    )
    assert manifest["started_at"].endswith("+00:00")
    assert all(p.read_bytes() == b for p, b in before.items())


@pytest.mark.parametrize(
    "damage", ["status", "hash", "snapshot", "registry", "missing"]
)
def test_chunk_rejects_invalid_inputs_and_records_failure(parsed, damage):
    data, parent, source, _ = parsed
    manifest = json.loads(parent.read_text())
    if damage == "status":
        manifest["status"] = "running"
    elif damage == "hash":
        (parent.parent / "legal_units.jsonl").write_text("untrusted-data")
    elif damage == "snapshot":
        path = next((data / "raw").glob("*/source.pdf"))
        path.write_bytes(b"corrupt snapshot")
    elif damage == "registry":
        source = source.model_copy(update={"expected_sha256": "b" * 64})
    else:
        del manifest["output_sha256"]["tables.json"]
    parent.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        chunk_parsed(parent, data, source, ChunkParameters())
    manifests = [
        json.loads(p.read_text()) for p in (data / "runs").glob("*/manifest.json")
    ]
    assert any(m.get("stage") == "chunk" and m["status"] == "failed" for m in manifests)


def test_chunk_command(parsed, capsys):
    data, parent, _, registry = parsed
    main(["ingest", "chunk", "--parsed", str(parent), "--registry", str(registry)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["fragment_count"] == 1
    assert Path(summary["directory"]).parent == data / "runs"


def test_chunk_command_hides_input_contents(parsed, capsys):
    _, parent, _, registry = parsed
    parent.write_text("malformed-secret")
    with pytest.raises(SystemExit) as error:
        main(["ingest", "chunk", "--parsed", str(parent), "--registry", str(registry)])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "Fragment preparation failed" in output
    assert "secret" not in output


def test_chunk_requires_parsed_manifest(capsys):
    with pytest.raises(SystemExit) as error:
        main(["ingest", "chunk"])
    assert error.value.code == 2
    assert "--parsed" in capsys.readouterr().err


def update_artifact(parent, name, content):
    (parent.parent / name).write_text(content)
    manifest = json.loads(parent.read_text())
    manifest["output_sha256"][name] = hashlib.sha256(content.encode()).hexdigest()
    parent.write_text(json.dumps(manifest))


def test_legal_units_from_a_different_snapshot_are_rejected(parsed):
    data, parent, source, _ = parsed
    path = parent.parent / "legal_units.jsonl"
    first = json.loads(path.read_text().splitlines()[0])
    content = path.read_text().replace(first["snapshot_id"], "b" * 64)
    update_artifact(parent, "legal_units.jsonl", content)
    with pytest.raises(ValueError, match="another snapshot"):
        chunk_parsed(parent, data, source, ChunkParameters())


def test_ready_report_with_verified_text_and_no_extraction_issues(parsed):
    data, parent, source, _ = parsed
    update_artifact(
        parent,
        "report.json",
        json.dumps({"excluded_block_count": 0, "extraction_issues": []}),
    )
    run = chunk_parsed(parent, data, source, ChunkParameters())
    assert run.review_status == "ready"
    report = json.loads((run.directory / "report.json").read_text())
    assert report["unresolved_references"] == report["omissions"] == []


def test_context_coverage_accounts_for_whole_notes(parsed):
    data, parent, source, _ = parsed
    path = parent.parent / "legal_units.jsonl"
    units = [json.loads(l) for l in path.read_text().splitlines()]
    note = {
        **units[-1],
        "kind": "note",
        "role": "transitional",
        "legal_unit_id": units[1]["legal_unit_id"] + ":note:1",
        "parent_id": units[1]["legal_unit_id"],
        "text_original": "Nota: Um esclarecimento.",
        "text_normalized": "Nota: Um esclarecimento.",
    }
    child = {
        **note,
        "kind": "paragraph",
        "legal_unit_id": note["legal_unit_id"] + ":paragraph:1",
        "parent_id": note["legal_unit_id"],
    }
    update_artifact(
        parent,
        "legal_units.jsonl",
        "\n".join(json.dumps(u) for u in [*units, note, child]) + "\n",
    )
    run = chunk_parsed(parent, data, source, ChunkParameters())
    report = json.loads((run.directory / "report.json").read_text())
    assert child["legal_unit_id"] not in report["uncovered_unit_ids"]


def test_partial_parent_context_does_not_hide_an_oversize_sibling(parsed):
    data, parent, source, _ = parsed
    path = parent.parent / "legal_units.jsonl"
    units = [json.loads(l) for l in path.read_text().splitlines()]
    text = "2 - " + "Condição indivisível. " * 600
    sibling = {
        **units[-1],
        "legal_unit_id": units[1]["legal_unit_id"] + ":paragraph:2",
        "paragraph": "2",
        "text_original": text,
        "text_normalized": text,
    }
    units[1]["text_original"] += "\n" + text
    update_artifact(
        parent,
        "legal_units.jsonl",
        "\n".join(json.dumps(u) for u in [*units, sibling]) + "\n",
    )
    run = chunk_parsed(parent, data, source, ChunkParameters())
    report = json.loads((run.directory / "report.json").read_text())
    assert sibling["legal_unit_id"] in report["uncovered_unit_ids"]
    assert report["omissions"][0]["legal_unit_id"] == sibling["legal_unit_id"]
    assert report["review_status"] == "needs_review"


def test_report_preserves_unresolved_references_and_unverified_tables(parsed):
    data, parent, source, _ = parsed
    path = parent.parent / "legal_units.jsonl"
    units = [json.loads(l) for l in path.read_text().splitlines()]
    units[-1]["references"] = [
        {
            "raw_text": "artigo 999.º",
            "kind": "article",
            "status": "unresolved",
            "target_ids": [],
            "reason": "absent",
        },
        {
            "raw_text": "artigo 1.º",
            "kind": "article",
            "status": "resolved",
            "target_ids": [units[1]["legal_unit_id"]],
            "reason": None,
        },
    ]
    update_artifact(
        parent, "legal_units.jsonl", "\n".join(json.dumps(u) for u in units) + "\n"
    )
    tables = [
        {
            "table_id": "unverified",
            "parts": [],
            "header_rows": [],
            "rows": [["2,5"]],
            "context": "Table requiring review",
            "review_status": "needs_review",
        },
        {
            "table_id": "verified",
            "parts": [],
            "header_rows": [],
            "rows": [["5"]],
            "context": "Checked table",
            "review_status": "verified",
        },
    ]
    update_artifact(parent, "tables.json", json.dumps({"tables": tables}))
    run = chunk_parsed(parent, data, source, ChunkParameters())
    report = json.loads((run.directory / "report.json").read_text())
    assert report["unresolved_references"] == [
        {
            "legal_unit_id": units[-1]["legal_unit_id"],
            "reference": units[-1]["references"][0],
        }
    ]
    assert report["table_issues"] == ["unverified"]


def test_empty_output_is_never_reported_ready(parsed):
    data, parent, source, _ = parsed
    first = (parent.parent / "legal_units.jsonl").read_text().splitlines()[0]
    update_artifact(parent, "legal_units.jsonl", first + "\n")
    update_artifact(
        parent,
        "report.json",
        json.dumps({"excluded_block_count": 0, "extraction_issues": []}),
    )
    run = chunk_parsed(parent, data, source, ChunkParameters())
    assert run.fragment_count == 0
    assert run.review_status == "needs_review"


def test_unresolved_reference_alone_requires_review(parsed):
    data, parent, source, _ = parsed
    path = parent.parent / "legal_units.jsonl"
    units = [json.loads(l) for l in path.read_text().splitlines()]
    reference = {
        "raw_text": "artigo 999.º",
        "kind": "article",
        "status": "unresolved",
        "target_ids": [],
        "reason": "absent",
    }
    units[-1]["references"] = [reference]
    update_artifact(
        parent, "legal_units.jsonl", "\n".join(json.dumps(u) for u in units) + "\n"
    )
    update_artifact(
        parent,
        "report.json",
        json.dumps({"excluded_block_count": 0, "extraction_issues": []}),
    )
    run = chunk_parsed(parent, data, source, ChunkParameters())
    report = json.loads((run.directory / "report.json").read_text())
    assert report["uncovered_unit_ids"] == report["table_issues"] == []
    assert run.review_status == "needs_review"
