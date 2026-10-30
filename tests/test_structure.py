import hashlib
import json
from pathlib import Path

import pytest

from atrag.cli import main
from atrag.ingestion.services.extract import extract_snapshot
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.ingestion.services.structure import parse_extraction


@pytest.fixture
def extracted(make_pdf, pdf_text, tmp_path):
    # Helvetica is regular in this tiny PDF; mark the located heading fonts as
    # bold in the synthetic extraction, without depending on local source PDFs.
    pdf = make_pdf(
        [
            pdf_text(
                100,
                "Código do Imposto sobre o Rendimento das Pessoas Singulares",
                x=100,
            )
            + pdf_text(140, "Artigo 1.º", x=260)
            + pdf_text(165, "1 - Não se aplica o número anterior.")
        ]
    )
    source = load_source_registry(Path("data/sources/cirs.json")).model_copy(
        update={"local_path": pdf, "expected_sha256": None}
    )
    data = tmp_path / "data"
    snapshot = import_snapshot(source, data).snapshot
    run = extract_snapshot(snapshot, data)
    path = run.directory / "pages.jsonl"
    page = json.loads(path.read_text())
    page["blocks"][0]["font_names"] = ["Arial-BoldMT"]
    page["blocks"][1]["font_names"] = ["Arial-BoldMT"]
    path.write_text(json.dumps(page, ensure_ascii=False) + "\n")
    registry = tmp_path / "registry.json"
    registry.write_text(source.model_dump_json())
    return data, run.directory / "manifest.json", registry


def test_local_parse_run_is_reproducible_and_preserves_input(extracted):
    data, parent, registry = extracted
    before = {p: p.read_bytes() for p in parent.parent.iterdir() if p.is_file()}
    source = load_source_registry(registry)
    first = parse_extraction(parent, data, source)
    second = parse_extraction(parent, data, source)
    assert first.run_id != second.run_id
    assert first.unit_count == 3
    assert first.review_status == "needs_review"
    for name in ("legal_units.jsonl", "tables.json", "report.json"):
        assert (first.directory / name).read_bytes() == (
            second.directory / name
        ).read_bytes()
    units_text = (first.directory / "legal_units.jsonl").read_text()
    assert "Não" in units_text
    units = [json.loads(l) for l in units_text.splitlines()]
    assert units[-1]["legal_unit_id"] == "pt:cirs:code:article:1:paragraph:1"
    report = json.loads((first.directory / "report.json").read_text())
    assert (
        report["schema_version"],
        report["article_count"],
        report["repealed_unit_count"],
    ) == (1, 1, 0)
    assert report["snapshot_id"] == units[0]["snapshot_id"]
    assert report["unresolved_reference_count"] >= 1
    manifest = json.loads((first.directory / "manifest.json").read_text())
    assert (manifest["stage"], manifest["status"], manifest["parser_version"]) == (
        "parse",
        "succeeded",
        "2",
    )
    assert (
        manifest["input_sha256"]["pages.jsonl"]
        == hashlib.sha256(before[parent.parent / "pages.jsonl"]).hexdigest()
    )
    assert (
        manifest["output_sha256"]["legal_units.jsonl"]
        == hashlib.sha256(units_text.encode()).hexdigest()
    )
    assert manifest["started_at"].endswith("+00:00")
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize(
    "damage",
    ["status", "stage", "snapshot", "order", "empty", "tables", "snapshot_bytes"],
)
def test_parse_rejects_inconsistent_inputs_and_records_failure(extracted, damage):
    data, parent, registry = extracted
    manifest = json.loads(parent.read_text())
    pages_path = parent.parent / "pages.jsonl"
    if damage in ("status", "stage"):
        manifest[damage] = "failed" if damage == "status" else "parse"
        parent.write_text(json.dumps(manifest))
    elif damage in ("snapshot", "order"):
        page = json.loads(pages_path.read_text())
        page["snapshot_id" if damage == "snapshot" else "page_number"] = (
            "b" * 64 if damage == "snapshot" else 2
        )
        pages_path.write_text(json.dumps(page) + "\n")
    elif damage == "empty":
        pages_path.write_text("")
    elif damage == "tables":
        (parent.parent / "tables.json").write_text('{"tables": null}')
    else:
        (data / "raw" / manifest["snapshot_id"] / "source.pdf").write_bytes(
            b"%PDF-corrupt"
        )
    before = set((data / "runs").iterdir())
    with pytest.raises(ValueError):
        parse_extraction(parent, data, load_source_registry(registry))
    new = set((data / "runs").iterdir()) - before
    assert len(new) == 1
    failure = json.loads((new.pop() / "manifest.json").read_text())
    assert failure["status"] == "failed"
    assert failure["finished_at"] is not None


def test_parse_rejects_registry_mismatch(extracted):
    data, parent, registry = extracted
    source = load_source_registry(registry).model_copy(
        update={"expected_sha256": "c" * 64}
    )
    with pytest.raises(ValueError, match="registry"):
        parse_extraction(parent, data, source)


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "table_id",
        "duplicate_table",
        "duplicate_part",
        "missing_part",
        "page_number",
        "duplicate_location",
    ],
)
def test_table_artifacts_must_have_identical_unique_locators(extracted, damage):
    data, parent, registry = extracted
    pages_path = parent.parent / "pages.jsonl"
    page = json.loads(pages_path.read_text())
    part = {
        "part_id": "p1-table1",
        "table_id": "p1-table1",
        "page_number": 1,
        "bbox": [100, 300, 500, 340],
        "rows": [["Montante", "Taxa"], ["100", "2,5"]],
    }
    table = {
        "table_id": "p1-table1",
        "parts": [dict(part)],
        "header_rows": [["Montante", "Taxa"]],
        "rows": [["100", "2,5"]],
        "context": "Uma regra",
    }
    page["tables"] = [dict(part)]
    tables = [table]
    if damage == "table_id":
        table["parts"][0]["table_id"] = page["tables"][0]["table_id"] = "missing"
    elif damage == "duplicate_table":
        tables.append(dict(table))
    elif damage == "duplicate_part":
        table["parts"].append(dict(part))
    elif damage == "missing_part":
        page["tables"] = []
    elif damage == "page_number":
        table["parts"][0]["page_number"] = page["tables"][0]["page_number"] = 2
    elif damage == "duplicate_location":
        page["tables"].append(dict(part))
    pages_path.write_text(json.dumps(page) + "\n")
    (parent.parent / "tables.json").write_text(json.dumps({"tables": tables}))
    if damage:
        with pytest.raises(ValueError):
            parse_extraction(parent, data, load_source_registry(registry))
    else:
        result = parse_extraction(parent, data, load_source_registry(registry))
        units = [
            json.loads(l)
            for l in (result.directory / "legal_units.jsonl").read_text().splitlines()
        ]
        assert units[-1]["table_ids"] == ["p1-table1"]
        assert "100 | 2,5" in units[-1]["text_original"]


def test_parse_command(extracted, monkeypatch, capsys):
    data, parent, registry = extracted
    monkeypatch.setenv("ATRAG_DATA_DIR", str(data))
    main(["ingest", "parse", "--extracted", str(parent), "--registry", str(registry)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["unit_count"] == 3
    assert summary["review_status"] == "needs_review"
    assert Path(summary["directory"]).parent == data / "runs"


def test_parse_command_reports_errors_without_input_contents(
    extracted, monkeypatch, capsys
):
    data, parent, registry = extracted
    monkeypatch.setenv("ATRAG_DATA_DIR", str(data))
    parent.write_text("not-json-secret")
    with pytest.raises(SystemExit) as error:
        main(
            ["ingest", "parse", "--extracted", str(parent), "--registry", str(registry)]
        )
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "Structure parsing failed" in message
    assert "secret" not in message


def test_parse_requires_extracted_manifest(capsys):
    with pytest.raises(SystemExit) as error:
        main(["ingest", "parse"])
    assert error.value.code == 2
    assert "--extracted" in capsys.readouterr().err


def test_upstream_issues_include_pages_without_legal_text(extracted):
    data, parent, registry = extracted
    path = parent.parent / "pages.jsonl"
    p = json.loads(path.read_text())
    p["blocks"] = []
    p["issues"] = ["short_text", "image_needs_review:cover"]
    second = json.loads(path.read_text())
    second["page_number"] = 2
    path.write_text(json.dumps(p) + "\n" + json.dumps(second) + "\n")
    manifest = json.loads(parent.read_text())
    manifest["page_count"] = 2
    parent.write_text(json.dumps(manifest))
    result = parse_extraction(parent, data, load_source_registry(registry))
    report = json.loads((result.directory / "report.json").read_text())
    assert report["extraction_issues"][0] == {"page_number": 1, "issues": p["issues"]}


def test_parse_into_another_data_directory_with_a_copied_snapshot(extracted, tmp_path):
    import shutil

    data, parent, registry = extracted
    destination = tmp_path / "other-data"
    shutil.copytree(data / "raw", destination / "raw")
    result = parse_extraction(parent, destination, load_source_registry(registry))
    assert result.directory.parent == destination / "runs"


def test_document_without_articles_stays_visible_for_review(extracted):
    data, parent, registry = extracted
    path = parent.parent / "pages.jsonl"
    page = json.loads(path.read_text())
    page["blocks"] = page["blocks"][:1]
    path.write_text(json.dumps(page) + "\n")
    result = parse_extraction(parent, data, load_source_registry(registry))
    report = json.loads((result.directory / "report.json").read_text())
    assert report["article_count"] == 0
    assert report["review_status"] == "needs_review"
