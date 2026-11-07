"""FAQ uses the same bounded parts and portable package as PDF sources."""

import json
from pathlib import Path

import pytest
from fixtures.faq_pipeline import make_faq_package, prepare_faq

from atrag.ingestion.part_validation import reassemble_unit
from atrag.ingestion.services.chunks import chunk_parsed
from atrag.ingestion.services.faq_extract import extract_collection
from atrag.ingestion.services.structure import parse_extraction
from atrag.schemas.ingestion.fragments import ChunkParameters, SearchFragment
from atrag.schemas.ingestion.structure import LegalUnit


@pytest.mark.parametrize(
    "faq_snapshot",
    [
        {},
        {"answer": "<p>Only if resident:</p>" + "<p>Complete condition.</p>" * 100},
        {
            "answer": "<p>Rates:</p><table><tr><th>Value</th><th>Rate</th></tr>"
            "<tr><td>100</td><td>2%</td></tr><tr><td>200</td><td>3%</td></tr></table>"
        },
    ],
    indirect=True,
)
def test_faq_parts_reassemble_answer_and_repeat_question(faq_snapshot, tmp_path):
    _, parsed, chunked = prepare_faq(
        faq_snapshot, tmp_path, ChunkParameters(target_tokens=90, max_tokens=180)
    )
    units = [
        LegalUnit.model_validate_json(l)
        for l in (parsed.directory / "legal_units.jsonl").read_bytes().splitlines()
    ]
    parts = [
        SearchFragment.model_validate_json(l)
        for l in (chunked.directory / "fragments.jsonl").read_bytes().splitlines()
    ]
    entry = next(u for u in units if u.kind == "faq_entry")
    assert chunked.review_status == "ready"
    assert reassemble_unit(entry, parts) == entry.faq.answer
    assert entry.legal_unit_id == "pt-at-faq:faqs-00267:42"
    assert all(p.faq == entry.faq and p.location == entry.location for p in parts)
    assert all(p.locators == [] and p.page_start is p.page_end is None for p in parts)
    assert all(entry.faq.question in p.embedding_text for p in parts)
    assert all("IRS" in p.embedding_text and "Topic" in p.embedding_text for p in parts)
    assert all(p.token_count <= 180 and p.article is None for p in parts)
    if "Only if" in entry.text_original:
        assert len(parts) > 1
        assert all("Only if resident:" in p.embedding_text for p in parts)
    if entry.table_ids:
        assert {t.table_id for p in parts for t in p.tables} == set(entry.table_ids)
        assert parts[0].tables[0].parts[0].extraction_method == "html_dom"
    repeated = chunk_parsed(
        parsed.directory / "manifest.json",
        tmp_path,
        faq_snapshot.source,
        ChunkParameters(target_tokens=90, max_tokens=180),
    )
    assert (repeated.directory / "fragments.jsonl").read_bytes() == (
        chunked.directory / "fragments.jsonl"
    ).read_bytes()


def test_separate_questions_are_never_packed_together(tmp_path, request):
    from fixtures.faq_pipeline import URL, card

    from atrag.ingestion.chunk import chunk_structure
    from atrag.ingestion.faq_extract import parse_faq_page
    from atrag.ingestion.faq_structure import faq_structure
    from atrag.ingestion.services.sources import load_source_registry

    page = parse_faq_page((card() + card("43", position=2)).encode(), URL, ("IRS",))
    source = load_source_registry(Path("data/sources/faq.json"))
    tree = faq_structure(page.faqs, [], "a" * 64, source)
    result = chunk_structure(
        tree,
        source,
        parser_version="faq-1",
        parameters=ChunkParameters(),
        count_tokens=len,
    )
    assert len(result.fragments) == 2
    assert {p.faq.faq_id for p in result.fragments} == {"42", "43"}
    assert all(len(p.legal_unit_ids) == 1 for p in result.fragments)


def test_damaged_extraction_cannot_be_parsed(faq_snapshot, tmp_path):
    extracted = Path(extract_collection(faq_snapshot, tmp_path)["directory"])
    (extracted / "faqs.jsonl").write_text("broken")
    with pytest.raises(ValueError, match="checksum"):
        parse_extraction(extracted / "manifest.json", tmp_path, faq_snapshot.source)


@pytest.mark.parametrize(
    "faq_snapshot", [{"answer": "<img src='meaningful.png'>Answer"}], indirect=True
)
def test_unverified_extraction_is_blocked(faq_snapshot, tmp_path):
    extracted = Path(extract_collection(faq_snapshot, tmp_path)["directory"])
    assert (
        json.loads((extracted / "report.json").read_bytes())["review_status"]
        == "needs_review"
    )
    with pytest.raises(ValueError, match="verified"):
        parse_extraction(extracted / "manifest.json", tmp_path, faq_snapshot.source)


@pytest.fixture
def faq_package(faq_snapshot, tmp_path):
    return make_faq_package(faq_snapshot, tmp_path)


def test_html_package_is_portable_and_preserves_legacy_cirs_format(
    faq_package, tmp_path
):
    import shutil

    from atrag.ingestion.services.packages import load_package

    first = load_package(faq_package)
    assert first.manifest.schema_version == 3
    assert first.manifest.source_format == "html_collection"
    assert set(first.manifest.source_files) == {
        "collection.json",
        "page-0.html",
        "page-1.html",
    }
    shutil.copytree(faq_package.parent, tmp_path / "relocated")
    assert load_package(tmp_path / "relocated/manifest.json") == first


@pytest.mark.parametrize(
    "damage", ["format", "composition", "legacy", "missing_source", "identity"]
)
def test_html_package_provenance_is_checked(faq_package, damage):
    from atrag.ingestion.services.packages import load_package

    manifest = json.loads(faq_package.read_bytes())
    if damage == "format":
        manifest["source_format"] = "pdf"
    elif damage == "composition":
        manifest["source_files"] = ["collection.json"]
    elif damage == "legacy":
        manifest["schema_version"] = 2
    elif damage == "missing_source":
        del manifest["files"]["page-1.html"]
    else:
        manifest["snapshot_id"] = "a" * 64
    faq_package.write_text(json.dumps(manifest))
    with pytest.raises(
        ValueError, match="format|composition|version 3|required files|identity"
    ):
        load_package(faq_package)


@pytest.mark.parametrize("damage", ["collection", "html"])
def test_rehashed_html_source_damage_is_checked(faq_package, damage):
    from fixtures.prepared import digest

    from atrag.ingestion.services.packages import load_package

    manifest = json.loads(faq_package.read_bytes())
    name = "collection.json" if damage == "collection" else "page-1.html"
    path = faq_package.parent / name
    if damage == "collection":
        collection = json.loads(path.read_bytes())
        collection["pages"][0]["headers"] = {"other": "value"}
        path.write_text(json.dumps(collection))
    else:
        path.write_bytes(b"broken")
    manifest["files"][name] = digest(path.read_bytes())
    faq_package.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="collection|HTML checksum"):
        load_package(faq_package)


@pytest.mark.parametrize(
    "damage", ["unready", "omission", "missing_entry", "answer", "location"]
)
def test_faq_document_must_keep_all_original_answers(faq_package, damage):
    from dataclasses import replace

    from atrag.ingestion.package_validation import _validate_faq_coverage
    from atrag.ingestion.part_validation import validate_part
    from atrag.ingestion.services.packages import load_package
    from atrag.schemas.ingestion.packages import ChunkQuality

    document = load_package(faq_package)
    report = ChunkQuality.model_validate(document.reports["chunk"])
    parts = {f.fragment_id: f for f in document.fragments}
    if damage == "unready":
        report = report.model_copy(update={"review_status": "needs_review"})
    elif damage == "omission":
        report = report.model_copy(update={"omissions": [{"reason": "lost"}]})
    elif damage == "missing_entry":
        parts = {}
    elif damage == "answer":
        document = replace(
            document,
            units=[
                u.model_copy(update={"text_original": "Changed"})
                if u.faq is not None
                else u
                for u in document.units
            ],
        )
    else:
        unit = next(u for u in document.units if u.faq is not None)
        part = document.fragments[0]
        changed = part.model_copy(
            update={"location": part.location.model_copy(update={"anchor": "other"})}
        )
        with pytest.raises(ValueError, match="HTML provenance"):
            validate_part(unit, changed)
        return
    with pytest.raises(ValueError, match="ready|complete|original answer"):
        _validate_faq_coverage(document, report, parts)


@pytest.mark.parametrize("damage", ["source", "checksum", "count", "duplicate"])
def test_faq_parse_checks_registry_and_entry_set(faq_snapshot, tmp_path, damage):
    from fixtures.prepared import refresh_manifest

    from atrag.ingestion.services.artifacts import write_json

    extracted = Path(extract_collection(faq_snapshot, tmp_path)["directory"])
    source = faq_snapshot.source
    manifest = json.loads((extracted / "manifest.json").read_bytes())
    if damage == "source":
        source = source.model_copy(update={"publisher": "Other"})
    elif damage == "checksum":
        source = source.model_copy(update={"expected_sha256": "a" * 64})
    elif damage == "count":
        manifest["question_count"] = 2
    else:
        path = extracted / "faqs.jsonl"
        path.write_bytes(path.read_bytes() * 2)
        manifest["question_count"] = 2
    write_json(extracted / "manifest.json", manifest)
    refresh_manifest(extracted / "manifest.json")
    with pytest.raises(ValueError, match="source|registry|count or identity"):
        parse_extraction(extracted / "manifest.json", tmp_path, source)


def test_repreparing_same_html_reuses_prepared_version(faq_snapshot, tmp_path):
    from atrag.ingestion.services.packages import load_package

    first_path = make_faq_package(faq_snapshot, tmp_path)
    original = first_path.read_bytes()
    first = load_package(first_path)
    second_path = make_faq_package(faq_snapshot, tmp_path)
    second = load_package(second_path)
    assert second.snapshot.snapshot_id == first.snapshot.snapshot_id
    assert {p.fragment_id for p in second.fragments} == {
        p.fragment_id for p in first.fragments
    }
    assert second_path.read_bytes() == original


@pytest.mark.parametrize(
    "faq_snapshot", [{"answer": "<table><tr><td>IVA</td></tr></table>"}], indirect=True
)
def test_html_table_must_exist_in_its_original_answer(faq_snapshot, tmp_path):
    from atrag.ingestion.table_parts import table_rows
    from atrag.schemas.ingestion.extraction import MaterialTables

    _, parsed, _ = prepare_faq(faq_snapshot, tmp_path)
    units = [
        LegalUnit.model_validate_json(line)
        for line in (parsed.directory / "legal_units.jsonl").read_bytes().splitlines()
    ]
    unit = next(u for u in units if u.faq is not None)
    tables = MaterialTables.model_validate_json(
        (parsed.directory / "tables.json").read_bytes()
    ).tables
    with pytest.raises(ValueError, match="HTML table cells are absent"):
        table_rows(
            unit.model_copy(update={"text_original": "missing"}),
            {t.table_id: t for t in tables},
        )


@pytest.mark.parametrize(
    "faq_snapshot",
    [{"answer": "<p>Answer</p><table><tr><td>&nbsp;</td></tr></table>"}],
    indirect=True,
)
def test_empty_layout_table_has_no_invented_text_range(faq_snapshot, tmp_path):
    from atrag.ingestion.table_parts import table_rows
    from atrag.schemas.ingestion.extraction import MaterialTables

    _, parsed, chunked = prepare_faq(faq_snapshot, tmp_path)
    units = [
        LegalUnit.model_validate_json(line)
        for line in (parsed.directory / "legal_units.jsonl").read_bytes().splitlines()
    ]
    unit = next(u for u in units if u.faq is not None)
    tables = MaterialTables.model_validate_json(
        (parsed.directory / "tables.json").read_bytes()
    ).tables
    assert len(tables) == 1
    assert table_rows(unit, {t.table_id: t for t in tables}) == []
    assert chunked.review_status == "ready"


@pytest.mark.parametrize(
    "faq_snapshot",
    [{"answer": "<p>IVA</p><table><tr><td>IVA</td></tr></table>"}],
    indirect=True,
)
def test_ambiguous_table_text_cannot_invent_a_location(faq_snapshot, tmp_path):
    with pytest.raises(ValueError, match="ambiguous HTML table"):
        prepare_faq(faq_snapshot, tmp_path)


def test_synthetic_smoke_rejects_non_test_database_before_writing(tmp_path):
    import runpy

    run = runpy.run_path("tools/faq_ingestion_smoke.py")["run"]
    with pytest.raises(ValueError, match="atrag_test_"):
        run(
            tmp_path / "missing.json",
            tmp_path / "output",
            "postgresql://localhost/production",
        )
    assert not (tmp_path / "output").exists()
