import json

import pytest

from atrag.ingestion.extraction import normalize_text
from atrag.ingestion.services.extract import extract_snapshot
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.settings import PROJECT_ROOT


@pytest.fixture
def pdf_source(tmp_path):
    # A small real PDF exercises the library boundary without external files/tools.
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument.new()
    doc.new_page(595, 842)
    path = tmp_path / "blank.pdf"
    doc.save(path)
    doc.close()
    source = load_source_registry(PROJECT_ROOT / "data/sources/cirs.json")
    return source.model_copy(update={"local_path": path, "expected_sha256": None})


def test_blank_page_keeps_locator_and_reports_gap(pdf_source, tmp_path):
    data = tmp_path / "data"
    snapshot = import_snapshot(pdf_source, data).snapshot
    result = extract_snapshot(snapshot, data)
    page = json.loads((result.directory / "pages.jsonl").read_text())
    assert page["snapshot_id"] == snapshot.snapshot_id
    assert page["page_number"] == 1
    assert page["width"] == 595
    assert page["height"] == 842
    assert page["raw_text"] == page["text_normalized"] == ""
    assert "short_text" in page["issues"]
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["status"] == "succeeded"
    assert manifest["page_count"] == 1
    assert manifest["snapshot_id"] == snapshot.snapshot_id
    assert manifest["extractor_version"]
    assert manifest["dependencies"]["pdfplumber"]
    assert manifest["review_status"] == "needs_review"


def test_each_extraction_is_isolated_and_deterministic(pdf_source, tmp_path):
    data = tmp_path / "data"
    snapshot = import_snapshot(pdf_source, data).snapshot
    first = extract_snapshot(snapshot, data)
    second = extract_snapshot(snapshot, data)
    assert first.directory != second.directory
    assert (first.directory / "pages.jsonl").read_bytes() == (
        second.directory / "pages.jsonl"
    ).read_bytes()
    assert first.run_id != second.run_id


def test_corrupt_snapshot_stops_extraction(pdf_source, tmp_path):
    data = tmp_path / "data"
    snapshot = import_snapshot(pdf_source, data).snapshot
    (data / snapshot.local_path).write_bytes(b"%PDF-broken")
    with pytest.raises(ValueError, match="checksum"):
        extract_snapshot(snapshot, data)
    manifests = [
        json.loads(p.read_text()) for p in (data / "runs").glob("*/manifest.json")
    ]
    failed = [m for m in manifests if m["status"] == "failed"]
    assert len(failed) == 1
    assert not list((data / "runs").glob("*/pages.jsonl"))


def test_normalization_preserves_legal_spelling_and_hard_hyphens():
    assert normalize_text("não se aplica ao n.º 1,\nalínea a), 12,50 % e 68-A") == (
        "não se aplica ao n.º 1, alínea a), 12,50 % e 68-A"
    )
    assert normalize_text("tributa\u00ad\nção") == "tributação"
    assert normalize_text("Decreto-\nLei") == "Decreto- Lei"


def test_invalid_pdf_records_failure(pdf_source, tmp_path):
    pdf_source.local_path.write_bytes(b"%PDF-not-a-real-document")
    data = tmp_path / "data"
    snapshot = import_snapshot(pdf_source, data).snapshot
    with pytest.raises(ValueError):
        extract_snapshot(snapshot, data)
    [failed] = [
        json.loads(p.read_text())
        for p in (data / "runs").glob("*/manifest.json")
        if json.loads(p.read_text())["status"] == "failed"
    ]
    assert failed["error_type"] == "ValueError"
    assert not (data / "runs" / failed["run_id"] / "pages.jsonl").exists()


@pytest.fixture
def located_source(pdf_source, make_pdf, pdf_text):
    def grid(top, bottom, header):
        drawing = ""
        for x in [100, 300, 400, 500]:
            drawing += f"{x} {842 - top} m {x} {842 - bottom} l S\n"
        ys = [top, bottom] if not header else [top, (top + bottom) / 2, bottom]
        for y in ys:
            drawing += f"100 {842 - y} m 500 {842 - y} l S\n"
        if header:
            drawing += pdf_text(top + 10, "Rendimento (euros)", 110)
            drawing += pdf_text(top + 10, "Normal (%)", 310)
            drawing += pdf_text(top + 10, "Média (%)", 410)
        y = bottom - 25
        drawing += pdf_text(y, "Até 8 342" if header else "Superior a 8 342", 110)
        drawing += pdf_text(y, "12,50" if header else "15,70", 310)
        drawing += pdf_text(y, "12,500" if header else "13,579", 410)
        return drawing

    def margin(n):
        return (
            pdf_text(50, "Repeated heading")
            + pdf_text(750, "Repeated footer")
            + pdf_text(785, str(n), 540)
        )

    first = margin(10) + pdf_text(200, "Artigo 68.º", 280)
    first += pdf_text(220, "1 - As taxas constam da tabela seguinte:") + grid(
        640, 720, True
    )
    second = margin(11) + grid(80, 120, False)
    second += pdf_text(170, "Artigo 68.º-A", 280)
    second += pdf_text(210, "1 - não se aplica ao n.º 1, alínea a), 2,5 %.")
    second += "q 240 0 0 80 150 462 cm /Im1 Do Q\n"  # top=300, bottom=380
    return pdf_source.model_copy(update={"local_path": make_pdf([first, second])})


def test_located_text_tables_images_and_review(located_source, tmp_path):
    from atrag.schemas.ingestion.ocr import OCRResult

    class OCR:
        def metadata(self):
            return {"engine": "synthetic OCR"}

        def extract(self, image_path):
            assert image_path.read_bytes().startswith(b"\x89PNG")
            return OCRResult("extracted", "Taxa 25", "raw OCR TSV")

    data = tmp_path / "data"
    snapshot = import_snapshot(located_source, data).snapshot
    result = extract_snapshot(snapshot, data, ocr=OCR())
    pages = [
        json.loads(line)
        for line in (result.directory / "pages.jsonl").read_text().splitlines()
    ]
    [table] = json.loads((result.directory / "tables.json").read_text())["tables"]
    assert table["header_rows"] == [["Rendimento (euros)", "Normal (%)", "Média (%)"]]
    assert table["rows"] == [
        ["Até 8 342", "12,50", "12,500"],
        ["Superior a 8 342", "15,70", "13,579"],
    ]
    assert [p["page_number"] for p in table["parts"]] == [1, 2]
    assert table["context"].startswith("Artigo 68.º\n1 -")
    assert table["parts"][0]["cell_bboxes"]
    assert table["table_id"] == table["parts"][0]["part_id"] == "p1-table1"
    assert pages[0]["printed_page_labels"] == ["10"]
    assert pages[1]["printed_page_labels"] == ["11"]
    for page in pages:
        assert "Repeated heading" in page["raw_text"]
        assert "Repeated heading" not in page["text_normalized"]
        assert "Repeated footer" not in page["text_normalized"]
    assert "Normal (%)" in pages[1]["text_normalized"]
    assert "não se aplica ao n.º 1, alínea a), 2,5 %." in pages[1]["text_normalized"]
    block = next(b for b in pages[1]["blocks"] if b["text"].startswith("Artigo"))
    assert block["font_names"] == ["Helvetica"]
    assert block["font_sizes"] == [10]
    [image] = pages[1]["images"]
    assert image["image_id"] == "p2-image1"
    assert image["bbox"] == [150, 300, 390, 380]
    assert image["source_size"] == [1, 1]
    assert image["ocr_text"] == "Taxa 25"
    assert (result.directory / image["ocr_path"]).read_text() == "raw OCR TSV"
    assert "Taxa 25" not in pages[1]["text_normalized"]

    review = {
        "snapshot_id": snapshot.snapshot_id,
        "tables": [
            {
                "source_kind": "text_table",
                "page_number": 1,
                "bbox": table["parts"][0]["bbox"],
                "header_rows": table["header_rows"],
                "rows": table["rows"],
                "context": "Article 68, paragraph 1",
                "review_note": "Synthetic reference",
            },
            {
                "page_number": 2,
                "bbox": image["bbox"],
                "header_rows": [["Rendimento (euros)", "Taxa (%)"]],
                "rows": [["Até 250 000", "2,5"]],
                "context": "Article 68-A, paragraph 1",
                "review_note": "Synthetic reference",
            },
        ],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    reviewed = extract_snapshot(snapshot, data, ocr=OCR(), review_path=path)
    p1, p2 = [
        json.loads(line)
        for line in (reviewed.directory / "pages.jsonl").read_text().splitlines()
    ]
    tables = json.loads((reviewed.directory / "tables.json").read_text())["tables"]
    assert all(t["review_status"] == "verified" for t in tables)
    assert all("table_needs_review" not in p["issues"] for p in [p1, p2])
    assert p2["images"][0]["ocr_text"] == "Taxa 25"
    assert p2["images"][0]["review_status"] == "verified"
    assert p2["text_normalized"].endswith("Até 250 000 | 2,5")
    assert (reviewed.directory / "review.json").read_bytes() == path.read_bytes()


@pytest.mark.parametrize("problem", ["checksum", "page", "image", "table", "rows"])
def test_wrong_review_is_rejected(located_source, tmp_path, problem):
    data = tmp_path / "data"
    snapshot = import_snapshot(located_source, data).snapshot
    review = {
        "snapshot_id": snapshot.snapshot_id,
        "tables": [
            {
                "page_number": 2,
                "bbox": [150, 300, 390, 380],
                "header_rows": [["Taxa"]],
                "rows": [["2,5"]],
                "context": "synthetic",
                "review_note": "Synthetic reference",
            }
        ],
    }
    if problem == "checksum":
        review["snapshot_id"] = "0" * 64
    elif problem == "page":
        review["tables"][0]["page_number"] = 3
    elif problem == "image":
        review["tables"][0]["bbox"][0] = 10
    elif problem == "table":
        review["tables"][0]["source_kind"] = "text_table"
    else:
        review["tables"][0].update(
            source_kind="text_table", page_number=1, bbox=[100, 640, 500, 720]
        )
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        extract_snapshot(snapshot, data, review_path=path)
    assert not list((data / "runs").glob("*/pages.jsonl"))


def test_repeated_footer_images_are_not_ocr_content(
    pdf_source, make_pdf, pdf_text, tmp_path
):
    content = pdf_text(100, "Body text") + "q 84 0 0 23 85 69 cm /Im1 Do Q\n"
    path = make_pdf(
        [content, content, content, content + "q 240 0 0 80 150 462 cm /Im1 Do Q\n"]
    )
    source = pdf_source.model_copy(update={"local_path": path})
    data = tmp_path / "data"
    snapshot = import_snapshot(source, data).snapshot
    result = extract_snapshot(snapshot, data)
    pages = [
        json.loads(line)
        for line in (result.directory / "pages.jsonl").read_text().splitlines()
    ]
    for page in pages:
        assert page["images"][0]["role"] == "margin"
        assert page["images"][0]["review_status"] == "not_applicable"
        assert page["images"][0]["image_path"] is None
    assert pages[-1]["images"][1]["role"] == "content"
    assert pages[-1]["images"][1]["review_status"] == "needs_review"
    assert pages[-1]["images"][1]["image_path"] == "images/p4-image2.png"


def test_empty_pdf_and_changed_manifest_are_rejected(pdf_source, make_pdf, tmp_path):
    from datetime import timedelta

    data = tmp_path / "data"
    snapshot = import_snapshot(
        pdf_source.model_copy(update={"local_path": make_pdf([])}), data
    ).snapshot
    with pytest.raises(ValueError, match="no pages"):
        extract_snapshot(snapshot, data)
    changed = snapshot.model_copy(
        update={"ingested_at": snapshot.ingested_at + timedelta(seconds=1)}
    )
    with pytest.raises(ValueError, match="manifest changed"):
        extract_snapshot(changed, data)


@pytest.mark.parametrize(
    "positions,roles",
    [
        ([900, 860, None, None], ["content", "content"]),
        ([900, 900, None, None], ["margin", "margin"]),
        ([900, 900, 860, None], ["margin", "margin", "content"]),
    ],
)
def test_footer_image_detection_uses_repetition_and_position(
    pdf_source, make_pdf, tmp_path, positions, roles
):
    streams = [
        "" if top is None else f"q 40 0 0 20 100 {1000 - top - 20} cm /Im1 Do Q\n"
        for top in positions
    ]
    source = pdf_source.model_copy(
        update={"local_path": make_pdf(streams, height=1000)}
    )
    data = tmp_path / "data"
    snapshot = import_snapshot(source, data).snapshot
    result = extract_snapshot(snapshot, data)
    pages = [
        json.loads(line)
        for line in (result.directory / "pages.jsonl").read_text().splitlines()
    ]
    assert [image["role"] for page in pages for image in page["images"]] == roles


@pytest.mark.parametrize("problem", ["page", "bbox"])
def test_text_table_review_requires_both_page_and_bbox(
    located_source, tmp_path, problem
):
    data = tmp_path / "data"
    snapshot = import_snapshot(located_source, data).snapshot
    first = extract_snapshot(snapshot, data)
    [table] = json.loads((first.directory / "tables.json").read_text())["tables"]
    entry = {
        "source_kind": "text_table",
        "page_number": 1,
        "bbox": table["parts"][0]["bbox"],
        "header_rows": table["header_rows"],
        "rows": table["rows"],
        "context": "synthetic",
        "review_note": "Synthetic reference",
    }
    if problem == "page":
        entry["page_number"] = 2
    else:
        entry["bbox"][0] += 5
    path = tmp_path / "review.json"
    path.write_text(
        json.dumps({"snapshot_id": snapshot.snapshot_id, "tables": [entry]})
    )
    with pytest.raises(ValueError, match="locate exactly one extracted table"):
        extract_snapshot(snapshot, data, review_path=path)
