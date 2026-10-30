"""Extract a verified snapshot into an isolated, inspectable local run."""

import hashlib
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import pdfplumber
from pdfminer.pdfexceptions import PDFException
from pdfplumber.utils.exceptions import PdfminerException

from atrag.ingestion.extraction import (
    join_tables,
    normalize_text,
    page_text,
    remove_margins,
)
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.ocr import TesseractOCR
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import read_snapshot
from atrag.schemas.ingestion.extraction import (
    ExtractedPage,
    ExtractedTable,
    ExtractionResult,
    ExtractionReview,
    ImageRegion,
    TablePart,
    TextBlock,
)
from atrag.schemas.ingestion.snapshots import SourceSnapshot

EXTRACTOR_VERSION = "1"


def _bbox(item: dict) -> tuple[float, float, float, float]:
    return tuple(item[key] for key in ("x0", "top", "x1", "bottom"))


def _text_blocks(page) -> list[TextBlock]:
    return [
        TextBlock(
            bbox=_bbox(line),
            text=line["text"],
            font_names=sorted({c["fontname"] for c in line["chars"]}),
            font_sizes=sorted({c["size"] for c in line["chars"]}),
        )
        for line in page.extract_text_lines()
    ]


def _table_parts(page) -> list[TablePart]:
    parts = []
    for i, table in enumerate(page.find_tables()):
        part_id = f"p{page.page_number}-table{i + 1}"
        parts.append(
            TablePart(
                part_id=part_id,
                table_id=part_id,
                page_number=page.page_number,
                bbox=table.bbox,
                rows=table.extract(),
                cell_bboxes=[row.cells for row in table.rows],
            )
        )
    return parts


def _repeated_margin_images(pdf) -> set[str]:
    counts = Counter()
    for page in pdf.pages:
        counts.update(
            {
                hashlib.sha256(i["stream"].get_data()).hexdigest()
                for i in page.images
                if i["top"] > page.height * 0.86
            }
        )
        page.close()
    return {
        digest
        for digest, count in counts.items()
        if count >= max(2, len(pdf.pages) / 2)
    }


def _image_regions(
    page, directory: Path, ocr: TesseractOCR | None, repeated: set[str]
) -> list[ImageRegion]:
    regions = []
    for i, item in enumerate(page.images):
        image_id = f"p{page.page_number}-image{i + 1}"
        digest = hashlib.sha256(item["stream"].get_data()).hexdigest()
        margin = item["top"] > page.height * 0.86 and digest in repeated
        region = ImageRegion(
            image_id=image_id,
            source_sha256=digest,
            bbox=_bbox(item),
            source_size=item["srcsize"],
            role="margin" if margin else "content",
            ocr_status="not_requested",
            review_status="not_applicable" if margin else "needs_review",
        )
        if not margin:
            image_path = Path("images") / f"{image_id}.png"
            page.crop(region.bbox).to_image(resolution=300).save(directory / image_path)
            region = region.model_copy(update={"image_path": str(image_path)})
            if ocr is not None:
                result = ocr.extract(directory / image_path)
                ocr_path = Path("images") / f"{image_id}.tsv"
                if result.tsv is not None:
                    (directory / ocr_path).write_text(result.tsv, encoding="utf-8")
                region = region.model_copy(
                    update={
                        "ocr_status": result.status,
                        "ocr_text": result.text,
                        "ocr_path": str(ocr_path) if result.tsv is not None else None,
                        "extraction_method": "tesseract:por",
                    }
                )
        regions.append(region)
    return regions


def _read_pages(
    path: Path, snapshot_id: str, directory: Path, ocr
) -> tuple[list, dict]:
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            metadata = pdf.metadata
            repeated = _repeated_margin_images(pdf)
            for page in pdf.pages:
                pages.append(
                    ExtractedPage(
                        snapshot_id=snapshot_id,
                        page_number=page.page_number,
                        width=page.width,
                        height=page.height,
                        raw_text=page.extract_text() or "",
                        blocks=_text_blocks(page),
                        tables=_table_parts(page),
                        images=_image_regions(page, directory, ocr, repeated),
                    )
                )
                page.close()
    except (PDFException, PdfminerException) as error:
        raise ValueError("Cannot parse the PDF snapshot") from error
    if not pages:
        raise ValueError("PDF has no pages")
    return remove_margins(pages), metadata


def _review_tables(
    pages: list[ExtractedPage], tables: list[ExtractedTable], review: ExtractionReview
) -> None:
    for entry in review.tables:
        if entry.page_number > len(pages):
            raise ValueError("Reviewed page does not exist")
        page = pages[entry.page_number - 1]
        if entry.source_kind == "text_table":
            _review_text_table(tables, entry)
            continue
        matching = [
            i
            for i in page.images
            if tuple(round(v, 2) for v in i.bbox)
            == tuple(round(v, 2) for v in entry.bbox)
        ]
        if len(matching) != 1:
            raise ValueError("Review must locate exactly one source image")
        image = matching[0]
        part_id = f"{image.image_id}-table"
        part = TablePart(
            part_id=part_id,
            table_id=part_id,
            page_number=entry.page_number,
            bbox=image.bbox,
            rows=[*entry.header_rows, *entry.rows],
            extraction_method="manual_transcription",
        )
        tables.append(
            ExtractedTable(
                table_id=part_id,
                parts=[part],
                header_rows=entry.header_rows,
                rows=entry.rows,
                context=entry.context,
                review_status="verified",
                review_note=entry.review_note,
            )
        )
        images = [
            i.model_copy(update={"review_status": "verified"})
            if i.image_id == image.image_id
            else i
            for i in page.images
        ]
        pages[entry.page_number - 1] = page.model_copy(
            update={
                "tables": [*page.tables, part],
                "images": images,
            }
        )


def _review_text_table(tables, entry) -> None:
    matching = [
        t
        for t in tables
        if t.parts[0].page_number == entry.page_number
        and tuple(round(v, 2) for v in t.parts[0].bbox)
        == tuple(round(v, 2) for v in entry.bbox)
    ]
    if len(matching) != 1:
        raise ValueError("Review must locate exactly one extracted table")
    table = matching[0]
    if (table.header_rows, table.rows) != (entry.header_rows, entry.rows):
        raise ValueError("Extracted table differs from the reviewed transcription")
    tables[tables.index(table)] = table.model_copy(
        update={"review_status": "verified", "review_note": entry.review_note}
    )


def _page_issues(page: ExtractedPage, tables: dict[str, ExtractedTable]) -> list[str]:
    issues = []
    if len(normalize_text(page.raw_text)) < 100:
        issues.append("short_text")
    for image in page.images:
        if image.review_status == "needs_review":
            issues.append(f"image_needs_review:{image.image_id}:{image.ocr_status}")
    intervals = sorted(
        [(b.bbox[1], b.bbox[3]) for b in page.blocks if b.role == "body"]
        + [(i.bbox[1], i.bbox[3]) for i in page.images if i.role == "content"]
        + [(t.bbox[1], t.bbox[3]) for t in page.tables]
    )
    bottom = page.height * 0.09
    for top, end in intervals:
        if top - bottom > page.height * 0.15:
            issues.append("large_blank_region")
        bottom = max(bottom, end)
    if any(tables[t.table_id].review_status == "needs_review" for t in page.tables):
        issues.append("table_needs_review")
    return sorted(set(issues))


def _save_artifacts(
    directory: Path, pages: list[ExtractedPage], tables: list[ExtractedTable]
) -> dict:
    by_id = {t.table_id: t for t in tables}
    pages = [
        p.model_copy(
            update={
                "issues": _page_issues(p, by_id),
                "text_normalized": page_text(p, by_id),
            }
        )
        for p in pages
    ]
    report = {
        "page_count": len(pages),
        "table_count": len(tables),
        "image_count": sum(len(p.images) for p in pages),
        "review_status": "needs_review" if any(p.issues for p in pages) else "verified",
        "issues": [
            {"page_number": p.page_number, "issues": p.issues}
            for p in pages
            if p.issues
        ],
    }
    write_json(
        directory / "tables.json",
        {"tables": [t.model_dump(mode="json") for t in tables]},
    )
    write_json(directory / "report.json", report)
    write_jsonl(directory / "pages.jsonl", pages)
    return report


def extract_snapshot(
    snapshot: SourceSnapshot,
    data_dir: Path,
    *,
    ocr: TesseractOCR | None = None,
    review_path: Path | None = None,
) -> ExtractionResult:
    with record_run(
        data_dir,
        stage="extract",
        snapshot_id=snapshot.snapshot_id,
        extractor_version=EXTRACTOR_VERSION,
        dependencies={
            name: version(name) for name in ("pdfplumber", "pdfminer.six", "pypdfium2")
        },
        ocr={"language": "por", "requested": ocr is not None},
    ) as run:
        directory, manifest = run.directory, run.manifest
        (directory / "images").mkdir()
        if ocr is not None:
            manifest["ocr"].update(ocr.metadata())
        stored = read_snapshot(
            data_dir / snapshot.local_path.parent,
            snapshot.snapshot_id,
            snapshot.size_bytes,
        )
        if stored != snapshot:
            raise ValueError("Snapshot manifest changed")
        pages, metadata = _read_pages(
            data_dir / snapshot.local_path, snapshot.snapshot_id, directory, ocr
        )
        pages, tables = join_tables(pages)
        if review_path is not None:
            raw_review = review_path.read_bytes()
            review = ExtractionReview.model_validate_json(raw_review)
            if review.snapshot_id != snapshot.snapshot_id:
                raise ValueError("Review checksum does not match the snapshot")
            _review_tables(pages, tables, review)
            (directory / "review.json").write_bytes(raw_review)
            manifest["review_sha256"] = hashlib.sha256(raw_review).hexdigest()
        report = _save_artifacts(directory, pages, tables)
        manifest.update(pdf_metadata=metadata, **report)
    return ExtractionResult(
        run.run_id, directory, len(pages), len(tables), report["review_status"]
    )
