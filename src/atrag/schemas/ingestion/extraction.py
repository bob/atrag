"""Located extraction artifacts for PDF coordinates and HTML source cells."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, PositiveInt

from atrag.schemas.materials import HTMLLocation

BBox = tuple[float, float, float, float]


class Artifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextBlock(Artifact):
    bbox: BBox
    text: str
    font_names: list[str] = Field(default_factory=list)
    font_sizes: list[float] = Field(default_factory=list)
    role: Literal["body", "header", "footer", "page_number"] = "body"


class TablePart(Artifact):
    part_id: str
    table_id: str
    page_number: PositiveInt
    bbox: BBox
    rows: list[list[str | None]]
    cell_bboxes: list[list[BBox | None]] = Field(default_factory=list)
    extraction_method: Literal["pdfplumber", "manual_transcription"] = "pdfplumber"


class HTMLTableCell(Artifact):
    text: str
    rowspan: PositiveInt = 1
    colspan: PositiveInt = 1


class HTMLTablePart(Artifact):
    part_id: str
    table_id: str
    location: HTMLLocation
    rows: list[list[str | None]]
    cells: list[list[HTMLTableCell]]
    extraction_method: Literal["html_dom"] = "html_dom"


class ExtractedTable[Part: (TablePart, HTMLTablePart) = TablePart](Artifact):
    table_id: str
    parts: list[Part]
    header_rows: list[list[str | None]]
    rows: list[list[str | None]]
    context: str
    review_status: Literal["needs_review", "verified"] = "needs_review"
    review_note: str | None = None


class ImageRegion(Artifact):
    image_id: str
    source_sha256: str
    bbox: BBox
    source_size: tuple[int, int]
    role: Literal["content", "margin"]
    image_path: str | None = None
    ocr_path: str | None = None
    ocr_text: str | None = None
    ocr_status: Literal["not_requested", "unavailable", "failed", "extracted"]
    extraction_method: str | None = None
    review_status: Literal["needs_review", "verified", "not_applicable"]


class ExtractedPage(Artifact):
    schema_version: Literal[1] = 1
    snapshot_id: str
    page_number: PositiveInt
    width: float
    height: float
    printed_page_labels: list[str] = Field(default_factory=list)
    raw_text: str
    text_normalized: str = ""
    blocks: list[TextBlock]
    tables: list[TablePart]
    images: list[ImageRegion]
    extraction_method: Literal["pdfplumber"] = "pdfplumber"
    issues: list[str] = Field(default_factory=list)


class ReviewedTable(Artifact):
    source_kind: Literal["image", "text_table"] = "image"
    page_number: PositiveInt
    bbox: BBox
    header_rows: list[list[str | None]]
    rows: list[list[str | None]]
    context: str
    review_note: str = Field(min_length=1)


class ExtractionReview(Artifact):
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    tables: list[ReviewedTable]


class ExtractionManifest(BaseModel):
    stage: Literal["extract"]
    status: Literal["succeeded"]
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: PositiveInt


class ExtractionTables(BaseModel):
    tables: list[ExtractedTable]


@dataclass(frozen=True)
class ExtractionResult:
    run_id: UUID
    directory: Path
    page_count: int
    table_count: int
    review_status: str
