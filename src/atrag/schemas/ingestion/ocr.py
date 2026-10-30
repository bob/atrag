"""Results returned by local OCR adapters."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OCRResult:
    status: str
    text: str | None = None
    tsv: str | None = None
