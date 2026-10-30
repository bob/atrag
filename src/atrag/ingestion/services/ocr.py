"""Local Portuguese Tesseract adapter; preserve TSV and never repair its numbers."""

import csv
import hashlib
import io
import subprocess
from dataclasses import dataclass
from pathlib import Path

from atrag.schemas.ingestion.ocr import OCRResult


def tsv_text(tsv: str) -> str:
    lines: dict[tuple[str, ...], list[str]] = {}
    for word in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        if word["level"] == "5" and word["text"].strip():
            key = tuple(word[k] for k in ("block_num", "par_num", "line_num"))
            lines.setdefault(key, []).append(word["text"])
    return "\n".join(" ".join(words) for words in lines.values())


@dataclass(frozen=True)
class TesseractOCR:
    tessdata_dir: Path | None = None
    timeout: float = 60

    def metadata(self) -> dict:
        metadata = {
            "engine": "tesseract",
            "version": None,
            "language": "por",
            "psm": 6,
            "render_dpi": 300,
            "traineddata_sha256": None,
        }
        try:
            result = subprocess.run(
                ["tesseract", "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                metadata["version"] = result.stdout.splitlines()[0]
        except OSError, subprocess.TimeoutExpired:
            pass
        if self.tessdata_dir is not None:
            path = self.tessdata_dir / "por.traineddata"
            if path.is_file():
                metadata["traineddata_sha256"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
        return metadata

    def extract(self, image_path: Path) -> OCRResult:
        command = [
            "tesseract",
            "stdin",
            "stdout",
            "-l",
            "por",
            "--psm",
            "6",
            "-c",
            "tessedit_create_tsv=1",
        ]
        if self.tessdata_dir is not None:
            command.extend(["--tessdata-dir", str(self.tessdata_dir)])
        try:
            # stdin also works with Tesseract installations that cannot open Unicode paths.
            result = subprocess.run(
                command,
                check=False,
                input=image_path.read_bytes(),
                capture_output=True,
                timeout=self.timeout,
            )
        except FileNotFoundError:
            return OCRResult(status="unavailable")
        except subprocess.TimeoutExpired:
            return OCRResult(status="failed")
        if result.returncode:
            return OCRResult(status="failed")
        tsv = result.stdout.decode("utf-8")
        return OCRResult(status="extracted", text=tsv_text(tsv), tsv=tsv)
