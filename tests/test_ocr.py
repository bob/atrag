import subprocess

import pytest

from atrag.ingestion.services.ocr import TesseractOCR, tsv_text

TSV = "level\tblock_num\tpar_num\tline_num\ttext\n1\t0\t0\t0\t\n4\t2\t2\t2\tignored non-word\n5\t1\t1\t1\tTaxa\n5\t1\t1\t1\t2,5\n5\t1\t1\t2\t5\n"


def test_tsv_preserves_words_lines_and_decimal_commas():
    assert tsv_text(TSV) == "Taxa 2,5\n5"


def test_local_ocr_invocation_keeps_raw_tsv(tmp_path, monkeypatch):
    path = tmp_path / "Português.png"
    path.write_bytes(b"PNG bytes")

    def run(command, **kwargs):
        assert command[:3] == ["tesseract", "stdin", "stdout"]
        assert command[command.index("-l") + 1] == "por"
        assert command[command.index("--psm") + 1] == "6"
        assert command[-2:] == ["--tessdata-dir", str(tmp_path)]
        assert kwargs == {
            "input": b"PNG bytes",
            "capture_output": True,
            "timeout": 4,
            "check": False,
        }
        return subprocess.CompletedProcess(command, 0, TSV.encode(), b"")

    monkeypatch.setattr(subprocess, "run", run)
    result = TesseractOCR(tmp_path, timeout=4).extract(path)
    assert result.status == "extracted"
    assert result.text == "Taxa 2,5\n5"
    assert result.tsv == TSV


@pytest.mark.parametrize("error", ["missing", "timeout", "exit"])
def test_ocr_failure_is_explicit_and_never_supplies_text(tmp_path, monkeypatch, error):
    path = tmp_path / "image.png"
    path.write_bytes(b"image")

    def run(command, **kwargs):
        assert "--tessdata-dir" not in command
        if error == "missing":
            raise FileNotFoundError
        if error == "timeout":
            raise subprocess.TimeoutExpired(command, 60)
        return subprocess.CompletedProcess(command, 1, b"", b"missing language")

    monkeypatch.setattr(subprocess, "run", run)
    result = TesseractOCR().extract(path)
    assert result.status == ("unavailable" if error == "missing" else "failed")
    assert result.text is result.tsv is None


def test_ocr_metadata_records_engine_and_exact_language_model(tmp_path, monkeypatch):
    import hashlib

    (tmp_path / "por.traineddata").write_bytes(b"synthetic Portuguese model")

    def run(command, **kwargs):
        assert command == ["tesseract", "--version"]
        assert kwargs["timeout"] == 5
        assert kwargs["check"] is False
        return subprocess.CompletedProcess(
            command, 0, "tesseract test\nleptonica test", ""
        )

    monkeypatch.setattr(subprocess, "run", run)
    assert TesseractOCR(tmp_path).metadata() == {
        "engine": "tesseract",
        "version": "tesseract test",
        "language": "por",
        "psm": 6,
        "render_dpi": 300,
        "traineddata_sha256": hashlib.sha256(b"synthetic Portuguese model").hexdigest(),
    }


@pytest.mark.parametrize("problem", ["missing", "timeout", "exit"])
def test_ocr_metadata_keeps_unknown_values_explicit(monkeypatch, problem):
    def run(command, **kwargs):
        if problem == "missing":
            raise FileNotFoundError
        if problem == "timeout":
            raise subprocess.TimeoutExpired(command, 5)
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    result = TesseractOCR().metadata()
    assert result["version"] is result["traineddata_sha256"] is None
