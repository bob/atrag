"""Read source bytes only on download, confined to content-addressed storage."""

import re
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from atrag.admin.schemas import DocumentVersion, Download
from atrag.ingestion.source_files import source_file_names, validate_source_files
from atrag.schemas.ingestion.snapshots import SourceSnapshot


def checked_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes storage")
    return path


def source_directory(data_dir: Path, version: DocumentVersion) -> Path:
    if re.fullmatch(r"[a-f0-9]{64}", version.source_sha256) is None:
        raise ValueError("Invalid source digest")
    directory = checked_path(data_dir, f"raw/{version.source_sha256}")
    expected = {"pdf": "source.pdf", "html_collection": "collection.json"}[
        version.source_format
    ]
    source = checked_path(data_dir, version.local_path)
    if source != checked_path(directory, expected):
        raise ValueError("Source path does not match its content address")
    return directory


def source_available(data_dir: Path, version: DocumentVersion) -> bool:
    try:
        directory = source_directory(data_dir, version)
        return (
            checked_path(directory, "manifest.json").is_file()
            and checked_path(data_dir, version.local_path).is_file()
        )
    except (OSError, ValueError, KeyError):
        return False


def download_source(data_dir: Path, version: DocumentVersion) -> Download:
    directory = source_directory(data_dir, version)
    snapshot = SourceSnapshot.model_validate_json(
        checked_path(directory, "manifest.json").read_bytes()
    )
    if (snapshot.source_sha256, snapshot.size_bytes) != (
        version.source_sha256,
        version.size_bytes,
    ):
        raise ValueError("Stored source differs from document provenance")
    files = {
        name: checked_path(directory, name).read_bytes()
        for name in source_file_names(snapshot)
    }
    validate_source_files(snapshot, files)
    if snapshot.collection is None:
        return Download(
            files["source.pdf"], f"{version.source_sha256}.pdf", "application/pdf"
        )
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        for name, raw in files.items():
            archive.writestr(name, raw)
    return Download(
        output.getvalue(), f"{version.source_sha256}.zip", "application/zip"
    )
