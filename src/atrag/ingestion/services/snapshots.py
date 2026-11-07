"""Content-addressed PDF and HTML import with a separate record for each attempt."""

import errno
import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from atrag.ingestion.faq_navigation import inspect_html, normalize_faq_url
from atrag.ingestion.services.artifacts import write_json
from atrag.ingestion.services.runs import record_run
from atrag.schemas.ingestion.collections import HTMLCollection, HTMLPage
from atrag.schemas.ingestion.snapshots import SnapshotImport, SourceSnapshot
from atrag.schemas.ingestion.sources import SourceDefinition, SourceFormat
from atrag.settings import PROJECT_ROOT


def _pdf_fingerprint(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("Source must have a PDF signature")
        stream.seek(0)
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return digest, path.stat().st_size


def read_snapshot(directory: Path, digest: str, size: int) -> SourceSnapshot:
    """Validate stored provenance and bytes against the expected content address."""
    snapshot = SourceSnapshot.model_validate_json(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    if (snapshot.snapshot_id, snapshot.size_bytes) != (digest, size):
        raise ValueError("Stored snapshot manifest does not match the source")
    if snapshot.collection is None:
        if _pdf_fingerprint(directory / "source.pdf") != (digest, size):
            raise ValueError("Stored snapshot bytes do not match the checksum")
    else:
        collection = read_html_collection(directory)
        if collection != snapshot.collection:
            raise ValueError("Stored collection differs from the snapshot manifest")
    return snapshot


def verify_html_page(directory: Path, page: HTMLPage) -> None:
    path = directory / page.local_path
    if not path.resolve().is_relative_to(directory.resolve()):
        raise ValueError("HTML page escapes the collection directory")
    raw = path.read_bytes()
    if (hashlib.sha256(raw).hexdigest(), len(raw)) != (page.sha256, page.size_bytes):
        raise ValueError("Stored HTML page bytes do not match the checksum/size")
    if normalize_faq_url(str(page.url), str(page.url)) != str(page.url):
        raise ValueError("HTML page URL is outside canonical FAQ navigation")
    if normalize_faq_url(str(page.final_url), str(page.final_url)) is None:
        raise ValueError("HTML page final URL is outside FAQ navigation")
    inspection = inspect_html(raw, str(page.final_url))
    if tuple(map(str, page.navigation_urls)) != inspection.navigation_urls:
        raise ValueError("Saved navigation differs from the source HTML")


def read_html_collection(directory: Path) -> HTMLCollection:
    collection = HTMLCollection.model_validate_json(
        (directory / "collection.json").read_bytes()
    )
    for page in collection.pages:
        verify_html_page(directory, page)
    return collection


def store_html_snapshot(
    source: SourceDefinition, data_dir: Path, directory: Path
) -> tuple[SourceSnapshot, bool]:
    collection = read_html_collection(directory)
    digest, size = collection.source_sha256, collection.size_bytes
    if source.expected_sha256 is not None and source.expected_sha256 != digest:
        raise ValueError("Collection checksum does not match the registry")
    raw = data_dir / "raw"
    raw.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".snapshot-", dir=raw) as temporary:
        staging = Path(temporary)
        for page in collection.pages:
            target = staging / page.local_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(directory / page.local_path, target)
        write_json(staging / "collection.json", collection.model_dump(mode="json"))
        snapshot = SourceSnapshot(
            schema_version=2,
            snapshot_id=digest,
            source_sha256=digest,
            local_path=Path("raw")
            / digest
            / SourceFormat.HTML_COLLECTION.snapshot_filename,
            size_bytes=size,
            ingested_at=datetime.now(UTC),
            source=source,
            collection=collection,
        )
        write_json(staging / "manifest.json", snapshot.model_dump(mode="json"))
        destination = raw / digest
        reused = destination.exists()
        if not reused:
            reused = _publish_snapshot(staging, destination)
        return read_snapshot(destination, digest, size), reused


def _publish_snapshot(staging: Path, destination: Path) -> bool:
    """Publish both files together; a competing complete directory wins intact."""
    try:
        staging.rename(destination)
    except OSError as error:
        if error.errno not in (errno.EEXIST, errno.ENOTEMPTY):
            raise
        return True
    return False


def _store_snapshot(
    source: SourceDefinition, data_dir: Path, project_root: Path
) -> tuple[SourceSnapshot, bool]:
    if source.source_format == SourceFormat.HTML_COLLECTION:
        path = project_root / source.local_path
        if path.is_file() and path.name != "collection.json":
            raise ValueError(
                "HTML source must point to collection.json or its directory"
            )
        directory = path if path.is_dir() else path.parent
        return store_html_snapshot(source, data_dir, directory)
    raw = data_dir / "raw"
    raw.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".snapshot-", dir=raw) as temporary:
        staging = Path(temporary)
        shutil.copyfile(project_root / source.local_path, staging / "source.pdf")
        digest, size = _pdf_fingerprint(staging / "source.pdf")
        if source.expected_sha256 is not None and source.expected_sha256 != digest:
            raise ValueError("Source checksum does not match the registry")
        snapshot = SourceSnapshot(
            snapshot_id=digest,
            source_sha256=digest,
            local_path=Path("raw") / digest / "source.pdf",
            size_bytes=size,
            ingested_at=datetime.now(UTC),
            source=source,
        )
        write_json(staging / "manifest.json", snapshot.model_dump(mode="json"))
        destination = raw / digest
        # Do not replace an existing directory, including an incomplete snapshot.
        reused = destination.exists()
        if not reused:
            reused = _publish_snapshot(staging, destination)
        return read_snapshot(destination, digest, size), reused


def import_snapshot(
    source: SourceDefinition,
    data_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> SnapshotImport:
    """Keep the first snapshot and record subsequent provenance in run manifests."""
    with record_run(
        data_dir, source=source.model_dump(mode="json"), snapshot_id=None, reused=None
    ) as run:
        snapshot, reused = _store_snapshot(source, data_dir, project_root)
        run.manifest.update(snapshot_id=snapshot.snapshot_id, reused=reused)
    return SnapshotImport(run_id=run.run_id, snapshot=snapshot, reused=reused)
