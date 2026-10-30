import errno
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from threading import Barrier
from uuid import UUID

import pytest

from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.settings import PROJECT_ROOT

PDF_BYTES = b"%PDF-1.7\nlocal snapshot fixture\n%%EOF\n"


@pytest.fixture
def source(tmp_path):
    original = tmp_path / "docs/source.pdf"
    original.parent.mkdir()
    original.write_bytes(PDF_BYTES)
    registered = load_source_registry(PROJECT_ROOT / "data/sources/cirs.json")
    return registered.model_copy(
        update={
            "local_path": Path("docs/source.pdf"),
            "expected_sha256": hashlib.sha256(PDF_BYTES).hexdigest(),
        }
    )


def read_run(data_dir, run_id):
    return json.loads((data_dir / "runs" / str(run_id) / "manifest.json").read_text())


def test_import_preserves_bytes_provenance_and_actual_dates(source, tmp_path):
    data_dir = tmp_path / "data"
    original = tmp_path / source.local_path
    before_stat = original.stat()
    before = datetime.now(UTC)
    result = import_snapshot(source, data_dir, project_root=tmp_path)
    after = datetime.now(UTC)

    digest = hashlib.sha256(PDF_BYTES).hexdigest()
    stored = data_dir / "raw" / digest
    manifest = json.loads((stored / "manifest.json").read_text())
    assert "Código" in (stored / "manifest.json").read_text(encoding="utf-8")
    assert result.snapshot.snapshot_id == digest
    assert result.snapshot.source_sha256 == digest
    assert result.reused is False
    assert isinstance(result.run_id, UUID)
    assert (stored / "source.pdf").read_bytes() == original.read_bytes() == PDF_BYTES
    assert original.stat().st_mtime_ns == before_stat.st_mtime_ns
    assert manifest["schema_version"] == 1
    assert manifest["local_path"] == f"raw/{digest}/source.pdf"
    assert manifest["size_bytes"] == len(PDF_BYTES)
    assert before <= result.snapshot.ingested_at <= after
    assert result.snapshot.ingested_at.tzinfo == UTC
    assert manifest["source"] == source.model_dump(mode="json")
    assert manifest["source"]["downloaded_at"] is None
    assert manifest["source"]["revision_publication_date"] is None
    assert manifest["source"]["source_url_status"] == "candidate"
    assert "run_id" not in manifest
    run = read_run(data_dir, result.run_id)
    assert run["status"] == "succeeded"
    assert run["run_id"] == str(result.run_id)
    assert run["snapshot_id"] == digest
    assert run["reused"] is False
    assert run["source"] == source.model_dump(mode="json")
    assert before <= datetime.fromisoformat(run["started_at"]) <= after
    assert before <= datetime.fromisoformat(run["finished_at"]) <= after


def test_reimport_reuses_snapshot_without_rewriting_manifest(source, tmp_path):
    data_dir = tmp_path / "data"
    first = import_snapshot(source, data_dir, project_root=tmp_path)
    manifest = data_dir / "raw" / first.snapshot.snapshot_id / "manifest.json"
    before = manifest.read_bytes(), manifest.stat().st_mtime_ns
    second = import_snapshot(source, data_dir, project_root=tmp_path)

    assert first.run_id != second.run_id
    assert first.snapshot == second.snapshot
    assert second.reused is True
    assert (manifest.read_bytes(), manifest.stat().st_mtime_ns) == before
    assert len(list((data_dir / "raw").iterdir())) == 1
    assert len(list((data_dir / "runs").iterdir())) == 2
    assert read_run(data_dir, second.run_id)["reused"] is True


def test_same_bytes_keep_first_origin_and_record_each_observation(source, tmp_path):
    data_dir = tmp_path / "data"
    first = import_snapshot(source, data_dir, project_root=tmp_path)
    other_path = tmp_path / "other.pdf"
    other_path.write_bytes(PDF_BYTES)
    other = source.model_copy(
        update={"source_id": "another-source", "local_path": other_path}
    )
    second = import_snapshot(other, data_dir, project_root=tmp_path)
    assert second.snapshot == first.snapshot
    assert read_run(data_dir, second.run_id)["source"]["source_id"] == "another-source"
    assert second.reused is True


def test_new_bytes_produce_a_new_snapshot(source, tmp_path):
    data_dir = tmp_path / "data"
    source = source.model_copy(update={"expected_sha256": None})
    first = import_snapshot(source, data_dir, project_root=tmp_path)
    (tmp_path / source.local_path).write_bytes(PDF_BYTES + b"new revision\n")
    second = import_snapshot(source, data_dir, project_root=tmp_path)
    assert first.snapshot.snapshot_id != second.snapshot.snapshot_id
    assert second.reused is False
    assert (data_dir / first.snapshot.local_path).read_bytes() == PDF_BYTES
    assert len(list((data_dir / "raw").iterdir())) == 2


@pytest.mark.parametrize("problem", ["missing", "not-pdf", "checksum"])
def test_invalid_input_records_failed_attempt_without_snapshot(
    source, tmp_path, problem
):
    original = tmp_path / source.local_path
    if problem == "missing":
        original.unlink()
    elif problem == "not-pdf":
        original.write_bytes(b"<html>not a PDF</html>")
    else:
        original.write_bytes(PDF_BYTES + b"changed")
    data_dir = tmp_path / "data"
    with pytest.raises((ValueError, OSError)):
        import_snapshot(source, data_dir, project_root=tmp_path)
    assert list((data_dir / "raw").iterdir()) == []
    [run_path] = (data_dir / "runs").glob("*/manifest.json")
    run = json.loads(run_path.read_text())
    assert run["status"] == "failed"
    assert run["snapshot_id"] is None
    assert run["finished_at"] is not None


@pytest.mark.parametrize(
    "problem",
    ["bytes", "missing-pdf", "missing-manifest", "json", "identity", "size", "path"],
)
def test_corrupted_snapshot_is_rejected_without_repair(source, tmp_path, problem):
    data_dir = tmp_path / "data"
    result = import_snapshot(source, data_dir, project_root=tmp_path)
    directory = data_dir / "raw" / result.snapshot.snapshot_id
    manifest_path = directory / "manifest.json"
    if problem == "bytes":
        (directory / "source.pdf").write_bytes(PDF_BYTES.replace(b"local", b"other"))
    elif problem == "missing-pdf":
        (directory / "source.pdf").unlink()
    elif problem == "missing-manifest":
        manifest_path.unlink()
    elif problem == "json":
        manifest_path.write_text("broken json")
    else:
        manifest = json.loads(manifest_path.read_text())
        field, value = {
            "identity": ("snapshot_id", "0" * 64),
            "size": ("size_bytes", len(PDF_BYTES) + 1),
            "path": ("local_path", "elsewhere.pdf"),
        }[problem]
        manifest[field] = value
        manifest_path.write_text(json.dumps(manifest))
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    with pytest.raises((ValueError, OSError)):
        import_snapshot(source, data_dir, project_root=tmp_path)
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    runs = [
        json.loads(p.read_text()) for p in (data_dir / "runs").glob("*/manifest.json")
    ]
    assert sorted(run["status"] for run in runs) == ["failed", "succeeded"]


def test_concurrent_imports_publish_one_complete_snapshot(
    source, tmp_path, monkeypatch
):
    data_dir = tmp_path / "data"
    barrier = Barrier(4)
    original_rename = Path.rename

    def simultaneous_publish(path, destination):
        if path.name.startswith(".snapshot-"):
            barrier.wait(timeout=5)
        return original_rename(path, destination)

    monkeypatch.setattr(Path, "rename", simultaneous_publish)
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: import_snapshot(source, data_dir, project_root=tmp_path),
                range(4),
            )
        )
    assert len({result.snapshot.snapshot_id for result in results}) == 1
    assert len({result.run_id for result in results}) == 4
    assert sum(not result.reused for result in results) == 1
    assert all(result.snapshot == results[0].snapshot for result in results)
    assert len(list((data_dir / "raw").iterdir())) == 1


def test_publication_failure_can_be_retried(source, tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    original_rename = Path.rename

    def failed_publish(path, destination):
        if path.name.startswith(".snapshot-"):
            raise OSError(errno.EACCES, "simulated write failure")
        return original_rename(path, destination)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", failed_publish)
        with pytest.raises(OSError):
            import_snapshot(source, data_dir, project_root=tmp_path)
    assert list((data_dir / "raw").iterdir()) == []
    retried = import_snapshot(source, data_dir, project_root=tmp_path)
    assert retried.reused is False
    runs = [
        json.loads(p.read_text()) for p in (data_dir / "runs").glob("*/manifest.json")
    ]
    assert sorted(run["status"] for run in runs) == ["failed", "succeeded"]


def test_incomplete_existing_directory_is_not_overwritten(source, tmp_path):
    data_dir = tmp_path / "data"
    directory = data_dir / "raw" / source.expected_sha256
    directory.mkdir(parents=True)
    with pytest.raises(OSError):
        import_snapshot(source, data_dir, project_root=tmp_path)
    assert list(directory.iterdir()) == []
