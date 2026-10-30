import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from uuid import UUID

import pytest

from atrag.ingestion.services.runs import record_run


def read_manifest(run):
    return json.loads((run.directory / "manifest.json").read_text(encoding="utf-8"))


def test_run_records_progress_then_publishes_success(tmp_path):
    before = datetime.now(UTC)
    with record_run(tmp_path, stage="extract", snapshot_id="a" * 64) as run:
        running = read_manifest(run)
        assert isinstance(run.run_id, UUID)
        assert run.directory == tmp_path / "runs" / str(run.run_id)
        assert running["run_id"] == str(run.run_id)
        assert running["stage"] == "extract"
        assert running["snapshot_id"] == "a" * 64
        assert running["status"] == "running"
        assert running["finished_at"] is None
        run.manifest.update(page_count=2, review_status="needs_review")
    after = datetime.now(UTC)
    completed = read_manifest(run)
    assert completed["status"] == "succeeded"
    assert completed["page_count"] == 2
    assert completed["review_status"] == "needs_review"
    assert completed["stage"] == running["stage"]
    assert completed["snapshot_id"] == running["snapshot_id"]
    started = datetime.fromisoformat(completed["started_at"])
    finished = datetime.fromisoformat(completed["finished_at"])
    assert started.tzinfo == finished.tzinfo == UTC
    assert before <= started <= finished <= after


def test_run_records_failure_and_reraises_the_original_error(tmp_path):
    error = ValueError("private input contents")
    with (
        pytest.raises(ValueError) as caught,
        record_run(tmp_path, stage="parse") as run,
    ):
        run.manifest["input_sha256"] = {"pages.jsonl": "b" * 64}
        raise error
    assert caught.value is error
    failed = read_manifest(run)
    assert failed["status"] == "failed"
    assert failed["error_type"] == "ValueError"
    assert failed["input_sha256"] == {"pages.jsonl": "b" * 64}
    assert datetime.fromisoformat(failed["finished_at"]) >= datetime.fromisoformat(
        failed["started_at"]
    )
    assert "private input contents" not in (run.directory / "manifest.json").read_text()


def test_concurrent_runs_keep_their_manifests_separate(tmp_path):
    def execute(index):
        with record_run(tmp_path, observation=index) as run:
            run.manifest["result"] = index * 2
        return run

    with ThreadPoolExecutor(max_workers=4) as executor:
        runs = list(executor.map(execute, range(4)))
    assert len({run.run_id for run in runs}) == 4
    assert len(list((tmp_path / "runs").iterdir())) == 4
    for index, run in enumerate(runs):
        manifest = read_manifest(run)
        assert manifest["status"] == "succeeded"
        assert manifest["observation"] == index
        assert manifest["result"] == index * 2
