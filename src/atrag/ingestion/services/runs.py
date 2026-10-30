"""Own one isolated ingestion attempt and record its terminal state."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from atrag.ingestion.services.artifacts import write_json


@dataclass(frozen=True)
class IngestionRun:
    run_id: UUID
    directory: Path
    manifest: dict[str, Any]


def _finish_run(run: IngestionRun, status: str, **metadata: Any) -> None:
    run.manifest.update(
        **metadata, status=status, finished_at=datetime.now(UTC).isoformat()
    )
    write_json(run.directory / "manifest.json", run.manifest)


@contextmanager
def record_run(data_dir: Path, **metadata: Any) -> Iterator[IngestionRun]:
    run_id = uuid4()
    directory = data_dir / "runs" / str(run_id)
    directory.mkdir(parents=True)
    manifest = {
        **metadata,
        "run_id": str(run_id),
        "started_at": datetime.now(UTC).isoformat(),
        "finished_at": None,
        "status": "running",
    }
    run = IngestionRun(run_id, directory, manifest)
    write_json(directory / "manifest.json", manifest)
    try:
        yield run
    except Exception as error:
        _finish_run(run, "failed", error_type=type(error).__name__)
        raise
    else:
        _finish_run(run, "succeeded")
