"""Inspect saved HTML acquisition; FAQ answer extraction is a separate stage."""

from pathlib import Path

from atrag.ingestion.faq_navigation import inspect_html
from atrag.ingestion.services.artifacts import write_json
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import read_snapshot
from atrag.schemas.ingestion.snapshots import SourceSnapshot


def inspect_collection(snapshot: SourceSnapshot, data_dir: Path) -> dict:
    directory = data_dir / snapshot.local_path.parent
    stored = read_snapshot(directory, snapshot.source_sha256, snapshot.size_bytes)
    if stored != snapshot or snapshot.collection is None:
        raise ValueError("HTML snapshot manifest changed or is missing a collection")
    with record_run(
        data_dir, stage="inspect_html", snapshot_id=snapshot.snapshot_id
    ) as run:
        pages = []
        question_count = 0
        for page in snapshot.collection.pages:
            inspected = inspect_html(
                (directory / page.local_path).read_bytes(), str(page.final_url)
            )
            question_count += inspected.question_count
            pages.append(
                {
                    "url": str(page.url),
                    "template": inspected.template,
                    "question_count": inspected.question_count,
                    "navigation_urls": inspected.navigation_urls,
                }
            )
        report = {
            "source_format": "html_collection",
            "page_count": len(pages),
            "question_count": question_count,
            "review_status": "needs_review",
            "pages": pages,
        }
        write_json(run.directory / "report.json", report)
        run.manifest.update(**report)
    return {
        "run_id": str(run.run_id),
        "snapshot_id": snapshot.snapshot_id,
        "directory": str(run.directory),
        "page_count": report["page_count"],
        "question_count": report["question_count"],
        "review_status": report["review_status"],
    }
