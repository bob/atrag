"""Extract verified saved HTML into isolated, deterministic FAQ artifacts."""

import hashlib
from collections import Counter
from importlib.metadata import version
from pathlib import Path

from bs4 import BeautifulSoup

from atrag.ingestion.faq_extract import PARSER_VERSION, parse_faq_page
from atrag.ingestion.faq_navigation import inspect_html
from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import read_snapshot
from atrag.schemas.ingestion.snapshots import SourceSnapshot


def _parse_page(raw: bytes, url: str) -> dict:
    inspection = inspect_html(raw, url)
    if inspection.template == "catalog-navigation":
        return {
            "url": url,
            "template": inspection.template,
            "detected_count": 0,
            "extracted_count": 0,
            "issues": [],
            "rejected_ordinals": [],
        }
    soup = BeautifulSoup(raw, "html.parser")
    title = soup.select_one("h1.tittle-tab-page")
    path = tuple(part.strip() for part in title.get_text().split(">")) if title else ()
    page = parse_faq_page(raw, url, path)
    return {
        "url": url,
        "template": inspection.template,
        "detected_count": page.detected_count,
        "extracted_count": len(page.faqs),
        "answer_count": page.answer_count,
        "detected_table_count": page.detected_table_count,
        "issues": [i.model_dump() for i in page.issues],
        "rejected_ordinals": list(page.rejected_ordinals),
        "parsed": page,
    }


def _report(snapshot, pages, faqs, tables) -> dict:
    counts = Counter(f.entry.entry_id for f in faqs)
    conflicts = sorted(entry_id for entry_id, count in counts.items() if count > 1)
    return {
        "schema_version": 1,
        "source_format": "html_collection",
        "snapshot_id": snapshot.snapshot_id,
        "parser_version": PARSER_VERSION,
        "page_count": len(pages),
        "question_count": sum(p["detected_count"] for p in pages),
        "extracted_count": len(faqs),
        "rejected_count": sum(len(p["rejected_ordinals"]) for p in pages),
        "table_count": len(tables),
        "detected_table_count": sum(p.get("detected_table_count", 0) for p in pages),
        "answer_count": sum(p.get("answer_count", 0) for p in pages),
        "link_count": sum(len(f.entry.links) for f in faqs),
        "reference_count": sum(len(f.references) for f in faqs),
        "conflicting_entry_ids": conflicts,
        "issues_by_kind": dict(Counter(i["kind"] for p in pages for i in p["issues"])),
        "review_status": "needs_review"
        if conflicts or any(p["issues"] for p in pages)
        else "verified",
        "pages": pages,
    }


def extract_collection(snapshot: SourceSnapshot, data_dir: Path) -> dict:
    with record_run(
        data_dir,
        stage="extract_faq",
        snapshot_id=snapshot.snapshot_id,
        parser_version=PARSER_VERSION,
        dependencies={"beautifulsoup4": version("beautifulsoup4")},
        dom_parser="html.parser",
    ) as run:
        directory = data_dir / snapshot.local_path.parent
        stored = read_snapshot(directory, snapshot.snapshot_id, snapshot.size_bytes)
        if stored != snapshot or snapshot.collection is None:
            raise ValueError(
                "HTML snapshot manifest changed or is missing a collection"
            )
        pages, faqs, tables = [], [], []
        for page in sorted(snapshot.collection.pages, key=lambda p: str(p.url)):
            parsed = _parse_page(
                (directory / page.local_path).read_bytes(), str(page.final_url)
            )
            parsed.update(url=str(page.url), final_url=str(page.final_url))
            result = parsed.pop("parsed", None)
            if result is not None:
                faqs.extend(result.faqs)
                tables.extend(result.tables)
            pages.append(parsed)
        report = _report(snapshot, pages, faqs, tables)
        write_jsonl(run.directory / "faqs.jsonl", faqs)
        write_json(
            run.directory / "tables.json",
            {"tables": [t.model_dump(mode="json") for t in tables]},
        )
        write_json(run.directory / "report.json", report)
        run.manifest.update(**report)
        run.manifest["output_sha256"] = {
            name: hashlib.sha256((run.directory / name).read_bytes()).hexdigest()
            for name in ("faqs.jsonl", "tables.json", "report.json")
        }
    return {
        "run_id": str(run.run_id),
        "directory": str(run.directory),
        **{k: v for k, v in report.items() if k != "pages"},
    }
