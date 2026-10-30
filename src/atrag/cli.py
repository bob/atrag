"""Local configuration, immutable snapshots and located PDF extraction."""

import argparse
import json
from pathlib import Path

from atrag.ingestion.services.extract import extract_snapshot
from atrag.ingestion.services.ocr import TesseractOCR
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.ingestion.services.structure import parse_extraction
from atrag.schemas.settings import Settings
from atrag.settings import PROJECT_ROOT, load_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atrag")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="Validate local configuration")
    checks = config.add_subparsers(dest="action", required=True)
    check = checks.add_parser("check")
    ingest = commands.add_parser("ingest", help="Prepare registered local documents")
    actions = ingest.add_subparsers(dest="action", required=True)
    snapshot = actions.add_parser("snapshot", help="Save an immutable PDF snapshot")
    inspect = actions.add_parser("inspect", help="Extract located pages and tables")
    parse = actions.add_parser(
        "parse", help="Parse legal structure from a completed extraction"
    )
    parse.add_argument(
        "--extracted",
        type=Path,
        required=True,
        help="Completed extraction manifest.json",
    )
    inspect.add_argument(
        "--source",
        type=Path,
        help="Override the registered path (checksum still checked)",
    )
    inspect.add_argument(
        "--tessdata-dir",
        type=Path,
        help="Tesseract data directory containing por.traineddata",
    )
    inspect.add_argument(
        "--no-ocr",
        action="store_true",
        help="Only extract the PDF text layer and flag images",
    )
    inspect.add_argument(
        "--review", type=Path, help="Snapshot-specific manual table review JSON"
    )
    for command in (check, snapshot, inspect, parse):
        command.add_argument(
            "--registry", type=Path, default=PROJECT_ROOT / "data/sources/cirs.json"
        )
    return parser


def _inspect(source, settings: Settings, args) -> dict:
    if args.source is not None:
        source = source.model_copy(update={"local_path": args.source})
    imported = import_snapshot(source, settings.data_dir)
    result = extract_snapshot(
        imported.snapshot,
        settings.data_dir,
        ocr=None if args.no_ocr else TesseractOCR(tessdata_dir=args.tessdata_dir),
        review_path=args.review,
    )
    return {
        "run_id": str(result.run_id),
        "snapshot_id": imported.snapshot.snapshot_id,
        "directory": str(result.directory),
        "page_count": result.page_count,
        "table_count": result.table_count,
        "review_status": result.review_status,
    }


def _settings_summary(settings: Settings) -> dict:
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "data_dir": str(settings.data_dir),
        "database_configured": settings.database_url is not None,
        "openai_configured": settings.openai_api_key is not None,
    }


def _parse(source, settings, args):
    result = parse_extraction(args.extracted, settings.data_dir, source)
    return {
        "run_id": str(result.run_id),
        "directory": str(result.directory),
        "unit_count": result.unit_count,
        "review_status": result.review_status,
    }


def _snapshot(source, settings, args):
    result = import_snapshot(source, settings.data_dir)
    return {
        "run_id": str(result.run_id),
        "reused": result.reused,
        "snapshot": result.snapshot.model_dump(mode="json"),
    }


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        source = load_source_registry(args.registry)
    except ValueError, OSError:
        parser.exit(
            2,
            "Invalid configuration or source registry; check .env and registry fields.\n",
        )
    if args.command == "ingest":
        operation, failure = {
            "parse": (
                _parse,
                "Structure parsing failed; check extraction and run manifest.\n",
            ),
            "inspect": (
                _inspect,
                "PDF extraction failed; check snapshot, review and run manifest.\n",
            ),
            "snapshot": (
                _snapshot,
                "Snapshot import failed; check source and stored artifacts.\n",
            ),
        }[args.action]
        try:
            summary = operation(source, settings, args)
        except ValueError, OSError:
            parser.exit(2, failure)
    else:
        summary = {
            "settings": _settings_summary(settings),
            "source": source.model_dump(mode="json"),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
