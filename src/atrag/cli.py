"""Local configuration validation and PDF snapshot import."""

import argparse
import json
from pathlib import Path

from atrag.ingestion.snapshots import import_snapshot
from atrag.ingestion.sources import load_source_registry
from atrag.settings import PROJECT_ROOT, Settings, load_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atrag")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="Validate local configuration")
    checks = config.add_subparsers(dest="action", required=True)
    check = checks.add_parser("check")
    ingest = commands.add_parser("ingest", help="Prepare registered local documents")
    actions = ingest.add_subparsers(dest="action", required=True)
    snapshot = actions.add_parser("snapshot", help="Save an immutable PDF snapshot")
    for command in (check, snapshot):
        command.add_argument(
            "--registry", type=Path, default=PROJECT_ROOT / "data/sources/cirs.json"
        )
    return parser


def _settings_summary(settings: Settings) -> dict:
    return {
        "embedding_model": settings.embedding_model,
        "embedding_dimensions": settings.embedding_dimensions,
        "data_dir": str(settings.data_dir),
        "database_configured": settings.database_url is not None,
        "openai_configured": settings.openai_api_key is not None,
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
        try:
            result = import_snapshot(source, settings.data_dir)
        except ValueError, OSError:
            parser.exit(
                2, "Snapshot import failed; check source and stored artifacts.\n"
            )
        summary = {
            "run_id": str(result.run_id),
            "reused": result.reused,
            "snapshot": result.snapshot.model_dump(mode="json"),
        }
    else:
        summary = {
            "settings": _settings_summary(settings),
            "source": source.model_dump(mode="json"),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
