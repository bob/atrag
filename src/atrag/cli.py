"""Local configuration validation without database or paid API calls."""

import argparse
import json
from pathlib import Path

from atrag.ingestion.sources import load_source_registry
from atrag.settings import PROJECT_ROOT, load_settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="atrag")
    commands = parser.add_subparsers(dest="command", required=True)
    config = commands.add_parser("config", help="Validate local configuration")
    checks = config.add_subparsers(dest="action", required=True)
    check = checks.add_parser("check")
    check.add_argument(
        "--registry", type=Path, default=PROJECT_ROOT / "data/sources/cirs.json"
    )
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        source = load_source_registry(args.registry)
    except ValueError, OSError:
        parser.exit(
            2,
            "Invalid configuration or source registry; check .env and registry fields.\n",
        )
    print(
        json.dumps(
            {
                "settings": {
                    "embedding_model": settings.embedding_model,
                    "embedding_dimensions": settings.embedding_dimensions,
                    "data_dir": str(settings.data_dir),
                    "database_configured": settings.database_url is not None,
                    "openai_configured": settings.openai_api_key is not None,
                },
                "source": source.model_dump(mode="json"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
