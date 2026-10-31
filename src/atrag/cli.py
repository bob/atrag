"""Local configuration, immutable snapshots and located PDF extraction."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from openai import OpenAI
from sqlalchemy.exc import SQLAlchemyError

from atrag.db import create_db_engine
from atrag.ingestion.services.chunks import chunk_parsed
from atrag.ingestion.services.embedding_cache import PostgresEmbeddingCache
from atrag.ingestion.services.embeddings import embed_chunked
from atrag.ingestion.services.extract import extract_snapshot
from atrag.ingestion.services.ocr import TesseractOCR
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.ingestion.services.structure import parse_extraction
from atrag.migrations import migrate_database
from atrag.schemas.ingestion.fragments import ChunkParameters
from atrag.schemas.settings import Settings
from atrag.settings import PROJECT_ROOT, load_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atrag")
    commands = parser.add_subparsers(dest="command", required=True)
    database = commands.add_parser("db", help="Manage the PostgreSQL schema")
    database.add_subparsers(dest="action", required=True).add_parser("migrate")
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
    chunk = actions.add_parser(
        "chunk", help="Build search fragments from a completed parse"
    )
    chunk.add_argument(
        "--parsed", type=Path, required=True, help="Completed parse manifest.json"
    )
    embed = actions.add_parser("embed", help="Cache embeddings for completed chunks")
    embed.add_argument(
        "--chunked", type=Path, required=True, help="Completed chunk manifest.json"
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
    for command in (check, snapshot, inspect, parse, chunk, embed):
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


def _chunk(source, settings, args):
    result = chunk_parsed(
        args.parsed,
        settings.data_dir,
        source,
        ChunkParameters(
            target_tokens=settings.chunk_target_tokens,
            max_tokens=settings.chunk_max_tokens,
        ),
    )
    return {
        "run_id": str(result.run_id),
        "directory": str(result.directory),
        "fragment_count": result.fragment_count,
        "review_status": result.review_status,
    }


def _embedding_client(settings):
    if settings.openai_api_key is None:
        return None
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url="https://api.openai.com/v1",
        max_retries=0,
        timeout=30,
    )


def _embed(source, settings, args):
    with ExitStack() as stack:
        engine = create_db_engine(settings)
        stack.callback(engine.dispose)
        client = _embedding_client(settings)
        if client is not None:
            stack.enter_context(client)
        result = embed_chunked(
            args.chunked,
            settings,
            source,
            cache=PostgresEmbeddingCache(engine),
            client=client,
        )
    return {
        "run_id": str(result.run_id),
        "directory": str(result.directory),
        "fragment_count": result.fragment_count,
        "model": settings.embedding_model,
        "dimensions": settings.embedding_dimensions,
    }


def _migrate(parser, settings):
    try:
        engine = create_db_engine(settings)
        try:
            migrate_database(engine)
        finally:
            engine.dispose()
    except (ValueError, SQLAlchemyError):
        parser.exit(
            2,
            "Database migration failed; check DATABASE_URL and database permissions.\n",
        )
    print(json.dumps({"status": "succeeded", "action": "migrate"}))


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        if args.command == "db":
            _migrate(parser, settings)
            return
        source = load_source_registry(args.registry)
    except (ValueError, OSError):
        parser.exit(
            2,
            "Invalid configuration or source registry; check .env and registry fields.\n",
        )
    if args.command == "ingest":
        operation, failure = {
            "embed": (
                _embed,
                "Embedding preparation failed; check configuration, chunk artifacts, database migrations and run report.\n",
            ),
            "chunk": (
                _chunk,
                "Fragment preparation failed; check parsed artifacts and run manifest.\n",
            ),
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
        except (ValueError, OSError, SQLAlchemyError):
            parser.exit(2, failure)
    else:
        summary = {
            "settings": _settings_summary(settings),
            "source": source.model_dump(mode="json"),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
