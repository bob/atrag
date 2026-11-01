"""Local configuration, immutable snapshots and located PDF extraction."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from typing import get_args

from sqlalchemy.exc import SQLAlchemyError

from atrag.db import create_db_engine
from atrag.ingestion.client.openai import create_openai_client
from atrag.ingestion.services.chunks import chunk_parsed
from atrag.ingestion.services.embedding_cache import PostgresEmbeddingCache
from atrag.ingestion.services.embeddings import embed_chunked
from atrag.ingestion.services.extract import extract_snapshot
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.ocr import TesseractOCR
from atrag.ingestion.services.packages import build_package, load_package
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.ingestion.services.structure import parse_extraction
from atrag.migrations import migrate_database
from atrag.retrieval.evaluate import evaluate_search
from atrag.retrieval.search import search_tax_documents
from atrag.schemas.ingestion.fragments import ChunkParameters
from atrag.schemas.retrieval import SearchFilters
from atrag.schemas.settings import Settings, SupportedLanguage
from atrag.settings import PROJECT_ROOT, load_settings


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atrag")
    commands = parser.add_subparsers(dest="command", required=True)
    _search_parser(commands)
    evaluation = commands.add_parser(
        "evaluate", help="Evaluate retrieval against reviewed expectations"
    )
    evaluation.add_argument("--package", type=Path, required=True)
    evaluation.add_argument("--benchmark", type=Path, required=True)
    evaluation.add_argument("--context-tokens", type=int, default=6000)
    database = commands.add_parser("db", help="Manage the PostgreSQL schema")
    database.add_subparsers(dest="action", required=True).add_parser("migrate")
    config = commands.add_parser("config", help="Validate local configuration")
    checks = config.add_subparsers(dest="action", required=True)
    check = checks.add_parser("check")
    ingest = commands.add_parser("ingest", help="Prepare registered local documents")
    actions = ingest.add_subparsers(dest="action", required=True)
    _corpus_parsers(actions)
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


def _corpus_parsers(actions):
    package = actions.add_parser("package", help="Export a checked portable document")
    for name in ("parsed", "chunked", "embedded", "output"):
        package.add_argument(f"--{name}", type=Path, required=True)
    package.add_argument("--review", type=Path, action="append", default=[])
    index = actions.add_parser("index", help="Atomically index a prepared package")
    index.add_argument(
        "--package", type=Path, required=True, help="Package manifest.json"
    )


def _search_parser(commands):
    search = commands.add_parser("search", help="Search one indexed snapshot")
    search.add_argument("query")
    search.add_argument("--snapshot", required=True)
    search.add_argument("--language", choices=get_args(SupportedLanguage))
    search.add_argument("--limit", type=int, default=5)
    search.add_argument("--context-tokens", type=int, default=6000)
    search.add_argument("--code")
    search.add_argument("--article")
    search.add_argument("--scope", choices=("code", "approving_decree", "preamble"))


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


def _embed(source, settings, args):
    with ExitStack() as stack:
        engine = create_db_engine(settings)
        stack.callback(engine.dispose)
        client = create_openai_client(settings)
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


def _package(source, settings, args):
    path = build_package(
        args.parsed,
        args.chunked,
        args.embedded,
        settings.data_dir,
        args.output,
        reviews=args.review,
    )
    document = load_package(path)
    return {
        "manifest": str(path),
        "snapshot_id": document.snapshot.snapshot_id,
        "fragment_count": len(document.fragments),
    }


def _index(source, settings, args):
    engine = create_db_engine(settings)
    try:
        result = index_package(args.package, engine, settings.data_dir)
    finally:
        engine.dispose()
    return {
        "snapshot_id": result.snapshot_id,
        "fragment_count": result.fragment_count,
        "directory": str(result.directory),
    }


def _search_command(source, settings, args):
    with ExitStack() as stack:
        engine = create_db_engine(settings)
        stack.callback(engine.dispose)
        client = create_openai_client(settings)
        if client is not None:
            stack.enter_context(client)
        result = search_tax_documents(
            args.query,
            args.limit,
            args.language,
            snapshot_id=args.snapshot,
            engine=engine,
            client=client,
            context_tokens=args.context_tokens,
            filters=SearchFilters(
                code=args.code, article=args.article, document_scope=args.scope
            ),
        )
    return result.model_dump(mode="json")


def _evaluate_command(source, settings, args):
    with ExitStack() as stack:
        engine = create_db_engine(settings)
        stack.callback(engine.dispose)
        client = create_openai_client(settings)
        if client is not None:
            stack.enter_context(client)
        directory = evaluate_search(
            args.package,
            args.benchmark,
            engine,
            settings.data_dir,
            client=client,
            context_tokens=args.context_tokens,
        )
    return {"directory": str(directory), "report": str(directory / "report.json")}


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
        source = _command_source(args)
    except (ValueError, OSError):
        parser.exit(
            2,
            "Invalid configuration or source registry; check .env and registry fields.\n",
        )
    if args.command in ("search", "evaluate"):
        operation, failure = {
            "search": (
                _search_command,
                "Search failed; check snapshot, query, language and database.\n",
            ),
            "evaluate": (
                _evaluate_command,
                "Evaluation failed; check package, benchmark, database and run report.\n",
            ),
        }[args.command]
        summary = _execute(
            parser,
            operation,
            failure,
            source,
            settings,
            args,
        )
    elif args.command == "ingest":
        operation, failure = {
            "package": (
                _package,
                "Package preparation failed; check prepared artifacts and output path.\n",
            ),
            "index": (
                _index,
                "Indexing failed; check package, database migrations and run report.\n",
            ),
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
        summary = _execute(parser, operation, failure, source, settings, args)
    else:
        summary = {
            "settings": _settings_summary(settings),
            "source": source.model_dump(mode="json"),
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _command_source(args):
    if args.command in ("search", "evaluate"):
        return None
    if args.action in ("index", "package"):
        return None
    return load_source_registry(args.registry)


def _execute(parser, operation, failure, source, settings, args):
    try:
        return operation(source, settings, args)
    except (ValueError, OSError, SQLAlchemyError):
        parser.exit(2, failure)
