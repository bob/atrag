"""Run the saved HTML CLI pipeline with synthetic vectors in a local test DB.

Usage: .venv/bin/python tools/faq_ingestion_smoke.py --snapshot raw/ID/manifest.json
       --data-dir data/faq-synthetic --database-url postgresql://localhost/atrag_test_...
"""

import argparse
import hashlib
import io
import json
import os
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from psycopg.conninfo import conninfo_to_dict
from sqlalchemy import text

from atrag import cli
from atrag.db import create_db_engine
from atrag.ingestion.part_validation import reassemble_unit
from atrag.ingestion.services.artifacts import write_json
from atrag.ingestion.services.packages import load_package
from atrag.ingestion.source_files import source_file_names
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.settings import Settings


class SyntheticClient:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    @property
    def embeddings(self):
        return self

    def create(self, *, model, dimensions, input, encoding_format):
        # Stable synthetic vectors are for pipeline checks, never semantic QA.
        vectors = []
        for index, value in enumerate(input):
            digest = hashlib.sha256(value.encode()).digest()
            vector = [0.0] * dimensions
            vector[int.from_bytes(digest[:4]) % dimensions] = 1.0
            vectors.append({"index": index, "embedding": vector})
        response = {
            "model": model,
            "data": vectors,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }
        return SimpleNamespace(model_dump=lambda: response)


def command(*args):
    output = io.StringIO()
    with redirect_stdout(output):
        cli.main(list(map(str, args)))
    return json.loads(output.getvalue())


def run(snapshot_path, data_dir, database_url):
    database_name = conninfo_to_dict(database_url).get("dbname")
    if not isinstance(database_name, str) or not database_name.startswith(
        "atrag_test_"
    ):
        raise ValueError("Synthetic smoke requires an atrag_test_* database")
    snapshot = SourceSnapshot.model_validate_json(snapshot_path.read_bytes())
    source = snapshot.source.model_copy(
        update={"local_path": snapshot_path.parent.resolve()}
    )
    data_dir.mkdir(parents=True, exist_ok=True)
    registry = data_dir / "registry.json"
    write_json(registry, source.model_dump(mode="json"))
    common = ("--registry", registry)
    settings = {
        "DATABASE_URL": database_url,
        "ATRAG_DATA_DIR": str(data_dir.resolve()),
        "OPENAI_EMBEDDING_MODEL": "text-embedding-3-large",
        "OPENAI_EMBEDDING_DIMENSIONS": "1536",
    }
    with (
        patch.dict(os.environ, settings),
        patch.object(cli, "create_openai_client", lambda _: SyntheticClient()),
    ):
        command("db", "migrate")
        extracted = command("ingest", "extract", *common)
        parsed = command(
            "ingest",
            "parse",
            "--extracted",
            Path(extracted["directory"]) / "manifest.json",
            *common,
        )
        parsed_path = Path(parsed["directory"]) / "manifest.json"
        chunked = command("ingest", "chunk", "--parsed", parsed_path, *common)
        chunk_path = Path(chunked["directory"]) / "manifest.json"
        embedded = command("ingest", "embed", "--chunked", chunk_path, *common)
        repeat = command("ingest", "embed", "--chunked", chunk_path, *common)
        cache_report = json.loads(
            (Path(repeat["directory"]) / "report.json").read_bytes()
        )
        assert cache_report["requests"] == 0
        result = command(
            "ingest",
            "package",
            "--parsed",
            parsed_path,
            "--chunked",
            chunk_path,
            "--embedded",
            Path(embedded["directory"]) / "manifest.json",
            "--output",
            data_dir / "package",
        )
        document = load_package(Path(result["manifest"]))
        parts = {}
        for part in document.fragments:
            parts.setdefault(part.legal_unit_id, []).append(part)
        for unit in document.units:
            if unit.faq is not None:
                assert (
                    reassemble_unit(unit, parts[unit.legal_unit_id]) == unit.faq.answer
                )
        for _ in range(2):
            command("ingest", "index", "--package", result["manifest"])
    engine = create_db_engine(Settings(database_url=database_url))
    try:
        with engine.connect() as connection:
            count = connection.scalar(
                text("SELECT count(*) FROM search_fragments WHERE snapshot_id = :id"),
                {"id": document.snapshot.snapshot_id},
            )
            assert count == len(document.fragments)
    finally:
        engine.dispose()
    summary = {
        "synthetic": True,
        "package": result["manifest"],
        "snapshot_id": document.snapshot.snapshot_id,
        "raw_snapshot_id": snapshot.source_sha256,
        "faq_count": len(parts),
        "fragment_count": count,
        "table_count": len(document.tables),
        "source_file_count": len(source_file_names(document.snapshot)),
        "cache_replay": cache_report,
        "total_tokens": sum(f.token_count for f in document.fragments),
    }
    write_json(data_dir / "verification.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--database-url", required=True)
    args = parser.parse_args()
    result = run(args.snapshot, args.data_dir, args.database_url)
    print(
        json.dumps({k: v for k, v in result.items() if k != "cache_replay"}, indent=2)
    )
