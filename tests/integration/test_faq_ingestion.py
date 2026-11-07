"""Offline CLI round trip with real PostgreSQL and a substituted embedding API."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import event, text

from atrag.cli import main
from atrag.ingestion.services.packages import load_package
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def schema(empty_corpus_schema, engine):
    migrate_database(engine)


def run_cli(capsys, *args):
    main(list(args))
    return json.loads(capsys.readouterr().out)


@pytest.fixture
def faq_package(faq_snapshot, tmp_path, database_dsn, monkeypatch, capsys):
    from atrag import cli

    calls = []

    def create(**request):
        calls.append(request)
        response = {
            "model": request["model"],
            "data": [
                {"index": i, "embedding": [1.0] + [0.0] * 1535}
                for i, _ in enumerate(request["input"])
            ],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        }
        return SimpleNamespace(model_dump=lambda: response)

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))

    class ClientContext:
        embeddings = client.embeddings

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(cli, "create_openai_client", lambda settings: ClientContext())
    monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DATABASE_URL", database_dsn)
    monkeypatch.setenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    registry = tmp_path / "registry.json"
    registry.write_text(faq_snapshot.source.model_dump_json())
    common = ("--registry", str(registry))
    extracted = run_cli(capsys, "ingest", "extract", *common)
    parsed = run_cli(
        capsys,
        "ingest",
        "parse",
        "--extracted",
        str(Path(extracted["directory"]) / "manifest.json"),
        *common,
    )
    chunked = run_cli(
        capsys,
        "ingest",
        "chunk",
        "--parsed",
        str(Path(parsed["directory"]) / "manifest.json"),
        *common,
    )
    chunk = str(Path(chunked["directory"]) / "manifest.json")
    first = run_cli(capsys, "ingest", "embed", "--chunked", chunk, *common)
    second = run_cli(capsys, "ingest", "embed", "--chunked", chunk, *common)
    assert len(calls) == 1
    assert (
        calls[0]["model"] == "text-embedding-3-large" and calls[0]["dimensions"] == 1536
    )
    assert (
        json.loads((Path(second["directory"]) / "report.json").read_bytes())[
            "cache_hits"
        ]
        == 1
    )
    package = run_cli(
        capsys,
        "ingest",
        "package",
        "--parsed",
        str(Path(parsed["directory"]) / "manifest.json"),
        "--chunked",
        chunk,
        "--embedded",
        str(Path(first["directory"]) / "manifest.json"),
        "--output",
        str(tmp_path / "package"),
    )
    return Path(package["manifest"])


@pytest.mark.parametrize(
    "faq_snapshot",
    [
        {},
        {"answer": "<table><tr><th>Tax</th></tr><tr><td><p>IVA</p></td></tr></table>"},
    ],
    indirect=True,
)
def test_cli_html_package_roundtrip_is_idempotent(
    faq_package, engine, tmp_path, capsys, monkeypatch
):
    document = load_package(faq_package)
    assert document.manifest.schema_version == 3
    assert "source.pdf" not in document.files
    assert document.snapshot.source.source_id == "pt-at-faq"
    server = tmp_path / "server"
    monkeypatch.setenv("ATRAG_DATA_DIR", str(server))
    from atrag.ingestion import faq_extract

    monkeypatch.setattr(
        faq_extract, "parse_faq_page", lambda *args: pytest.fail("Server parsed HTML")
    )
    for _ in range(2):
        result = run_cli(capsys, "ingest", "index", "--package", str(faq_package))
        assert result["snapshot_id"] == document.snapshot.snapshot_id
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM source_snapshots")) == 1

        assert connection.scalar(text("SELECT count(*) FROM search_fragments")) == 1
        assert connection.scalar(text("SELECT count(*) FROM embeddings")) == 1
        assert connection.scalar(text("SELECT count(*) FROM corpus_members")) == 0
    for name in document.manifest.source_files:
        assert (
            server / document.snapshot.local_path.parent / name
        ).read_bytes() == document.files[name]


@pytest.mark.parametrize("damage", ["html", "vectors"])
def test_bad_faq_package_keeps_previous_index(faq_package, engine, tmp_path, damage):
    from fixtures.prepared import digest

    from atrag.ingestion.services.index import index_package

    index_package(faq_package, engine, tmp_path / "server")
    manifest = json.loads(faq_package.read_bytes())
    name = (
        next(n for n in manifest["source_files"] if n.endswith(".html"))
        if damage == "html"
        else "embeddings.jsonl"
    )
    (faq_package.parent / name).write_bytes(b"broken" if damage == "html" else b"")
    if damage == "vectors":
        manifest["files"][name] = digest(b"")
        stage = faq_package.parent / "embed/manifest.json"
        content = json.loads(stage.read_bytes())
        content["output_sha256"][name] = digest(b"")
        stage.write_text(json.dumps(content))
        manifest["files"]["embed/manifest.json"] = digest(stage.read_bytes())
        faq_package.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="checksum|complete fragment set"):
        index_package(faq_package, engine, tmp_path / "server")
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM search_fragments")) == 1


def test_faq_index_failure_rolls_back(faq_package, engine, tmp_path, prepared_document):
    from atrag.ingestion.services.index import index_package
    from atrag.ingestion.services.packages import build_package

    data, parsed, chunked, embedded = prepared_document
    cirs = build_package(parsed, chunked, embedded, data, tmp_path / "cirs")
    index_package(cirs, engine, tmp_path / "server")

    def reject(connection, cursor, statement, parameters, context, many):
        if statement.startswith("INSERT INTO search_fragments"):
            raise RuntimeError("injected failure")

    event.listen(engine, "before_cursor_execute", reject)
    try:
        with pytest.raises(RuntimeError, match="injected"):
            index_package(faq_package, engine, tmp_path / "server")
    finally:
        event.remove(engine, "before_cursor_execute", reject)
    with engine.connect() as connection:
        assert connection.scalar(text("SELECT count(*) FROM search_fragments")) == 3
        assert connection.scalar(text("SELECT count(*) FROM source_snapshots")) == 1


def test_corrupt_installed_snapshot_is_rejected(faq_package, engine, tmp_path):
    from atrag.ingestion.services.index import index_package

    data = tmp_path / "server"
    index_package(faq_package, engine, data)
    document = load_package(faq_package)
    path = data / document.snapshot.local_path.parent / "manifest.json"
    manifest = json.loads(path.read_bytes())
    manifest.update(snapshot_id="a" * 64, preparation_sha256="a" * 64)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Stored snapshot manifest"):
        index_package(faq_package, engine, data)


def test_saved_html_smoke_reprepares_without_duplicate_versions(
    faq_snapshot, tmp_path, database_dsn
):
    import runpy

    run = runpy.run_path("tools/faq_ingestion_smoke.py")["run"]
    path = tmp_path / faq_snapshot.local_path.parent / "manifest.json"
    first = run(path, tmp_path / "synthetic", database_dsn)
    second = run(path, tmp_path / "synthetic", database_dsn)
    assert first["synthetic"] and first["faq_count"] == first["fragment_count"] == 1
    assert first["snapshot_id"] == second["snapshot_id"]
    assert second["cache_replay"]["requests"] == 0
