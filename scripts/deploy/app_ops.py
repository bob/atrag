"""Service-user operations. No model calls are permitted by smoke checks."""

import argparse
import json
import os
import tomllib
from pathlib import Path

import tiktoken
from alembic.config import Config
from alembic.script import ScriptDirectory
from langgraph.checkpoint.postgres import PostgresSaver
from sqlalchemy import func, select, text

from atrag.corpus import publish_corpus, resolve_corpus
from atrag.db import create_db_engine
from atrag.db_models import FragmentRow, LegalUnitRow, SnapshotRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import load_package
from atrag.ingestion.services.sources import load_source_registry
from atrag.migrations import migrate_database
from atrag.retrieval.search import search_tax_documents
from atrag.settings import load_settings

ROOT = Path(__file__).resolve().parents[2]
SHARED = Path("/opt/atrag/shared")


def warm():
    for name in ("cl100k_base", "o200k_base"):
        tiktoken.get_encoding(name).encode_ordinary("ATRAG")


def check_config(settings):
    load_source_registry(ROOT / "data/sources/cirs.json")
    secrets = tomllib.loads((ROOT / ".streamlit/secrets.toml").read_text())
    assert settings.database_url and settings.openai_api_key
    assert settings.data_dir == SHARED / "data"
    assert secrets["auth"]["redirect_uri"] == (
        "https://tax.example.com/oauth2callback"
    )
    assert secrets["auth"]["google"]["client_id"]
    assert secrets["auth"]["google"]["client_secret"]
    assert len(secrets["auth"]["cookie_secret"]) >= 32
    assert secrets["access"]["check_allowed_emails"] is False
    assert os.environ["TIKTOKEN_CACHE_DIR"] == str(SHARED / "cache/tiktoken")


def bootstrap(engine, settings):
    # This path is only invoked explicitly on the first deployment.
    manifest = SHARED / "packages/cirs-large/manifest.json"
    document = load_package(manifest)
    result = index_package(manifest, engine, settings.data_dir)
    members = {document.snapshot.source.source_id: result.snapshot_id}
    publish_corpus(engine, dict(members))
    expected = {
        "members": members,
        "snapshots": {
            result.snapshot_id: {
                "units": len(document.units),
                "fragments": len(document.fragments),
                "model": document.manifest.model,
                "dimensions": document.manifest.dimensions,
            }
        },
    }
    (SHARED / "corpus-changes.json").write_text(json.dumps(members) + "\n")
    (SHARED / "corpus-expected.json").write_text(json.dumps(expected) + "\n")


def smoke(engine, settings):
    # Reject paid HTTP even if exact retrieval accidentally changes behavior.
    import httpx

    def reject(*args, **kwargs):
        raise RuntimeError("HTTP to model providers is forbidden in deploy smoke")

    httpx.HTTPTransport.handle_request = reject
    expected = json.loads((SHARED / "corpus-expected.json").read_text())
    config = Config()
    config.set_main_option("script_location", str(ROOT / "src/atrag/migrations"))
    heads = set(ScriptDirectory.from_config(config).get_heads())
    with engine.connect() as connection:
        assert (
            set(connection.scalars(text("SELECT version_num FROM alembic_version")))
            == heads
        )
        assert connection.scalar(text("SELECT max(v) FROM checkpoint_migrations")) == (
            len(PostgresSaver.MIGRATIONS) - 1
        )
        assert dict(resolve_corpus(connection).members) == expected["members"]
        for snapshot, counts in expected["snapshots"].items():
            row = connection.execute(
                select(
                    SnapshotRow.embedding_model, SnapshotRow.embedding_dimensions
                ).where(SnapshotRow.snapshot_id == snapshot)
            ).one()
            assert tuple(row) == (counts["model"], counts["dimensions"])
            for model, key in ((LegalUnitRow, "units"), (FragmentRow, "fragments")):
                assert (
                    connection.scalar(
                        select(func.count())
                        .select_from(model)
                        .where(model.snapshot_id == snapshot)
                    )
                    == counts[key]
                )
    snapshot = expected["members"]["pt-at-cirs"]
    for language in ("pt", "en", "uk", "ru"):
        result = search_tax_documents(
            "CIRS 16", 5, language, snapshot_id=snapshot, engine=engine, client=None
        )
        assert result.status == "ok" and result.hits
        assert all(hit.fragment.article == "16" for hit in result.hits)
    print(
        json.dumps(
            {
                "sha": json.loads((ROOT / "release.json").read_text())["sha"],
                "smoke": "ok",
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("config", "warm", "migrate", "bootstrap", "smoke")
    )
    args = parser.parse_args()
    settings = load_settings(ROOT)
    check_config(settings)
    if args.action == "warm":
        warm()
        return
    if args.action == "config":
        return
    engine = create_db_engine(settings)
    try:
        if args.action == "migrate":
            migrate_database(engine)
        elif args.action == "bootstrap":
            bootstrap(engine, settings)
        else:
            smoke(engine, settings)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
