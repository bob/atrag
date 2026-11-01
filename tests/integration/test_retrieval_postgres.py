"""Search against the migrated PostgreSQL corpus, with synthetic query vectors."""

from types import SimpleNamespace

import pytest

from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package
from atrag.migrations import migrate_database
from atrag.retrieval.search import search_tax_documents
from atrag.schemas.retrieval import SearchFilters

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("mode", ["vector", "fts", "hybrid"])
def test_retrieval_modes_are_independent(indexed, engine, mode):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raw = {
            "model": kwargs["model"],
            "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1535}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        }
        return SimpleNamespace(model_dump=lambda: raw)

    result = search_tax_documents(
        "residência",
        query_language="pt",
        snapshot_id=indexed,
        engine=engine,
        mode=mode,
        client=SimpleNamespace(embeddings=SimpleNamespace(create=create)),
    )
    assert result.status == "ok"
    assert len(calls) == (0 if mode == "fts" else 1)
    if mode == "fts":
        assert all(
            h.vector_rank is None and h.text_rank is not None for h in result.hits
        )
    if mode == "vector":
        assert all(
            h.text_rank is None and h.vector_rank is not None for h in result.hits
        )


def test_evaluate_command_records_all_modes_without_live_model(
    indexed, engine, prepared_document, tmp_path, monkeypatch, capsys, database_dsn
):
    import json
    from pathlib import Path

    import httpx
    from openai import OpenAI

    from atrag.cli import main
    from atrag.ingestion.services.packages import load_package

    data, parsed, chunked, embedded = prepared_document
    package = build_package(parsed, chunked, embedded, data, tmp_path / "eval-package")
    document = load_package(package)
    fragment = next(f for f in document.fragments if f.paragraph == "1")
    identities = (
        fragment.legal_unit_ids
        + fragment.context_unit_ids
        + ["pt:cirs:code:article:68-A"]
    )
    path = tmp_path / "benchmark.json"
    path.write_text(
        json.dumps(
            {
                "snapshot_id": indexed,
                "unit_pages": {
                    u.legal_unit_id: [u.page_start, u.page_end]
                    for u in document.units
                    if u.legal_unit_id in identities
                },
                "cases": [
                    {
                        "case_id": "address",
                        "query": "CIRS 16",
                        "language": "pt",
                        "kind": "exact",
                        "expected_unit_ids": fragment.legal_unit_ids,
                        "required_context_ids": [fragment.legal_unit_id],
                    }
                ],
                "expansion_cases": [
                    {
                        "case_id": "reference",
                        "language": "pt",
                        "kind": "references",
                        "request": {
                            "snapshot_id": indexed,
                            "legal_unit_id": fragment.legal_unit_id,
                            "reference_ids": ["pt:cirs:code:article:68-A"],
                        },
                        "required_unit_ids": [
                            fragment.legal_unit_id,
                            "pt:cirs:code:article:68-A",
                        ],
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("DATABASE_URL", database_dsn)
    monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path / "eval-output"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    main(["evaluate", "--package", str(package), "--benchmark", str(path)])
    output = json.loads(capsys.readouterr().out)
    report = json.loads(Path(output["report"]).read_bytes())
    assert len(report["records"]) == 3
    assert all(r["passed"] for r in report["records"])
    assert report["expansion_records"][0]["passed"]
    assert report["expansion_records"][0]["citations_correct"]
    assert all(r["usage"]["requests"] == 0 for r in report["records"])
    client = OpenAI(
        api_key="synthetic-key",
        http_client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: pytest.fail("Exact evaluation must not call API")
            )
        ),
    )
    monkeypatch.setattr("atrag.cli.create_openai_client", lambda settings: client)
    main(["evaluate", "--package", str(package), "--benchmark", str(path)])
    assert Path(json.loads(capsys.readouterr().out)["report"]).is_file()


@pytest.fixture
def indexed(empty_corpus_schema, engine, prepared_document, tmp_path):
    data, parsed, chunked, embedded = prepared_document
    migrate_database(engine)
    package = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    return index_package(package, engine, tmp_path / "server").snapshot_id


def test_exact_search_preserves_citations_context_and_does_not_need_model(
    indexed, engine
):
    result = search_tax_documents(
        "CIRS 16", query_language="ru", snapshot_id=indexed, engine=engine
    )
    assert result.status == "ok"
    assert result.query_language == "ru"
    assert result.snapshot_id == indexed
    assert result.consolidation_label
    assert len(result.hits) == 2
    assert [h.score for h in result.hits] == pytest.approx([1 / 61, 1 / 62])
    assert all(
        h.fragment.article == "16" and h.fragment.snapshot_id == indexed
        for h in result.hits
    )
    assert all(h.fragment.page_start == h.fragment.page_end == 1 for h in result.hits)
    first = next(h for h in result.hits if h.fragment.paragraph == "1")
    assert "Exceto" in first.fragment.text_original
    assert first.available_reference_ids == ["pt:cirs:code:article:68-A"]
    assert not first.related_units
    assert result.extraction_issues
    assert result.usage == {
        "requests": 0,
        "retries": 0,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def test_absent_document_is_explicit_and_never_returns_cirs(indexed, engine):
    result = search_tax_documents(
        "art. 53 CIVA", query_language="pt", snapshot_id=indexed, engine=engine
    )
    assert result.status == "no_match"
    assert result.reason == "document_not_loaded:CIVA"
    assert result.hits == []
    missing = search_tax_documents(
        "CIRS 999", query_language="en", snapshot_id=indexed, engine=engine
    )
    assert missing.reason == "address_not_found"


@pytest.mark.parametrize(
    "language,query",
    [
        ("ru", "Как определяется резидентство?"),
        ("en", "How is tax residency determined?"),
        ("pt", "Como se determina a residência fiscal?"),
    ],
)
def test_hybrid_uses_indexed_model_and_returns_portuguese(
    indexed, engine, language, query
):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raw = {
            "model": kwargs["model"],
            "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1535}],
            "usage": {"prompt_tokens": 8, "total_tokens": 8},
        }
        return SimpleNamespace(model_dump=lambda: raw)

    client = SimpleNamespace(embeddings=SimpleNamespace(create=create))
    result = search_tax_documents(
        query,
        query_language=language,
        snapshot_id=indexed,
        engine=engine,
        client=client,
    )
    assert result.status == "ok"
    assert result.query_language == language
    assert result.hits[0].fragment.article == "16"
    assert all(h.fragment.source.language == "pt" for h in result.hits)
    assert calls == [
        {
            "model": "text-embedding-3-small",
            "dimensions": 1536,
            "input": [query],
            "encoding_format": "float",
        }
    ]
    assert result.usage == {
        "requests": 1,
        "retries": 0,
        "usage": {"prompt_tokens": 8, "total_tokens": 8},
    }


def test_snapshot_and_context_budget_are_enforced(indexed, engine):
    result = search_tax_documents(
        "CIRS 16", query_language="pt", snapshot_id="b" * 64, engine=engine
    )
    assert result.status == "no_match"
    assert result.reason == "snapshot_not_indexed"
    limited = search_tax_documents(
        "CIRS 16",
        query_language="pt",
        snapshot_id=indexed,
        engine=engine,
        context_tokens=1,
    )
    assert limited.hits == []
    assert limited.reason == "context_budget_exceeded"
    assert "context_budget_exceeded" in limited.limitations


def test_structured_filter_restricts_exact_search(indexed, engine):
    result = search_tax_documents(
        "CIRS 16",
        query_language="pt",
        snapshot_id=indexed,
        engine=engine,
        filters=SearchFilters(paragraph="2"),
    )
    assert len(result.hits) == 1
    assert result.hits[0].fragment.paragraph == "2"


def test_commands_index_and_search_without_source_registry(
    indexed, engine, prepared_document, tmp_path, monkeypatch, capsys, database_dsn
):
    import json

    import httpx
    from openai import OpenAI

    from atrag.cli import main

    data, parsed, chunked, embedded = prepared_document
    package = build_package(parsed, chunked, embedded, data, tmp_path / "cli-package")
    monkeypatch.setenv("DATABASE_URL", database_dsn)
    monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path / "server"))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.chdir(tmp_path)
    main(["ingest", "index", "--package", str(package)])
    assert json.loads(capsys.readouterr().out)["fragment_count"] == 3
    main(
        [
            "search",
            "CIRS 16",
            "--snapshot",
            indexed,
            "--language",
            "en",
            "--limit",
            "1",
            "--context-tokens",
            "12000",
            "--scope",
            "code",
            "--article",
            "16",
            "--code",
            "CIRS",
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert len(result["hits"]) == 1
    assert result["query_language"] == "en"
    assert result["hits"][0]["fragment"]["article"] == "16"

    def handler(request):
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1535}],
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )

    client = OpenAI(
        api_key="synthetic-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    monkeypatch.setattr("atrag.cli.create_openai_client", lambda settings: client)
    main(["search", "Residência fiscal", "--snapshot", indexed, "--language", "pt"])
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_semantic_filters_apply_to_both_candidate_lists(indexed, engine):
    def create(**kwargs):
        raw = {
            "model": kwargs["model"],
            "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1535}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        return SimpleNamespace(model_dump=lambda: raw)

    result = search_tax_documents(
        "residência",
        query_language="pt",
        snapshot_id=indexed,
        engine=engine,
        client=SimpleNamespace(embeddings=SimpleNamespace(create=create)),
        filters=SearchFilters(article="68-A"),
    )
    assert len(result.hits) == 1
    assert result.hits[0].fragment.article == "68-A"
    assert result.hits[0].text_rank is None


def test_search_never_crosses_snapshot_boundaries(indexed, engine, tmp_path):
    from fixtures.prepared import prepare_document

    data, parsed, chunked, embedded = prepare_document(tmp_path / "another-document")
    package = build_package(
        parsed, chunked, embedded, data, tmp_path / "another-package"
    )
    other = index_package(package, engine, tmp_path / "server").snapshot_id
    assert other != indexed

    def create(**kwargs):
        raw = {
            "model": kwargs["model"],
            "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1535}],
            "usage": {"prompt_tokens": 1, "total_tokens": 1},
        }
        return SimpleNamespace(model_dump=lambda: raw)

    for snapshot_id in (indexed, other):
        for query in ("CIRS 16", "residência fiscal"):
            result = search_tax_documents(
                query,
                query_language="pt",
                snapshot_id=snapshot_id,
                engine=engine,
                client=SimpleNamespace(embeddings=SimpleNamespace(create=create)),
            )
            assert result.hits
            assert all(h.fragment.snapshot_id == snapshot_id for h in result.hits)
            assert all(
                u.snapshot_id == snapshot_id
                for h in result.hits
                for u in h.related_units
            )
