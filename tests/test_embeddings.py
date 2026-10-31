import hashlib
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest
from openai import OpenAI
from pydantic import SecretStr

from atrag.ingestion.services.artifacts import write_json, write_jsonl
from atrag.ingestion.services.embeddings import embed_chunked
from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.schemas.ingestion.fragments import ChunkParameters, SearchFragment
from atrag.schemas.settings import Settings


class MemoryCache:
    def __init__(self):
        self.rows = {}

    @contextmanager
    def writer(self, space):
        self.space = space
        yield self

    def get(self, hashes):
        return {
            h: self.rows[(self.space.model, self.space.dimensions, h)]
            for h in hashes
            if (self.space.model, self.space.dimensions, h) in self.rows
        }

    def save(self, records):
        for record in records:
            self.rows[(self.space.model, self.space.dimensions, record.input_hash)] = (
                record
            )


@pytest.fixture
def cache():
    return MemoryCache()


@pytest.fixture
def chunked(tmp_path):
    pdf = tmp_path / "original.pdf"
    pdf.write_bytes(b"%PDF-1.7\nEmbedding test\n%%EOF\n")
    source = load_source_registry(Path("data/sources/cirs.json")).model_copy(
        update={"local_path": pdf, "expected_sha256": None}
    )
    settings = Settings(
        data_dir=tmp_path / "data", embedding_dimensions=2, embedding_batch_size=1
    )
    snapshot = import_snapshot(source, settings.data_dir).snapshot
    directory = tmp_path / "chunked"
    directory.mkdir()

    def publish(texts=("Uma regra.", "Outra regra."), **manifest_updates):
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        fragments = [
            SearchFragment(
                fragment_id=hashlib.sha256(str(i).encode()).hexdigest(),
                legal_unit_id=f"pt:cirs:code:article:{i + 1}",
                parent_id=None,
                instrument_id="pt:cirs",
                code="CIRS",
                document_scope="code",
                kind="article",
                role="norm",
                article=str(i + 1),
                citation_label=f"CIRS, artigo {i + 1}",
                snapshot_id=snapshot.snapshot_id,
                page_start=1,
                page_end=1,
                printed_page_labels=[],
                locators=[],
                text_original=text,
                text_normalized=text,
                editorial_notes=[],
                table_ids=[],
                issues=[],
                parser_version="2",
                parameters=ChunkParameters(),
                source=source,
                legal_unit_ids=[f"pt:cirs:code:article:{i + 1}"],
                context_unit_ids=[],
                context=[],
                embedding_text=text,
                embedding_input_hash=hashlib.sha256(text.encode()).hexdigest(),
                token_count=max(1, len(encoding.encode_ordinary(text))),
                tables=[],
                review_status="ready",
            )
            for i, text in enumerate(texts)
        ]
        write_jsonl(directory / "fragments.jsonl", fragments)
        write_json(
            directory / "report.json",
            {"review_status": "needs_review", "omissions": ["known gap"]},
        )
        write_json(
            directory / "manifest.json",
            {
                "stage": "chunk",
                "status": "succeeded",
                "snapshot_id": snapshot.snapshot_id,
                "fragment_count": len(fragments),
                "output_sha256": {
                    name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                    for name in ("fragments.jsonl", "report.json")
                },
                **manifest_updates,
            },
        )
        return directory / "manifest.json", settings, source

    return publish


def response(texts, dimensions=2, **updates):
    return {
        "object": "list",
        "model": "text-embedding-3-small",
        "data": [
            {
                "object": "embedding",
                "index": i,
                "embedding": [float(i + 1)] * dimensions,
            }
            for i, _ in enumerate(texts)
        ],
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
        **updates,
    }


@pytest.fixture
def api():
    clients = []

    def make(handler=None, *, strict=False):
        requests = []

        def handle(request):
            body = json.loads(request.content)
            requests.append(body)
            assert request.url == "https://api.openai.com/v1/embeddings"
            assert body["encoding_format"] == "float"
            assert body["dimensions"] == 2
            if handler:
                return handler(body, len(requests))
            return httpx.Response(200, json=response(body["input"]))

        client = OpenAI(
            api_key="test-secret",
            max_retries=0,
            _strict_response_validation=strict,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        clients.append(client)
        return client, requests

    yield make
    for client in clients:
        client.close()


def run(chunked, cache, client, **kwargs):
    path, settings, source = chunked
    return embed_chunked(path, settings, source, cache=cache, client=client, **kwargs)


def test_batches_are_saved_and_repeat_needs_no_api(chunked, cache, api):
    inputs = chunked()
    client, requests = api()
    first = run(inputs, cache, client)
    second = run(inputs, cache, None)
    assert first.fragment_count == 2
    assert first.run_id != second.run_id
    assert len(requests) == 2
    assert len(cache.rows) == 2
    assert (first.directory / "embeddings.jsonl").read_bytes() == (
        second.directory / "embeddings.jsonl"
    ).read_bytes()
    manifest = json.loads((first.directory / "manifest.json").read_text())
    assert (manifest["stage"], manifest["status"]) == ("embed", "succeeded")
    for name, digest in manifest["output_sha256"].items():
        assert (
            hashlib.sha256((first.directory / name).read_bytes()).hexdigest() == digest
        )
    report = json.loads((second.directory / "report.json").read_text())
    assert report["cache_hits"] == 2
    assert report["usage"] == {"prompt_tokens": 0, "total_tokens": 0}
    assert report["chunk_report"]["omissions"] == ["known gap"]
    first_report = json.loads((first.directory / "report.json").read_text())
    assert first_report["usage"] == {"prompt_tokens": 14, "total_tokens": 14}
    assert first_report["embedded_inputs"] == 2
    assert first_report["schema_version"] == 1


def test_duplicate_text_is_paid_once_and_out_of_order_response_is_mapped(
    chunked, cache, api
):
    path, settings, source = chunked(("Primeira.", "Segunda.", "Primeira."))
    settings = settings.model_copy(update={"embedding_batch_size": 32})

    def handle(body, attempt):
        result = response(body["input"])
        result["data"].reverse()
        return httpx.Response(200, json=result)

    client, requests = api(handle)
    result = run((path, settings, source), cache, client)
    rows = [
        json.loads(line)
        for line in (result.directory / "embeddings.jsonl").read_text().splitlines()
    ]
    assert [row["embedding"] for row in rows] == [[1, 1], [2, 2], [1, 1]]
    assert requests[0]["input"] == ["Primeira.", "Segunda."]


@pytest.mark.parametrize("status", [400, 401, 403, 422])
def test_fatal_api_errors_do_not_retry_and_resume_keeps_success(
    chunked, cache, api, status
):
    inputs = chunked()
    client, requests = api(
        lambda body, n: (
            httpx.Response(200, json=response(body["input"]))
            if n == 1
            else httpx.Response(status, json={"error": {"message": "secret error"}})
        )
    )
    with pytest.raises(ValueError, match="Embedding request failed"):
        run(inputs, cache, client, sleep=lambda _: pytest.fail("fatal error retried"))
    assert len(requests) == 2
    assert len(cache.rows) == 1
    manifests = list(inputs[1].data_dir.glob("runs/*/manifest.json"))
    failed = [
        json.loads(p.read_text())
        for p in manifests
        if json.loads(p.read_text())["status"] == "failed"
    ]
    assert len(failed) == 1
    report = json.loads(
        (inputs[1].data_dir / "runs" / failed[0]["run_id"] / "report.json").read_text()
    )
    assert report["usage"]["total_tokens"] == 7
    assert "secret" not in json.dumps(failed) + json.dumps(report)
    good, resumed = api()
    result = run(inputs, cache, good)
    assert len(resumed) == 1
    assert json.loads((result.directory / "report.json").read_text())["cache_hits"] == 1


@pytest.mark.parametrize("status", [408, 409, 429, 500, 503])
def test_transient_errors_retry_with_bounded_jitter(chunked, cache, api, status):
    inputs = chunked(("Uma regra.",))
    client, requests = api(
        lambda body, n: (
            httpx.Response(status, json={"error": {"message": "temporary"}})
            if n < 3
            else httpx.Response(200, json=response(body["input"]))
        )
    )
    delays = []
    result = run(inputs, cache, client, sleep=delays.append)
    assert len(requests) == 3
    assert 1 <= delays[0] <= 2
    assert 2 <= delays[1] <= 3
    report = json.loads((result.directory / "report.json").read_text())
    assert report["requests"] == 3
    assert report["retries"] == 2


@pytest.mark.parametrize(
    "data",
    [
        [],
        [{"index": 0, "embedding": [1]}],
        [{"index": 1, "embedding": [1, 1]}],
        [{"index": 0, "embedding": [0, 0]}],
        [{"index": 0, "embedding": ["1", 1]}],
        [{"index": 0, "embedding": [1e39, 1]}],
    ],
)
def test_invalid_response_is_not_cached_and_usage_survives(chunked, cache, api, data):
    inputs = chunked(("Uma regra.",))
    client, requests = api(
        lambda body, n: httpx.Response(200, json=response(body["input"], data=data))
    )
    with pytest.raises(ValueError):
        run(inputs, cache, client)
    assert not cache.rows
    assert len(requests) == 1
    reports = list(inputs[1].data_dir.glob("runs/*/report.json"))
    assert json.loads(reports[0].read_text())["usage"]["total_tokens"] == 7


@pytest.mark.parametrize("texts", [(" ",), ("x " * 8200,), ()])
def test_invalid_input_fails_before_any_request(chunked, cache, api, texts):
    inputs = chunked(texts)
    client, requests = api()
    with pytest.raises(ValueError):
        run(inputs, cache, client)
    assert not requests
    reports = list(inputs[1].data_dir.glob("runs/*/report.json"))
    report = json.loads(reports[0].read_text())
    assert report["cache_hits"] == report["embedded_inputs"] == 0


@pytest.mark.parametrize("arguments", [["db"], ["ingest", "embed"]])
def test_embedding_and_database_arguments_are_required(arguments, monkeypatch):
    from atrag.cli import main

    monkeypatch.setattr(
        "atrag.cli.load_settings",
        lambda: pytest.fail(
            "Missing arguments must be rejected before loading configuration"
        ),
    )
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2


def test_tampering_is_rejected_before_api(chunked, cache, api):
    inputs = chunked()
    (inputs[0].parent / "fragments.jsonl").write_text("")
    client, requests = api()
    with pytest.raises(ValueError, match="checksum"):
        run(inputs, cache, client)
    assert not requests


def test_embed_command_reports_safe_summary(chunked, cache, api, monkeypatch, capsys):
    from types import SimpleNamespace

    from atrag.cli import main

    path, settings, source = chunked()
    client, requests = api()
    registry = path.parent / "registry.json"
    registry.write_text(source.model_dump_json())
    settings = settings.model_copy(update={"openai_api_key": SecretStr("test-secret")})
    monkeypatch.setattr("atrag.cli.load_settings", lambda: settings)
    monkeypatch.setattr(
        "atrag.cli.create_db_engine", lambda _: SimpleNamespace(dispose=lambda: None)
    )
    monkeypatch.setattr("atrag.cli.PostgresEmbeddingCache", lambda _: cache)

    def make_client(**kwargs):
        assert kwargs == {
            "api_key": "test-secret",
            "base_url": "https://api.openai.com/v1",
            "max_retries": 0,
            "timeout": 30,
        }
        return client

    monkeypatch.setattr("atrag.cli.OpenAI", make_client)
    main(["ingest", "embed", "--chunked", str(path), "--registry", str(registry)])
    output = capsys.readouterr().out
    summary = json.loads(output)
    assert summary["fragment_count"] == 2
    assert summary["model"] == "text-embedding-3-small"
    assert summary["dimensions"] == 2
    assert len(requests) == 2
    assert "test-secret" not in output
    monkeypatch.setattr(
        "atrag.cli.load_settings",
        lambda: settings.model_copy(update={"openai_api_key": None}),
    )
    main(["ingest", "embed", "--chunked", str(path), "--registry", str(registry)])
    assert json.loads(capsys.readouterr().out)["fragment_count"] == 2
    assert len(requests) == 2


def test_embed_command_sanitizes_database_failure(chunked, monkeypatch, capsys):
    from sqlalchemy.exc import OperationalError

    from atrag.cli import main

    path, settings, source = chunked()
    registry = path.parent / "registry.json"
    registry.write_text(source.model_dump_json())
    monkeypatch.setattr("atrag.cli.load_settings", lambda: settings)

    def fail(_):
        raise OperationalError("secret-dsn", {}, Exception("db-secret"))

    monkeypatch.setattr("atrag.cli.create_db_engine", fail)
    with pytest.raises(SystemExit) as error:
        main(["ingest", "embed", "--chunked", str(path), "--registry", str(registry)])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert "Embedding" in output
    assert "secret" not in output


def test_vector_roundtrip_uses_pgvector_precision():
    from atrag.schemas.ingestion.embeddings import EmbeddingRecord

    record = EmbeddingRecord(
        model="text-embedding-3-small",
        dimensions=2,
        input_hash="a" * 64,
        embedding=[0.1, 0.2],
    )
    assert record.embedding == [0.10000000149011612, 0.20000000298023224]


def test_request_limit_and_input_limit_boundaries():
    from atrag.ingestion.embeddings import input_batches, prepare_input

    item = prepare_input("válido", lambda _: 8192)
    assert item.token_count == 8192
    assert item.input_hash == hashlib.sha256("válido".encode()).hexdigest()
    with pytest.raises(ValueError):
        prepare_input("too long", lambda _: 8193)
    assert [len(b) for b in input_batches([item] * 37, 2048)] == [36, 1]
    exact = prepare_input("text", lambda _: 7500)
    assert [len(b) for b in input_batches([exact] * 41, 2048)] == [40, 1]
    assert [len(b) for b in input_batches([exact] * 81, 2048)] == [40, 40, 1]
    small = prepare_input("small", lambda _: 1)
    assert [len(b) for b in input_batches([small] * 2049, 2048)] == [2048, 1]
    for size in (0, 2049):
        with pytest.raises(ValueError):
            list(input_batches([small], size))
    assert list(input_batches([], 1)) == []


@pytest.mark.parametrize("failure", ["network", "timeout", "rate_limit"])
def test_retries_stop_after_three_attempts(chunked, cache, api, failure):
    inputs = chunked(("Uma regra.",))

    def handle(body, n):
        if failure == "network":
            raise httpx.ConnectError("secret transport error")
        if failure == "timeout":
            raise httpx.ReadTimeout("secret timeout")
        return httpx.Response(429, json={"error": {"message": "rate limit"}})

    client, requests = api(handle)
    delays = []
    with pytest.raises(ValueError, match="Embedding request failed"):
        run(inputs, cache, client, sleep=delays.append)
    assert len(requests) == 3
    assert len(delays) == 2
    assert not cache.rows


def test_exhausted_quota_does_not_retry(chunked, cache, api):
    inputs = chunked(("Uma regra.",))
    client, requests = api(
        lambda body, n: httpx.Response(
            429, json={"error": {"message": "quota", "code": "insufficient_quota"}}
        )
    )
    with pytest.raises(ValueError):
        run(inputs, cache, client, sleep=lambda _: pytest.fail("quota retried"))
    assert len(requests) == 1


@pytest.mark.parametrize(
    "updates",
    [
        {"model": "text-embedding-3-large"},
        {"usage": {"prompt_tokens": -1, "total_tokens": 7}},
        {"usage": None},
        {
            "data": [
                {"index": 0, "embedding": [1, 1]},
                {"index": 0, "embedding": [1, 1]},
            ]
        },
    ],
)
def test_response_metadata_is_validated(chunked, cache, api, updates):
    inputs = chunked()
    client, requests = api(
        lambda body, n: httpx.Response(200, json=response(body["input"], **updates))
    )
    with pytest.raises(ValueError):
        run(inputs, cache, client)
    assert len(requests) == 1
    assert not cache.rows


@pytest.mark.parametrize(
    "field,value",
    [
        ("embedding_input_hash", "e" * 64),
        ("token_count", 1),
        ("snapshot_id", "e" * 64),
        ("source", None),
    ],
)
def test_rehashed_but_inconsistent_fragments_are_rejected(
    chunked, cache, api, field, value
):
    inputs = chunked()
    directory = inputs[0].parent
    rows = [
        json.loads(line)
        for line in (directory / "fragments.jsonl").read_text().splitlines()
    ]
    rows[0][field] = value
    (directory / "fragments.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows)
    )
    manifest = json.loads(inputs[0].read_text())
    manifest["output_sha256"]["fragments.jsonl"] = hashlib.sha256(
        (directory / "fragments.jsonl").read_bytes()
    ).hexdigest()
    write_json(inputs[0], manifest)
    client, requests = api()
    with pytest.raises(ValueError):
        run(inputs, cache, client)
    assert not requests


def test_registry_checksum_is_checked_with_and_without_match(chunked, cache, api):
    path, settings, source = chunked()
    digest = json.loads(path.read_text())["snapshot_id"]
    client, requests = api()
    good = source.model_copy(update={"expected_sha256": digest})
    assert run((path, settings, good), cache, client).fragment_count == 2
    bad = source.model_copy(update={"expected_sha256": "f" * 64})
    with pytest.raises(ValueError, match="registry"):
        run((path, settings, bad), cache, client)
    assert len(requests) == 2


@pytest.mark.parametrize("change", ["count", "duplicate"])
def test_chunk_identity_and_count_are_checked(chunked, cache, api, change):
    path, settings, source = chunked()
    manifest = json.loads(path.read_text())
    if change == "count":
        manifest["fragment_count"] += 1
    else:
        artifact = path.parent / "fragments.jsonl"
        row = artifact.read_text().splitlines()[0]
        artifact.write_text(row + "\n" + row + "\n")
        manifest["output_sha256"]["fragments.jsonl"] = hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest()
    write_json(path, manifest)
    client, requests = api()
    with pytest.raises(ValueError):
        run((path, settings, source), cache, client)
    assert not requests


def test_missing_api_key_stops_uncached_run(chunked, cache):
    inputs = chunked()
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        run(inputs, cache, None)
    reports = list(inputs[1].data_dir.glob("runs/*/report.json"))
    assert json.loads(reports[0].read_text())["requests"] == 0


def test_strict_sdk_response_error_is_sanitized(chunked, cache, api):
    inputs = chunked()
    client, requests = api(
        lambda body, n: httpx.Response(200, json={"secret": "malformed"}), strict=True
    )
    with pytest.raises(ValueError, match="invalid API response"):
        run(inputs, cache, client)
    assert len(requests) == 1


def test_database_migrate_command_disposes_engine(monkeypatch, capsys):
    from types import SimpleNamespace

    from atrag.cli import main

    events = []
    engine = SimpleNamespace(dispose=lambda: events.append("disposed"))
    monkeypatch.setattr("atrag.cli.load_settings", Settings)
    monkeypatch.setattr("atrag.cli.create_db_engine", lambda _: engine)
    monkeypatch.setattr("atrag.cli.migrate_database", lambda e: events.append(e))
    main(["db", "migrate"])
    assert events == [engine, "disposed"]
    assert json.loads(capsys.readouterr().out) == {
        "status": "succeeded",
        "action": "migrate",
    }


def test_database_migrate_command_sanitizes_failure(monkeypatch, capsys):
    from types import SimpleNamespace

    from sqlalchemy.exc import OperationalError

    from atrag.cli import main

    events = []
    monkeypatch.setattr("atrag.cli.load_settings", Settings)
    monkeypatch.setattr(
        "atrag.cli.create_db_engine",
        lambda _: SimpleNamespace(dispose=lambda: events.append("disposed")),
    )

    def fail(_):
        raise OperationalError("secret-dsn", {}, Exception("db-secret"))

    monkeypatch.setattr("atrag.cli.migrate_database", fail)
    with pytest.raises(SystemExit) as error:
        main(["db", "migrate"])
    assert error.value.code == 2
    assert events == ["disposed"]
    assert "secret" not in capsys.readouterr().err
