import fcntl
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from threading import Barrier, Lock
from uuid import UUID

import httpx
import pytest

from atrag.cli import main
from atrag.ingestion.client.faq import fetch_html
from atrag.ingestion.faq_navigation import inspect_html, normalize_faq_url
from atrag.ingestion.services.fetch import fetch_collection
from atrag.ingestion.services.snapshots import import_snapshot, read_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.settings import PROJECT_ROOT

HOST = "https://info.portaldasfinancas.gov.pt"
BASE = f"{HOST}/pt/apoio_contribuinte/questoes_frequentes"
CATALOG = f"{BASE}/Pages/faqs.aspx"
TOPIC = f"{BASE}/pages/faqs-00267.aspx"
OTHER = f"{BASE}/pages/faqs-00720.aspx"
CONTENT = "ctl00_PlaceHolderMain_ctl02__ControlWrapper_RichHtmlField"


def catalog(*targets):
    links = "".join(f'<li><a href="{url}">Topic</a></li>' for url in targets)
    return (
        f'<html><div id="{CONTENT}"><ul class="submenu-itens-sub">'
        f"{links}</ul></div></html>"
    ).encode()


def topic(extra=""):
    return (
        f'<html><div id="{CONTENT}"><div id="faqAccordion">'
        '<div class="card-header" id="heading01"><a aria-controls="collapse01" '
        'href="#collapse01">3286 - Question?</a></div>'
        '<div id="collapse01" data-bs-parent="#faqAccordion" '
        'aria-labelledby="heading01"><div class="card-body">Answer'
        f'<a href="{OTHER}">Related answer</a></div></div></div>{extra}'
        "</div></html>"
    ).encode()


@pytest.fixture
def source():
    return load_source_registry(PROJECT_ROOT / "data/sources/faq.json")


def client_for(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def response(body, **kwargs):
    return httpx.Response(
        200, content=body, headers={"content-type": "text/html"}, **kwargs
    )


def test_fetch_preserves_bytes_and_complete_provenance(source, tmp_path):
    requests = []
    inputs = {CATALOG: catalog(TOPIC, TOPIC + "#heading01"), TOPIC: topic()}

    def handler(request):
        requests.append(str(request.url))
        return response(inputs[str(request.url)])

    with client_for(handler) as client:
        result = fetch_collection(source, tmp_path, client=client)
    snapshot = result.snapshot
    assert sorted(requests) == sorted(inputs)
    assert result.reused is False
    assert isinstance(result.fetch_id, UUID)
    assert snapshot.source == source
    assert snapshot.schema_version == 2
    assert snapshot.size_bytes == sum(map(len, inputs.values()))
    assert snapshot.collection.catalog_url == source.source_url
    assert len(snapshot.collection.pages) == 2
    for page in snapshot.collection.pages:
        assert page.http_status == 200
        assert page.headers["content-type"] == "text/html"
        assert page.fetched_at.tzinfo is not None
        assert str(page.final_url) == str(page.url)
        assert page.sha256 == hashlib.sha256(inputs[str(page.url)]).hexdigest()
        assert (
            tmp_path / snapshot.local_path.parent / page.local_path
        ).read_bytes() == inputs[str(page.url)]
    assert (
        read_snapshot(
            tmp_path / snapshot.local_path.parent,
            snapshot.source_sha256,
            snapshot.size_bytes,
        )
        == snapshot
    )
    run = json.loads(
        (tmp_path / "runs" / str(result.run_id) / "manifest.json").read_text()
    )
    assert run["status"] == "succeeded"
    assert run["fetch_id"] == str(result.fetch_id)
    assert run["snapshot_id"] == snapshot.snapshot_id


@pytest.mark.parametrize(
    ("href", "expected"),
    [
        ("../pages/faqs-00267.aspx#heading20", TOPIC),
        (
            "http://info.portaldasfinancas.gov.pt:80/pt/apoio_contribuinte/questoes_frequentes/Pages/faqs-00267.aspx",
            TOPIC,
        ),
        (TOPIC + "?page=2&filter=irs#heading20", TOPIC + "?page=2&filter=irs"),
        (CATALOG.lower(), CATALOG),
        ("https://example.org/faqs.aspx", None),
        (HOST + "/pt/codigos_tributarios/Pages/cirs.aspx", None),
        (
            "https://user:password@info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/questoes_frequentes/pages/faqs-00267.aspx",
            None,
        ),
        (TOPIC.replace(HOST, HOST + ":444"), None),
        ("javascript:__doPostBack('x','2')", None),
        (TOPIC.replace(HOST, HOST + ":bad"), None),
        (TOPIC.replace(HOST, HOST + ":99999"), None),
        (TOPIC.replace("https:", "ftp:"), None),
        ("http://[broken", None),
    ],
)
def test_navigation_url_scope(href, expected):
    assert normalize_faq_url(CATALOG, href) == expected


def test_navigation_retains_pagination_and_ignores_answer_links():
    next_url = TOPIC + "?page=2"
    raw = topic(f'<a rel="next" href="{next_url}#top">Next</a>')
    inspection = inspect_html(raw, TOPIC)
    assert inspection.navigation_urls == (next_url,)
    assert inspection.template == "faq-accordion"
    assert inspection.question_count == 1


@pytest.mark.parametrize(
    "raw",
    [b"<html>Error</html>", topic().replace(b"3286 - Question?", b"Error"), catalog()],
)
def test_unknown_or_empty_template_is_rejected(raw):
    with pytest.raises(ValueError):
        inspect_html(raw, CATALOG)


@pytest.mark.parametrize("failure", [429, 500, 503, "timeout"])
def test_transient_failures_are_retried_with_delay(source, tmp_path, failure):
    calls, delays = [], []

    def handler(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            if failure == "timeout":
                raise httpx.ReadTimeout("slow", request=request)
            return httpx.Response(failure, headers={"retry-after": "2"})
        return response(catalog(TOPIC) if str(request.url) == CATALOG else topic())

    with client_for(handler) as client:
        result = fetch_collection(source, tmp_path, client=client, sleep=delays.append)
    assert calls.count(CATALOG) == 2
    assert delays == [2.0 if failure != "timeout" else 1.0]
    assert len(result.snapshot.collection.pages) == 2


@pytest.mark.parametrize("failure", [404, 206, 503, "type", "template", "redirect"])
def test_incomplete_fetch_never_publishes_and_can_resume(source, tmp_path, failure):
    calls = []

    def handler(request):
        url = str(request.url)
        calls.append(url)
        if url == CATALOG:
            return response(catalog(TOPIC))
        if isinstance(failure, int):
            return httpx.Response(failure)
        if failure == "type":
            return httpx.Response(
                200, content=b"PDF", headers={"content-type": "application/pdf"}
            )
        if failure == "redirect":
            return httpx.Response(
                302, headers={"location": "https://example.org/error"}
            )
        return response(b"<html>SharePoint error</html>")

    with client_for(handler) as client, pytest.raises(ValueError):
        fetch_collection(source, tmp_path, client=client, sleep=lambda _: None)
    assert not list((tmp_path / "raw").glob("[0-9a-f]*"))
    [state_path] = (tmp_path / "fetch").glob("*/state.json")
    state = json.loads(state_path.read_text())
    assert len(state["pages"]) == 1
    assert TOPIC in state["failures"]
    fetch_id = UUID(state_path.parent.name)
    calls.clear()

    def recovered(request):
        calls.append(str(request.url))
        return response(topic())

    with client_for(recovered) as client:
        result = fetch_collection(source, tmp_path, client=client, resume=fetch_id)
    assert calls == [TOPIC]
    assert result.fetch_id == fetch_id
    assert len(result.snapshot.collection.pages) == 2


def test_saved_snapshot_import_and_inspect_are_offline(
    source, tmp_path, monkeypatch, capsys
):
    def handler(request):
        return response(catalog(TOPIC) if str(request.url) == CATALOG else topic())

    with client_for(handler) as client:
        fetched = fetch_collection(source, tmp_path, client=client)
    directory = tmp_path / fetched.snapshot.local_path.parent
    saved_source = source.model_copy(
        update={"local_path": directory / "collection.json"}
    )
    first = import_snapshot(saved_source, tmp_path)
    second = import_snapshot(saved_source, tmp_path)
    assert first.snapshot == second.snapshot == fetched.snapshot
    assert first.reused is second.reused is True
    registry = tmp_path / "registry.json"
    registry.write_text(saved_source.model_dump_json())
    monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path))
    main(["ingest", "inspect", "--registry", str(registry)])
    summary = json.loads(capsys.readouterr().out)
    assert summary["page_count"] == 2
    assert summary["question_count"] == 1
    assert summary["review_status"] == "needs_review"
    assert summary["snapshot_id"] == fetched.snapshot.snapshot_id
    report = json.loads((Path(summary["directory"]) / "report.json").read_text())
    assert report["source_format"] == "html_collection"
    assert report["question_count"] == 1


def test_fetch_cli_exposes_resumable_run(source, tmp_path, monkeypatch, capsys):
    def handler(request):
        return response(catalog(TOPIC) if str(request.url) == CATALOG else topic())

    with client_for(handler) as client:
        monkeypatch.setattr(
            "atrag.cli.fetch_collection",
            lambda source, data_dir, **kwargs: fetch_collection(
                source, data_dir, client=client, **kwargs
            ),
        )
        monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path))
        main(
            [
                "ingest",
                "fetch",
                "--registry",
                str(PROJECT_ROOT / "data/sources/faq.json"),
            ]
        )
    result = json.loads(capsys.readouterr().out)
    assert UUID(result["fetch_id"])
    assert result["snapshot"]["source"]["source_id"] == source.source_id


def test_real_step0_pages_are_recognized_without_accordion_container_limit():
    evidence = PROJECT_ROOT / "data/faq-step0-20261107/html"
    if not evidence.exists():
        pytest.skip("Full Step 0 crawl is intentionally outside Git")
    pages = [
        inspect_html(path.read_bytes(), CATALOG) for path in evidence.glob("*.html")
    ]
    assert len(pages) == 151
    assert sum(page.question_count for page in pages) == 1630


def test_pagination_cycle_and_redirect_keep_requested_and_final_urls(source, tmp_path):
    paged = TOPIC + "?page=2"
    final = TOPIC.replace("/pages/", "/Pages/")
    inputs = {
        CATALOG: catalog(TOPIC, TOPIC.replace("/pages/", "/Pages/") + "#a"),
        final: topic(f'<a rel="next" href="{paged}">Next</a>'),
        paged: topic(f'<a rel="prev" href="{TOPIC}">Back</a>'),
    }
    calls = []

    def handler(request):
        url = str(request.url)
        calls.append(url)
        if url == TOPIC:
            return httpx.Response(302, headers={"location": final + "#heading01"})
        return response(inputs[url])

    with client_for(handler) as client:
        result = fetch_collection(source, tmp_path, client=client)
    pages = {str(p.url): p for p in result.snapshot.collection.pages}
    assert len(pages) == 3
    assert str(pages[TOPIC].final_url) == final
    assert tuple(map(str, pages[TOPIC].navigation_urls)) == (paged,)
    assert len(calls) == 4
    assert len({p.local_path for p in pages.values()}) == 3


@pytest.mark.parametrize("failure", [429, 500, "transport"])
def test_retry_budget_is_exact_and_failure_has_no_snapshot(source, tmp_path, failure):
    calls, delays = [], []

    def handler(request):
        calls.append(str(request.url))
        if failure == "transport":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(failure, headers={"retry-after": "600"})

    with client_for(handler) as client, pytest.raises(ValueError):
        fetch_collection(source, tmp_path, client=client, sleep=delays.append)
    assert calls == [CATALOG] * 3
    assert delays == ([1.0, 2.0] if failure == "transport" else [30.0, 30.0])
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("retry_after", ["bad-date", "0", "-1", "date"])
def test_retry_after_variants(retry_after):
    calls, delays = [], []
    value = (
        format_datetime(datetime.now(UTC) + timedelta(seconds=10))
        if retry_after == "date"
        else retry_after
    )

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": value})
        return response(topic())

    with client_for(handler) as client:
        fetched = fetch_html(client, TOPIC, sleep=delays.append)
    assert fetched.raw == topic()
    if retry_after == "date":
        assert 8 < delays[0] <= 10
    else:
        assert delays == ([1.0] if retry_after == "bad-date" else [0.0])


@pytest.mark.parametrize(
    "headers", [{}, {"location": TOPIC}, {"location": "https://example.org/x"}]
)
def test_bad_redirect_does_not_escape_or_loop_forever(headers):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers=headers)

    with client_for(handler) as client, pytest.raises(ValueError):
        fetch_html(client, TOPIC)
    assert set(calls) == {TOPIC}
    assert len(calls) == (6 if headers.get("location") == TOPIC else 1)


def test_keyboard_interrupt_resumes_verified_pages(source, tmp_path, monkeypatch):
    from atrag.ingestion.services import fetch

    def handler(request):
        return response(catalog(TOPIC) if str(request.url) == CATALOG else topic())

    original = fetch._get_page

    def interrupted(client, url, sleep):
        if url == TOPIC:
            raise KeyboardInterrupt
        return original(client, url, sleep)

    with client_for(handler) as client, monkeypatch.context() as patch:
        patch.setattr(fetch, "_get_page", interrupted)
        with pytest.raises(KeyboardInterrupt):
            fetch_collection(source, tmp_path, client=client)
    [checkpoint] = (tmp_path / "fetch").glob("*/state.json")
    assert len(json.loads(checkpoint.read_text())["pages"]) == 1
    assert not (tmp_path / "raw").exists()
    with client_for(handler) as client:
        result = fetch_collection(
            source, tmp_path, client=client, resume=UUID(checkpoint.parent.name)
        )
    assert len(result.snapshot.collection.pages) == 2


def fetch_small(source, tmp_path):
    with client_for(
        lambda request: response(
            catalog(TOPIC) if str(request.url) == CATALOG else topic()
        )
    ) as client:
        return fetch_collection(source, tmp_path, client=client)


def test_resume_has_only_one_owner_and_rejects_changed_source(source, tmp_path):
    first = fetch_small(source, tmp_path)
    directory = tmp_path / "fetch" / str(first.fetch_id)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with pytest.raises(ValueError, match="owner"):
            fetch_collection(source, tmp_path, resume=first.fetch_id)
    changed = source.model_copy(update={"document_title": "Changed"})
    with pytest.raises(ValueError, match="registry"):
        fetch_collection(changed, tmp_path, resume=first.fetch_id)


@pytest.mark.parametrize(
    "damage", ["bytes", "size", "navigation", "final-url", "url", "path"]
)
def test_resume_rejects_corrupt_checkpoint_before_http(source, tmp_path, damage):
    first = fetch_small(source, tmp_path)
    directory = tmp_path / "fetch" / str(first.fetch_id)
    state_path = directory / "state.json"
    state = json.loads(state_path.read_text())
    page = state["pages"][0]
    if damage == "bytes":
        (directory / page["local_path"]).write_bytes(b"corrupt")
    elif damage == "size":
        page["size_bytes"] += 1
    elif damage == "navigation":
        page["navigation_urls"] = []
    elif damage == "final-url":
        page["final_url"] = "https://example.org/faq"
    elif damage == "url":
        page["url"] = "https://example.org/faq"
    else:
        page["local_path"] = "../escape.html"
    state_path.write_text(json.dumps(state))
    with (
        client_for(
            lambda _: pytest.fail("Corrupt checkpoints must fail before HTTP")
        ) as client,
        pytest.raises(ValueError),
    ):
        fetch_collection(source, tmp_path, client=client, resume=first.fetch_id)


@pytest.mark.parametrize("damage", ["page", "collection", "manifest"])
def test_offline_import_rejects_corrupt_snapshot(source, tmp_path, damage):
    first = fetch_small(source, tmp_path)
    directory = tmp_path / first.snapshot.local_path.parent
    if damage == "page":
        page = first.snapshot.collection.pages[0]
        (directory / page.local_path).write_bytes(b"corrupt")
    elif damage == "collection":
        (directory / "collection.json").write_text("{}")
    else:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifest["size_bytes"] += 1
        (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        read_snapshot(
            directory, first.snapshot.source_sha256, first.snapshot.size_bytes
        )


def test_repeated_and_concurrent_fetches_keep_first_immutable_snapshot(
    source, tmp_path
):
    barrier = Barrier(3)
    active = 0
    maximum = 0
    lock = Lock()

    def handler(request):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        if str(request.url) != CATALOG:
            barrier.wait(timeout=5)
        raw = (
            catalog(TOPIC, OTHER, TOPIC + "?page=2")
            if str(request.url) == CATALOG
            else topic()
        )
        with lock:
            active -= 1
        return response(raw)

    with client_for(handler) as client:
        first = fetch_collection(source, tmp_path, client=client)
    assert maximum == 3
    manifest_path = tmp_path / first.snapshot.local_path.parent / "manifest.json"
    manifest_before = manifest_path.read_bytes(), manifest_path.stat().st_mtime_ns

    with (
        client_for(
            lambda request: response(
                catalog(TOPIC, OTHER, TOPIC + "?page=2")
                if str(request.url) == CATALOG
                else topic()
            )
        ) as client,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        results = list(
            pool.map(
                lambda _: fetch_collection(source, tmp_path, client=client),
                range(2),
            )
        )
    assert all(
        result.reused and result.snapshot == first.snapshot for result in results
    )
    assert manifest_before == (
        manifest_path.read_bytes(),
        manifest_path.stat().st_mtime_ns,
    )
    assert len(list((tmp_path / "raw").iterdir())) == 1


def test_page_limit_blocks_unbounded_pagination(source, tmp_path, monkeypatch):
    monkeypatch.setattr("atrag.ingestion.services.fetch.MAX_PAGES", 2)
    with (
        client_for(lambda _: response(catalog(TOPIC, OTHER))) as client,
        pytest.raises(ValueError, match="pagination"),
    ):
        fetch_collection(source, tmp_path, client=client)
    assert not (tmp_path / "raw").exists()


def test_unsupported_pagination_blocks_completeness():
    raw = topic('<a href="javascript:__doPostBack()">Next</a>')
    with pytest.raises(ValueError, match="pagination"):
        inspect_html(raw, TOPIC)


def test_numbered_heading_whitespace_and_partial_error_are_checked():
    raw = topic().replace(b"3286 - Question?", b" \n <span>3286</span> - Question? \n ")
    assert inspect_html(raw, TOPIC).question_count == 1
    broken = (
        raw + b'<div class="card-header"><a aria-controls="missing">Error</a></div>'
    )
    with pytest.raises(ValueError, match="accordion"):
        inspect_html(broken, TOPIC)
    with pytest.raises(ValueError, match="template"):
        inspect_html(f'<div id="{CONTENT}">Error</div>'.encode(), TOPIC)
    with pytest.raises(ValueError, match="no in-scope"):
        inspect_html(catalog("https://example.org/topic"), CATALOG)


def test_symlink_escape_is_rejected(source, tmp_path):
    first = fetch_small(source, tmp_path)
    directory = tmp_path / first.snapshot.local_path.parent
    page = first.snapshot.collection.pages[0]
    outside = tmp_path / "outside.html"
    original = directory / page.local_path
    outside.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        read_snapshot(
            directory, first.snapshot.source_sha256, first.snapshot.size_bytes
        )


def test_matching_and_wrong_expected_collection_checksums(source, tmp_path):
    first = fetch_small(source, tmp_path)
    saved = tmp_path / "fetch" / str(first.fetch_id)
    for digest in (first.snapshot.source_sha256, "0" * 64):
        checked = source.model_copy(
            update={"local_path": saved, "expected_sha256": digest}
        )
        if digest == first.snapshot.source_sha256:
            assert import_snapshot(checked, tmp_path).snapshot == first.snapshot
        else:
            with pytest.raises(ValueError, match="checksum"):
                import_snapshot(checked, tmp_path)


def test_collection_manifest_change_cannot_be_hidden_by_same_byte_identity(
    source, tmp_path
):
    first = fetch_small(source, tmp_path)
    directory = tmp_path / first.snapshot.local_path.parent
    path = directory / "collection.json"
    data = json.loads(path.read_text())
    data["pages"][0]["headers"]["x-new"] = "different provenance"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs"):
        read_snapshot(
            directory, first.snapshot.source_sha256, first.snapshot.size_bytes
        )


def test_wrong_source_and_page_limit_boundary(source, tmp_path, monkeypatch):
    pdf = load_source_registry(PROJECT_ROOT / "data/sources/cirs.json")
    with pytest.raises(ValueError, match="HTML collection"):
        fetch_collection(pdf, tmp_path)
    with pytest.raises(ValueError, match="catalog"):
        fetch_collection(source.model_copy(update={"source_url": None}), tmp_path)
    monkeypatch.setattr("atrag.ingestion.services.fetch.MAX_PAGES", 2)
    assert len(fetch_small(source, tmp_path).snapshot.collection.pages) == 2


def test_failed_publication_resumes_without_http(source, tmp_path, monkeypatch):
    original = Path.rename

    def fail(path, target):
        if path.name.startswith(".snapshot-"):
            raise PermissionError("simulate interrupted publication")
        return original(path, target)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "rename", fail)
        with pytest.raises(PermissionError):
            fetch_small(source, tmp_path)
    assert list((tmp_path / "raw").iterdir()) == []
    [state] = (tmp_path / "fetch").glob("*/state.json")
    with client_for(lambda _: pytest.fail("Completed pages must be reused")) as client:
        resumed = fetch_collection(
            source, tmp_path, client=client, resume=UUID(state.parent.name)
        )
    assert len(resumed.snapshot.collection.pages) == 2


def test_snapshot_source_override_cli_and_invalid_fetch_cli(
    source, tmp_path, monkeypatch, capsys
):
    fetched = fetch_small(source, tmp_path)
    directory = tmp_path / fetched.snapshot.local_path.parent
    monkeypatch.setenv("ATRAG_DATA_DIR", str(tmp_path))
    main(
        [
            "ingest",
            "snapshot",
            "--registry",
            str(PROJECT_ROOT / "data/sources/faq.json"),
            "--source",
            str(directory),
        ]
    )
    result = json.loads(capsys.readouterr().out)
    assert result["reused"] is True
    assert result["snapshot"]["snapshot_id"] == fetched.snapshot.snapshot_id
    with pytest.raises(SystemExit) as error:
        main(["ingest", "fetch"])
    assert error.value.code == 2
    assert "fetch" in capsys.readouterr().err


def test_html_inspect_rejects_changed_snapshot(source, tmp_path):
    from atrag.ingestion.services.html_inspect import inspect_collection

    fetched = fetch_small(source, tmp_path)
    changed = fetched.snapshot.model_copy(update={"ingested_at": datetime.now(UTC)})
    with pytest.raises(ValueError, match="changed"):
        inspect_collection(changed, tmp_path)


def test_fetch_owns_and_closes_its_http_client(source, tmp_path, monkeypatch):
    client_type = httpx.Client
    clients = []

    def local_client(**kwargs):
        client = client_type(
            transport=httpx.MockTransport(
                lambda request: response(
                    catalog(TOPIC) if str(request.url) == CATALOG else topic()
                )
            ),
            **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr("atrag.ingestion.services.fetch.httpx.Client", local_client)
    fetched = fetch_collection(source, tmp_path)
    assert len(fetched.snapshot.collection.pages) == 2
    assert len(clients) == 1
    assert clients[0].is_closed
