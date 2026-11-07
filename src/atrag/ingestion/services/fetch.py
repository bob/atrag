"""Resume a bounded navigation crawl and publish only a complete collection."""

import fcntl
import hashlib
import time
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID, uuid4

import httpx

from atrag.ingestion.client.faq import fetch_html
from atrag.ingestion.faq_navigation import inspect_html, normalize_faq_url
from atrag.ingestion.services.artifacts import write_json
from atrag.ingestion.services.runs import record_run
from atrag.ingestion.services.snapshots import store_html_snapshot, verify_html_page
from atrag.schemas.ingestion.collections import HTMLCollection, HTMLPage
from atrag.schemas.ingestion.fetch import CollectionFetch, FetchState
from atrag.schemas.ingestion.sources import SourceDefinition, SourceFormat

MAX_PAGES = 1000


@contextmanager
def _own_checkpoint(directory: Path) -> Generator[None]:
    with (directory / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("This FAQ fetch already has an active owner") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _load_state(
    directory: Path, source: SourceDefinition, resume: UUID | None
) -> FetchState:
    if resume is None:
        state = FetchState(source=source)
        write_json(directory / "state.json", state.model_dump(mode="json"))
        return state
    state = FetchState.model_validate_json((directory / "state.json").read_bytes())
    if state.source != source:
        raise ValueError("Resume registry differs from the saved fetch source")
    for page in state.pages:
        verify_html_page(directory, page)
    return state


def _pending(state: FetchState) -> list[str]:
    targets = {str(state.source.source_url)}
    targets.update(str(url) for page in state.pages for url in page.navigation_urls)
    return sorted(targets - {str(page.url) for page in state.pages})


def _get_page(client: httpx.Client, url: str, sleep: Callable[[float], None]):
    response = fetch_html(client, url, sleep=sleep)
    inspection = inspect_html(response.raw, response.final_url)
    key = hashlib.sha256(url.encode()).hexdigest()
    page = HTMLPage(
        url=url,
        local_path=f"html/{key}.html",
        sha256=hashlib.sha256(response.raw).hexdigest(),
        size_bytes=len(response.raw),
        final_url=response.final_url,
        fetched_at=response.fetched_at,
        http_status=200,
        headers=response.headers,
        navigation_urls=inspection.navigation_urls,
    )
    return page, response.raw


def _crawl(state: FetchState, directory: Path, client: httpx.Client, sleep) -> None:
    with ThreadPoolExecutor(max_workers=3) as pool:
        while pending := _pending(state):
            if len(state.pages) + len(pending) > MAX_PAGES:
                raise ValueError("FAQ crawl exceeded 1000 pages; inspect pagination")
            failed = False
            futures = {
                url: pool.submit(_get_page, client, url, sleep) for url in pending[:3]
            }
            for url, future in futures.items():
                try:
                    page, raw = future.result()
                except ValueError as error:
                    failed = True
                    state.failures[url] = str(error)
                else:
                    path = directory / page.local_path
                    path.parent.mkdir(exist_ok=True)
                    temporary = path.with_suffix(".tmp")
                    temporary.write_bytes(raw)
                    temporary.replace(path)
                    state.pages.append(page)
                    state.failures.pop(url, None)
                write_json(directory / "state.json", state.model_dump(mode="json"))
            if failed:
                raise ValueError("FAQ fetch incomplete; resume the saved fetch_id")


def fetch_collection(
    source: SourceDefinition,
    data_dir: Path,
    *,
    resume: UUID | None = None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> CollectionFetch:
    if source.source_format != SourceFormat.HTML_COLLECTION:
        raise ValueError("Fetch requires an HTML collection source")
    if normalize_faq_url(str(source.source_url), "faqs.aspx") != str(source.source_url):
        raise ValueError("FAQ source requires the canonical official catalog URL")
    fetch_id = resume or uuid4()
    directory = data_dir / "fetch" / str(fetch_id)
    directory.mkdir(parents=True, exist_ok=resume is not None)
    with (
        _own_checkpoint(directory),
        record_run(
            data_dir, stage="fetch", fetch_id=str(fetch_id), snapshot_id=None
        ) as run,
    ):
        state = _load_state(directory, source, resume)
        if client is None:
            with httpx.Client(headers={"User-Agent": "ATRAG FAQ snapshot/1"}) as owned:
                _crawl(state, directory, owned, sleep)
        else:
            _crawl(state, directory, client, sleep)
        pages = sorted(state.pages, key=lambda page: str(page.url))
        collection = HTMLCollection(
            catalog_url=source.source_url,
            pages=tuple(pages),
        )
        write_json(directory / "collection.json", collection.model_dump(mode="json"))
        snapshot, reused = store_html_snapshot(source, data_dir, directory)
        run.manifest.update(snapshot_id=snapshot.snapshot_id, reused=reused)
    return CollectionFetch(run.run_id, fetch_id, snapshot, reused)
