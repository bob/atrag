"""Saved HTML collection fixture shared by parser and pipeline tests."""

from pathlib import Path

import pytest

from atrag.ingestion.services.chunks import chunk_parsed
from atrag.ingestion.services.faq_extract import extract_collection
from atrag.ingestion.services.structure import parse_extraction
from atrag.schemas.ingestion.fragments import ChunkParameters, SearchFragment

URL = "https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/questoes_frequentes/pages/faqs-00267.aspx"


def card(
    number="42", answer="<p>First condition.</p><p>Second condition.</p>", position=1
):
    return (
        f'<div class="card"><div class="card-header" id="heading{position}">'
        f'<a aria-controls="collapse{position}" href="#collapse{position}">{number} - Question?</a></div>'
        f'<div id="collapse{position}" data-bs-parent="#faqAccordion" aria-labelledby="heading{position}">'
        f'<div class="card-body">{answer}</div></div></div>'
    )


@pytest.fixture
def faq_snapshot(tmp_path, request):
    return make_faq_snapshot(tmp_path, getattr(request, "param", {}))


def make_faq_snapshot(tmp_path, options):
    import hashlib
    from datetime import UTC, datetime

    from atrag.ingestion.faq_navigation import CONTENT_ID, inspect_html
    from atrag.ingestion.services.artifacts import write_json
    from atrag.ingestion.services.snapshots import import_snapshot
    from atrag.ingestion.services.sources import load_source_registry
    from atrag.schemas.ingestion.collections import HTMLCollection, HTMLPage
    from atrag.settings import PROJECT_ROOT

    source = load_source_registry(PROJECT_ROOT / "data/sources/faq.json")
    saved = tmp_path / "saved"
    saved.mkdir()
    catalog_url = str(source.source_url)
    title = options.get("title", '<h1 class="tittle-tab-page">IRS &gt; Topic</h1>')
    cards = options.get("cards", card(answer=options.get("answer", "Answer")))
    contents = {
        catalog_url: f'<div id="{CONTENT_ID}"><ul class="submenu-itens-sub"><li><a href="{URL}">Topic</a></li></ul></div>'.encode(),
        URL: f'{title}<div id="{CONTENT_ID}"><div id="faqAccordion">{cards}</div></div>'.encode(),
    }
    if extra_url := options.get("extra_url"):
        contents[catalog_url] = contents[catalog_url].replace(
            b"</ul>", f'<li><a href="{extra_url}">Other topic</a></li></ul>'.encode()
        )
        contents[extra_url] = contents[URL]
    pages = []
    for n, (url, raw) in enumerate(contents.items()):
        name = f"page-{n}.html"
        (saved / name).write_bytes(raw)
        pages.append(
            HTMLPage(
                url=url,
                final_url=options.get("final_url", url) if url == URL else url,
                local_path=name,
                sha256=hashlib.sha256(raw).hexdigest(),
                size_bytes=len(raw),
                fetched_at=datetime.now(UTC),
                http_status=200,
                navigation_urls=inspect_html(raw, url).navigation_urls,
            )
        )
    collection = HTMLCollection(catalog_url=catalog_url, pages=tuple(pages))
    write_json(saved / "collection.json", collection.model_dump(mode="json"))
    return import_snapshot(
        source.model_copy(update={"local_path": saved}), tmp_path
    ).snapshot


def prepare_faq(snapshot, data, parameters=None):
    extracted = Path(extract_collection(snapshot, data)["directory"]) / "manifest.json"
    parsed = parse_extraction(extracted, data, snapshot.source)
    chunked = chunk_parsed(
        parsed.directory / "manifest.json",
        data,
        snapshot.source,
        parameters or ChunkParameters(),
    )
    return extracted, parsed, chunked


def make_faq_package(
    faq_snapshot, tmp_path, *, model="text-embedding-3-large", parameters=None
):
    from fixtures.prepared import digest, refresh_manifest

    from atrag.ingestion.services.artifacts import write_json, write_jsonl
    from atrag.ingestion.services.packages import build_package
    from atrag.schemas.ingestion.embeddings import FragmentEmbedding

    _, parsed, chunked = prepare_faq(faq_snapshot, tmp_path, parameters)
    parts = [
        SearchFragment.model_validate_json(line)
        for line in (chunked.directory / "fragments.jsonl").read_bytes().splitlines()
    ]
    embedded = tmp_path / "embedded"
    embedded.mkdir(exist_ok=True)
    write_jsonl(
        embedded / "embeddings.jsonl",
        [
            FragmentEmbedding(
                model=model,
                dimensions=1536,
                input_hash=p.embedding_input_hash,
                embedding=[1.0] + [0.0] * 1535,
                snapshot_id=p.snapshot_id,
                fragment_id=p.fragment_id,
            )
            for p in parts
        ],
    )
    write_json(embedded / "report.json", {})
    write_json(
        embedded / "manifest.json",
        {
            "stage": "embed",
            "status": "succeeded",
            "snapshot_id": faq_snapshot.snapshot_id,
            "model": model,
            "dimensions": 1536,
            "fragment_count": len(parts),
            "input_sha256": {
                name: digest((chunked.directory / name).read_bytes())
                for name in ("manifest.json", "fragments.jsonl", "report.json")
            },
            "output_sha256": {name: "" for name in ("embeddings.jsonl", "report.json")},
        },
    )
    refresh_manifest(embedded / "manifest.json")
    return build_package(
        parsed.directory / "manifest.json",
        chunked.directory / "manifest.json",
        embedded / "manifest.json",
        tmp_path,
        tmp_path / "package",
    )
