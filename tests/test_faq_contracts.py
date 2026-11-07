"""FAQ collection contracts, without fetching, parsing or inventing PDF data."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from atrag.ingestion.services.snapshots import import_snapshot
from atrag.ingestion.services.sources import load_source_registry
from atrag.schemas.ingestion.faq import FAQEntry
from atrag.schemas.ingestion.snapshots import SourceSnapshot
from atrag.schemas.ingestion.sources import SourceDefinition
from atrag.schemas.ingestion.structure import LegalUnit
from atrag.schemas.materials import HTMLLocation, PreparedMaterial

CATALOG = "https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/questoes_frequentes/Pages/faqs.aspx"
PAGE = "https://info.portaldasfinancas.gov.pt/pt/apoio_contribuinte/questoes_frequentes/pages/faqs-00267.aspx"
WHEN = "2026-11-07T10:00:00Z"


@pytest.fixture
def legacy_contracts():
    directory = Path("tests/fixtures/cirs")
    fixtures = json.loads((directory / "legacy-contracts.json").read_bytes())
    unit = fixtures["unit"]
    unit["data"] = json.loads((directory / unit["data_file"]).read_bytes())[
        unit["index"]
    ]
    return fixtures


@pytest.fixture
def faq_source():
    return load_source_registry(Path("data/sources/faq.json"))


@pytest.fixture
def collection_data():
    return {
        "catalog_url": CATALOG,
        "pages": [
            {
                "url": url,
                "local_path": f"html/{name}.html",
                "sha256": digest,
                "size_bytes": size,
                "final_url": url,
                "fetched_at": WHEN,
                "http_status": 200,
                "headers": {"content-type": "text/html"},
                "navigation_urls": navigation,
            }
            for url, name, digest, size, navigation in (
                (CATALOG, "faqs", "a" * 64, 100, [PAGE]),
                (PAGE, "faqs-00267", "b" * 64, 200, []),
            )
        ],
    }


def collection_snapshot(source, collection):
    # Independent specification of collection identity from Step 0.
    content = sorted(
        ({"url": p["url"], "sha256": p["sha256"]} for p in collection["pages"]),
        key=lambda p: p["url"],
    )
    digest = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()
    return {
        "schema_version": 2,
        "snapshot_id": digest,
        "source_sha256": digest,
        "local_path": f"raw/{digest}/collection.json",
        "size_bytes": sum(p["size_bytes"] for p in collection["pages"]),
        "ingested_at": WHEN,
        "source": source.model_dump(mode="json"),
        "collection": collection,
    }


@pytest.fixture
def faq_unit_data():
    records = json.loads(Path("tests/fixtures/faq/expected.json").read_bytes())
    record = next(r for r in records["records"] if r["entry_id"] == "faqs-00267:3286")
    faq = {
        k: record[k]
        for k in (
            "entry_id",
            "faq_id",
            "question",
            "answer",
            "topic_path",
            "page_url",
            "anchor",
            "links",
            "ordinal",
        )
    }
    return {
        "legal_unit_id": "pt-at-faq:faqs-00267:3286",
        "parent_id": "pt-at-faq:faqs-00267",
        "instrument_id": "pt-at-faq",
        "code": None,
        "document_scope": "faq",
        "kind": "faq_entry",
        "role": "at_explanation",
        "citation_label": "FAQ AT 3286",
        "snapshot_id": "c" * 64,
        "location": {"kind": "html", "page_url": PAGE, "anchor": "heading20"},
        "printed_page_labels": [],
        "locators": [],
        "text_original": faq["question"] + "\n" + faq["answer"],
        "text_normalized": " ".join((faq["question"] + " " + faq["answer"]).split()),
        "editorial_notes": [],
        "table_ids": [],
        "issues": [],
        "faq": faq,
    }


def test_faq_registry_explicitly_declares_collection_format(faq_source):
    assert faq_source.source_id == "pt-at-faq"
    assert faq_source.document_type == "faq"
    assert faq_source.source_format == "html_collection"
    assert str(faq_source.source_url) == CATALOG
    assert faq_source.language == "pt"
    assert faq_source.publisher == "Autoridade Tributária e Aduaneira"
    assert faq_source.downloaded_at is None
    # The publication's meaning does not choose its file format.
    for kind, format_ in (("law", "html_collection"), ("faq", "pdf")):
        source = SourceDefinition.model_validate(
            {
                **faq_source.model_dump(mode="json"),
                "document_type": kind,
                "source_format": format_,
            }
        )
        assert source.source_format == format_
    with pytest.raises(ValidationError):
        SourceDefinition.model_validate(
            {**faq_source.model_dump(), "source_format": "html"}
        )


def test_collection_snapshot_has_no_pdf_and_roundtrips(faq_source, collection_data):
    data = collection_snapshot(faq_source, collection_data)
    snapshot = SourceSnapshot.model_validate(data)
    assert snapshot.snapshot_id == data["source_sha256"]
    assert snapshot.local_path.name == "collection.json"
    assert snapshot.size_bytes == 300
    assert str(snapshot.collection.pages[1].url) == PAGE
    assert snapshot.collection.source_sha256 == snapshot.source_sha256
    assert SourceSnapshot.model_validate_json(snapshot.model_dump_json()) == snapshot
    prepared = SourceSnapshot.model_validate(
        {
            **data,
            "snapshot_id": "d" * 64,
            "preparation_sha256": "d" * 64,
        }
    )
    assert prepared.source_sha256 == snapshot.source_sha256
    assert prepared.local_path == snapshot.local_path


def test_collection_identity_ignores_observations_and_order(
    faq_source, collection_data
):
    first = SourceSnapshot.model_validate(
        collection_snapshot(faq_source, collection_data)
    )
    collection_data["pages"].reverse()
    for page in collection_data["pages"]:
        page.update(fetched_at="2026-11-08T10:00:00Z", headers={}, final_url=CATALOG)
    second = SourceSnapshot.model_validate(
        collection_snapshot(faq_source, collection_data)
    )
    assert second.snapshot_id == first.snapshot_id
    collection_data["pages"][0]["sha256"] = "e" * 64
    changed = SourceSnapshot.model_validate(
        collection_snapshot(faq_source, collection_data)
    )
    assert changed.snapshot_id != first.snapshot_id
    collection_data["pages"][0]["url"] += "?page=2"
    collection_data["pages"][1]["navigation_urls"] = [
        collection_data["pages"][0]["url"]
    ]
    relocated = SourceSnapshot.model_validate(
        collection_snapshot(faq_source, collection_data)
    )
    assert relocated.snapshot_id != changed.snapshot_id


@pytest.mark.parametrize(
    "change",
    [
        {"local_path": "source.pdf"},
        {"local_path": "../escape.html"},
        {"local_path": "/absolute.html"},
        {"local_path": "html\\page.html"},
        {"local_path": "html/./page.html"},
        {"sha256": "invalid"},
        {"size_bytes": 0},
        {"http_status": 404},
        {"http_status": 199},
        {"http_status": 206},
        {"fetched_at": "2026-11-07T10:00:00"},
        {"url": PAGE + "#heading20"},
        {"navigation_urls": [PAGE + "?missing=1"]},
    ],
)
def test_collection_rejects_invalid_page(faq_source, collection_data, change):
    data = collection_snapshot(faq_source, collection_data)
    collection_data["pages"][1].update(change)
    with pytest.raises(ValidationError):
        SourceSnapshot.model_validate(data)


@pytest.mark.parametrize(
    "damage",
    [
        "empty",
        "duplicate_url",
        "duplicate_file",
        "missing_catalog",
        "wrong_catalog",
        "checksum",
        "size",
        "path",
        "format",
        "missing_collection",
        "version",
        "identity",
    ],
)
def test_collection_rejects_inconsistent_snapshot(faq_source, collection_data, damage):
    data = collection_snapshot(faq_source, collection_data)
    pages = collection_data["pages"]
    if damage == "empty":
        pages.clear()
    elif damage == "duplicate_url":
        pages[1]["url"] = pages[0]["url"]
    elif damage == "duplicate_file":
        pages[1]["local_path"] = pages[0]["local_path"]
    elif damage == "missing_catalog":
        pages.pop(0)
    elif damage == "wrong_catalog":
        data["source"]["source_url"] = PAGE
    else:
        field, value = {
            "checksum": ("source_sha256", "f" * 64),
            "size": ("size_bytes", 301),
            "path": ("local_path", "raw/source.pdf"),
            "format": ("source", {**data["source"], "source_format": "pdf"}),
            "missing_collection": ("collection", None),
            "version": ("schema_version", 1),
            "identity": ("snapshot_id", "d" * 64),
        }[damage]
        data[field] = value
    with pytest.raises(ValidationError):
        SourceSnapshot.model_validate(data)


def test_faq_unit_preserves_original_text_and_stable_identity(faq_unit_data):
    unit = LegalUnit.model_validate(faq_unit_data)
    assert unit.page_start is unit.page_end is unit.code is unit.article is None
    assert unit.locators == []
    assert unit.faq.entry_id == "faqs-00267:3286"
    assert unit.faq.question in unit.text_original
    assert unit.faq.answer in unit.text_original
    assert str(unit.location.page_url) == PAGE
    assert LegalUnit.model_validate_json(unit.model_dump_json()) == unit
    faq_unit_data["faq"].update(ordinal=1, anchor="heading01")
    faq_unit_data["location"]["anchor"] = "heading01"
    assert LegalUnit.model_validate(faq_unit_data).legal_unit_id == unit.legal_unit_id
    # Published numbers need not be globally unique.
    faq_unit_data["faq"].update(
        entry_id="faqs-00001:3286", page_url=PAGE.replace("00267", "00001")
    )
    faq_unit_data["location"]["page_url"] = faq_unit_data["faq"]["page_url"]
    faq_unit_data["legal_unit_id"] = "pt-at-faq:faqs-00001:3286"
    assert LegalUnit.model_validate(faq_unit_data).legal_unit_id != unit.legal_unit_id


@pytest.mark.parametrize(
    "change",
    [
        {"code": "CIRS"},
        {"article": "16"},
        {"paragraph": "1"},
        {"subparagraph": "a"},
        {"chapter": "I"},
        {"section": "II"},
        {"page_start": 1},
        {"page_end": 1},
        {"printed_page_labels": ["1"]},
        {
            "locators": [
                {"page_number": 1, "printed_page_labels": [], "bbox": [0, 0, 1, 1]}
            ]
        },
        {"location": {"kind": "pdf", "page": 1}},
        {"location": {"kind": "html", "anchor": "heading20"}},
        {"location": {"kind": "html", "page_url": PAGE, "anchor": "other"}},
        {"location": {"kind": "html", "page_url": CATALOG, "anchor": "heading20"}},
        {"location": None},
        {"faq": None},
        {"role": "norm"},
        {"kind": "article"},
        {"legal_status": "repealed"},
        {
            "editorial_notes": [
                {"kind": "addition", "raw_text": "Aditado", "locators": []}
            ]
        },
        {"document_scope": "code"},
        {"legal_unit_id": "pt:cirs:code:article:16"},
    ],
)
def test_faq_rejects_fabricated_or_inconsistent_addresses(faq_unit_data, change):
    with pytest.raises(ValidationError):
        LegalUnit.model_validate({**faq_unit_data, **change})


@pytest.mark.parametrize(
    "change",
    [
        {"faq_id": ""},
        {"faq_id": "heading20"},
        {"faq_id": "3287"},
        {"entry_id": "faqs-00001:3286"},
        {"question": " "},
        {"answer": ""},
        {"topic_path": []},
        {"topic_path": [" "]},
        {"ordinal": 0},
        {"page_url": "file:///tmp/page.html"},
        {"page_url": PAGE + "#heading20"},
        {
            "links": [
                {
                    "text": "AT",
                    "href": "javascript:alert(1)",
                    "url": "javascript:alert(1)",
                }
            ]
        },
    ],
)
def test_faq_metadata_is_validated_at_boundary(faq_unit_data, change):
    faq_unit_data["faq"].update(change)
    with pytest.raises(ValidationError):
        LegalUnit.model_validate(faq_unit_data)


@pytest.mark.parametrize("kind", ["document", "topic"])
def test_collection_and_topic_are_explanatory_units(faq_unit_data, kind):
    faq_unit_data.update(kind=kind, faq=None, legal_unit_id="pt-at-faq:faqs-00267")
    unit = LegalUnit.model_validate(faq_unit_data)
    assert unit.kind == kind
    assert unit.role == "at_explanation"


def test_reviewed_faq_metadata_preserves_all_records_and_original_links():
    records = json.loads(Path("tests/fixtures/faq/expected.json").read_bytes())[
        "records"
    ]
    fields = set(FAQEntry.model_fields)
    entries = [FAQEntry.model_validate({k: r[k] for k in fields}) for r in records]
    assert len(entries) == 18
    for entry, record in zip(entries, records, strict=True):
        assert entry.model_dump(mode="json") == {k: record[k] for k in fields}
    link = next(link for entry in entries for link in entry.links)
    assert link.href.startswith("/")
    assert str(link.url).startswith("https://info.portaldasfinancas.gov.pt/")


def test_faq_unit_does_not_require_a_legal_code(faq_unit_data):
    del faq_unit_data["code"]
    assert LegalUnit.model_validate(faq_unit_data).code is None


def test_html_location_preserves_page_url_in_prepared_identity():
    content = {
        "source_id": "pt-at-faq",
        "source_sha256": "a" * 64,
        "preparation_version": "1",
        "metadata": {},
        "blocks": [
            {
                "block_id": "3286",
                "text": "Residence",
                "location": {"kind": "html", "page_url": PAGE, "anchor": "heading20"},
            }
        ],
    }
    material = PreparedMaterial.model_validate(content)
    assert str(material.blocks[0].location.page_url) == PAGE
    content["blocks"][0]["location"]["page_url"] = CATALOG
    assert PreparedMaterial.model_validate(content).snapshot_id != material.snapshot_id
    legacy = HTMLLocation(kind="html", anchor="one")
    assert legacy.model_dump(mode="json") == {
        "kind": "html",
        "section": None,
        "anchor": "one",
    }


def test_legacy_pdf_contracts_keep_their_exact_serialization(legacy_contracts):
    fixtures = legacy_contracts
    for name, model in (
        ("source", SourceDefinition),
        ("snapshot", SourceSnapshot),
        ("unit", LegalUnit),
    ):
        fixture = fixtures[name]
        parsed = model.model_validate(fixture["data"])
        assert (
            hashlib.sha256(parsed.model_dump_json().encode()).hexdigest()
            == fixture["serialized_sha256"]
        )
        assert model.model_validate_json(parsed.model_dump_json()) == parsed
    assert (
        SourceDefinition.model_validate(fixtures["source"]["data"]).source_format
        == "pdf"
    )
    for change in ({"local_path": "raw/wrong/source.pdf"}, {"snapshot_id": "d" * 64}):
        with pytest.raises(ValidationError):
            SourceSnapshot.model_validate({**fixtures["snapshot"]["data"], **change})


def test_complete_navigation_can_link_back_to_catalog(faq_source, collection_data):
    collection_data["pages"][1]["navigation_urls"] = [CATALOG]
    snapshot = SourceSnapshot.model_validate(
        collection_snapshot(faq_source, collection_data)
    )
    assert tuple(map(str, snapshot.collection.pages[1].navigation_urls)) == (CATALOG,)


def test_pdf_snapshot_cannot_carry_collection_metadata(faq_source, collection_data):
    data = collection_snapshot(faq_source, collection_data)
    data["source"]["source_format"] = "pdf"
    data["local_path"] = f"raw/{data['source_sha256']}/source.pdf"
    with pytest.raises(ValidationError, match="PDF snapshot"):
        SourceSnapshot.model_validate(data)


def test_faq_requires_a_page_key_in_its_identity(faq_unit_data):
    data = {
        **faq_unit_data["faq"],
        "page_url": "https://example.org/",
        "entry_id": ":3286",
    }
    with pytest.raises(ValidationError):
        FAQEntry.model_validate(data)


@pytest.mark.parametrize("kind", ["document", "topic"])
def test_only_faq_entries_carry_question_metadata(faq_unit_data, kind):
    with pytest.raises(ValidationError, match="Only FAQ entries"):
        LegalUnit.model_validate({**faq_unit_data, "kind": kind})


@pytest.mark.parametrize(
    "field,value",
    [
        ("kind", "topic"),
        ("kind", "faq_entry"),
        ("role", "at_explanation"),
        ("faq", "record"),
    ],
)
def test_legal_units_cannot_acquire_isolated_faq_semantics(
    faq_unit_data, legacy_contracts, field, value
):
    data = legacy_contracts["unit"]["data"]
    data[field] = faq_unit_data["faq"] if field == "faq" else value
    with pytest.raises(ValidationError, match="FAQ metadata"):
        LegalUnit.model_validate(data)


def test_pdf_import_does_not_accept_a_collection_registry(faq_source, tmp_path):
    path = tmp_path / "source.pdf"
    path.write_bytes(b"%PDF-1.7\n%%EOF\n")
    source = SourceDefinition.model_validate(
        {**faq_source.model_dump(), "local_path": path}
    )
    with pytest.raises(ValueError, match="collection"):
        import_snapshot(source, tmp_path / "data")
