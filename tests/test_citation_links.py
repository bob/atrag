"""Official links use the saved location, never the catalogue or model text."""

from pathlib import Path

import pytest
from pydantic import HttpUrl

from atrag.chat.core.sources import citation_url
from atrag.ingestion.services.sources import load_source_registry


@pytest.mark.parametrize(
    "registry,locations,expected",
    [
        ("cirs", [{"kind": "pdf", "page": 47}], "CIRS.pdf#page=47"),
        ("cirs", [], "CIRS.pdf"),
        (
            "faq",
            [
                {
                    "kind": "html",
                    "page_url": "https://example.org/topic",
                    "anchor": "heading20",
                }
            ],
            "https://example.org/topic#heading20",
        ),
        (
            "faq",
            [
                {
                    "kind": "html",
                    "page_url": "https://example.org/topic",
                    "anchor": "a b/#",
                }
            ],
            "https://example.org/topic#a%20b%2F%23",
        ),
        (
            "faq",
            [{"kind": "html", "page_url": "https://example.org/topic"}],
            "https://example.org/topic",
        ),
        ("faq", [{"kind": "html", "anchor": "heading20"}], None),
        ("faq", [], None),
    ],
)
def test_citation_url_uses_only_the_saved_source_and_location(
    registry, locations, expected
):
    source = load_source_registry(Path(f"data/sources/{registry}.json"))
    result = citation_url(source, locations)
    if expected is None:
        assert result is None
    else:
        assert result is not None and result.endswith(expected)


def test_absent_source_url_does_not_invent_a_pdf_link():
    source = load_source_registry(Path("data/sources/cirs.json"))
    assert (
        citation_url(
            source.model_copy(update={"source_url": None}), [{"kind": "pdf", "page": 1}]
        )
        is None
    )


def test_pdf_link_replaces_an_existing_fragment_and_keeps_the_query():
    source = load_source_registry(Path("data/sources/cirs.json")).model_copy(
        update={"source_url": HttpUrl("https://example.org/CIRS.pdf?download=1#old")}
    )
    assert (
        citation_url(source, [{"kind": "pdf", "page": 48}])
        == "https://example.org/CIRS.pdf?download=1#page=48"
    )
