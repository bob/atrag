"""References select original registry entries in order of first use."""

import pytest

from atrag.chat.core.answers import answer_sources


def test_named_evidence_keys_resolve_to_the_original_numbers():
    from atrag.chat.core.answers import render_citations

    registry = [
        {"citation_id": 1, "citation_key": "source:1:faq:FAQ AT 3286"},
        {"citation_id": 2, "citation_key": "source:2:law:CIRS 16"},
    ]
    answer = render_citations(
        "Law [source:2:law:CIRS 16]; AT [source:1:faq:FAQ AT 3286].", registry
    )
    assert answer == "Law [2]; AT [1]."
    assert answer_sources(answer, registry) == [registry[1], registry[0]]


def test_model_cannot_bypass_keys_by_assigning_a_numeric_citation():
    from atrag.chat.core.answers import InvalidAnswer, render_citations

    with pytest.raises(InvalidAnswer):
        render_citations(
            "Law [1]", [{"citation_id": 1, "citation_key": "source:1:faq:FAQ 3286"}]
        )


@pytest.mark.parametrize(
    "text", ["Wrong [source:1:law:CIRS 16]", "Cut [source:2:law:CIRS 16"]
)
def test_invented_or_incomplete_evidence_keys_are_rejected(text):
    from atrag.chat.core.answers import InvalidAnswer, render_citations

    with pytest.raises(InvalidAnswer):
        render_citations(
            text, [{"citation_id": 2, "citation_key": "source:2:law:CIRS 16"}]
        )


def test_stream_buffers_a_split_key_without_exposing_it():
    from atrag.chat.core.answers import CitationStream

    stream = CitationStream(
        lambda: [{"citation_id": 2, "citation_key": "source:2:law:CIRS 16"}]
    )
    assert stream.feed("Ready ") == "Ready "
    assert stream.feed("Law [sou") == "Law "
    assert stream.feed("rce:2:law:CIRS 16") == ""
    assert stream.feed("] and [ordinary] text") == "[2] and [ordinary] text"
    assert stream.finish() == ""


def test_stream_preserves_ordinary_unclosed_brackets_but_rejects_cut_keys():
    from atrag.chat.core.answers import CitationStream, InvalidAnswer

    stream = CitationStream(list)
    assert stream.feed("Text [draft") == "Text "
    assert stream.finish() == "[draft"
    stream.feed("[source:cut")
    with pytest.raises(InvalidAnswer):
        stream.finish()


def test_sources_keep_original_metadata_and_exclude_uncited_evidence():
    registry = [
        {
            "citation_id": 1,
            "text_original": "Um",
            "locations": [{"kind": "html", "anchor": "one"}],
        },
        {"citation_id": 2, "text_original": "Dois", "issues": ["partial"]},
        {"citation_id": 3, "text_original": "Unused"},
    ]
    assert answer_sources("Text [2] [1] [2]", registry) == [registry[1], registry[0]]
    assert len(registry) == 3
