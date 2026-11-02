"""References select original registry entries in order of first use."""

from atrag.chat.core.answers import answer_sources


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
