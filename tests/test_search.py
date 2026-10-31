import pytest

from atrag.retrieval.query import (
    normalize_address,
    query_language,
    reciprocal_rank_fusion,
)


@pytest.mark.parametrize(
    "query", ["artigo 53.º CIVA", "art. 53 CIVA", "CIVA 53", "ARTIGO 53.º DO CIVA"]
)
def test_exact_address_normalization(query):
    address = normalize_address(query)
    assert (address.code, address.article, address.document_scope) == (
        "CIVA",
        "53",
        "code",
    )


def test_letter_suffix_is_preserved_and_unqualified_article_is_ambiguous():
    assert normalize_address("CIRS 68-A").article == "68-A"
    assert normalize_address("artigo 68.º-A CIRS").article == "68-A"
    assert normalize_address("artigo 1") is None
    assert normalize_address("Como se determina a residência fiscal?") is None


def test_rrf_combines_ranks_and_breaks_ties_deterministically():
    ranked = reciprocal_rank_fusion(["a", "b", "c"], ["b", "d", "a"], 60)
    assert [r.fragment_id for r in ranked] == ["b", "a", "d", "c"]
    assert ranked[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert (ranked[0].vector_rank, ranked[0].text_rank) == (2, 1)
    assert reciprocal_rank_fusion([], [], 60) == []


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Как определяется налоговое резидентство?", "ru"),
        ("How is tax residency determined?", "en"),
        ("Como se determina a residência fiscal?", "pt"),
    ],
)
def test_query_language_at_boundary(query, expected):
    assert query_language(query, None) == expected
    assert query_language(query, "en") == "en"


def test_short_address_requires_interface_language():
    with pytest.raises(ValueError, match="query_language"):
        query_language("CIRS 16", None)


def test_conflicting_filters_are_rejected():
    from atrag.retrieval.query import merge_filters
    from atrag.schemas.retrieval import SearchFilters

    with pytest.raises(ValueError, match="conflicts"):
        merge_filters(SearchFilters(code="CIRS"), normalize_address("CIVA 53"))
    merged = merge_filters(SearchFilters(paragraph="1"), normalize_address("CIRS 16"))
    assert merged == SearchFilters(
        code="CIRS", article="16", document_scope="code", paragraph="1"
    )
