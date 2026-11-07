import pytest

from atrag.retrieval.query import (
    normalize_address,
    query_language,
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


@pytest.mark.parametrize(
    "query",
    [
        "Under CIRS 16, am I resident?",
        "Segundo o artigo 16 do CIRS, sou residente?",
        "Як CIRS 16 визначає резидентство?",
        "Как CIRS 16 определяет резидентство?",
        "CIRS 16 residência fiscal",
        "Explain artigo 16 do CIRS",
        "CIRS 16 and CIRS 15",
        "CIRS 16-A extra",
        "CIRS 160foo",
    ],
)
def test_article_mention_does_not_restrict_semantic_search(query):
    assert normalize_address(query) is None


@pytest.mark.parametrize("query", [" CIVA 53.?! ", "artigo 53.º do CIVA."])
def test_address_punctuation_and_language_use_the_same_route(query):
    assert normalize_address(query).article == "53"
    with pytest.raises(ValueError, match="query_language"):
        query_language(query, None)


def test_letter_suffix_is_preserved_and_unqualified_article_is_ambiguous():
    assert normalize_address("CIRS 68-A").article == "68-A"
    assert normalize_address("artigo 68.º-A CIRS").article == "68-A"
    assert normalize_address("artigo 1") is None
    assert normalize_address("Como se determina a residência fiscal?") is None


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
