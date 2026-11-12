"""Accumulate embeddings and preserve usage saved by earlier retrieval versions."""

from atrag.chat.core.usage import add_usage, new_usage


def test_legacy_reranking_usage_survives_later_tools():
    first = new_usage()
    first["reranking"] = {
        "requests": 1,
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
    }
    saved = add_usage({}, first)
    assert saved["requests"] == saved["retries"] == 0
    assert saved["usage"] == {"prompt_tokens": 0, "total_tokens": 0}
    assert add_usage(saved, new_usage())["reranking"] == first["reranking"]
    assert add_usage(saved, first)["reranking"] == {
        "requests": 2,
        "input_tokens": 40,
        "output_tokens": 10,
        "total_tokens": 50,
    }


def test_embedding_counters_accumulate_requests_retries_and_tokens():
    previous = {
        "requests": 2,
        "retries": 1,
        "usage": {"prompt_tokens": 7, "total_tokens": 7},
    }
    current = {
        "requests": 1,
        "retries": 2,
        "usage": {"prompt_tokens": 3, "total_tokens": 3},
    }
    assert add_usage(previous, current) == {
        "requests": 3,
        "retries": 3,
        "usage": {"prompt_tokens": 10, "total_tokens": 10},
    }
