"""Independent dense and semantic ranks select existing server-owned evidence."""

import json
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError

from atrag.retrieval.reranking import RerankingRequestError, rerank_hits


def hit(index):
    return SimpleNamespace(
        fragment=SimpleNamespace(
            citation_label=f"Source {index}",
            embedding_text=f"Evidence {index}",
            source=SimpleNamespace(document_title="A document"),
            role="code",
        )
    )


def response(ids, *, status="completed"):
    return SimpleNamespace(
        status=status,
        output_text=json.dumps({"ids": ids}),
        usage=SimpleNamespace(input_tokens=20, output_tokens=5, total_tokens=25),
    )


def test_dense_and_semantic_ranks_are_combined_without_changing_evidence():
    evidence = [hit(i) for i in range(5)]
    calls = []

    def create(**request):
        calls.append(request)
        return response([4, 1, 0, 2])

    usage = {}
    question = "Quem é residente?"
    result = rerank_hits(
        question,
        evidence,
        2,
        client=SimpleNamespace(responses=SimpleNamespace(create=create)),
        usage=usage,
    )
    assert result == [evidence[i] for i in [0, 4, 1, 2, 3]]
    sent = json.loads(calls[0]["input"])
    assert sent["query"] == question and sent["count"] == 4
    assert question in calls[0]["input"]
    assert [p["text"] for p in sent["candidates"]] == [
        f"Evidence {i}" for i in range(5)
    ]
    assert calls[0]["store"] is False
    assert calls[0]["text"]["format"]["strict"] is True
    ids = calls[0]["text"]["format"]["schema"]["properties"]["ids"]
    assert ids["minItems"] == ids["maxItems"] == 4
    assert ids["items"]["maximum"] == 4
    assert usage["reranking"] == {
        "requests": 1,
        "input_tokens": 20,
        "output_tokens": 5,
        "total_tokens": 25,
    }


def test_agreement_preserves_dense_order():
    evidence = [hit(i) for i in range(5)]
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: response([0, 1, 2, 3]))
    )
    assert rerank_hits("Question?", evidence, 2, client=client, usage={}) == evidence


def test_no_request_when_all_evidence_fits_the_limit():
    evidence = [hit(0)]
    assert rerank_hits("Question?", evidence, 1, client=None, usage={}) == evidence
    assert rerank_hits("Question?", [], 5, client=None, usage={}) == []


@pytest.mark.parametrize(
    "ids", [[0, 0], [0, 3], [-1, 0], [True, 0], [0], [], [0, 1, 2]]
)
def test_invalid_selections_are_operational_errors(ids):
    client = SimpleNamespace(
        responses=SimpleNamespace(create=lambda **_: response(ids))
    )
    with pytest.raises(RerankingRequestError):
        rerank_hits("Question?", [hit(i) for i in range(3)], 1, client=client, usage={})


def test_incomplete_response_is_not_silently_replaced_by_cosine():
    client = SimpleNamespace(
        responses=SimpleNamespace(
            create=lambda **_: response([0, 1], status="incomplete")
        )
    )
    with pytest.raises(RerankingRequestError):
        rerank_hits("Question?", [hit(i) for i in range(3)], 1, client=client, usage={})


def test_response_without_usage_is_an_operational_error():
    result = response([0, 1])
    result.usage = None
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **_: result))
    with pytest.raises(RerankingRequestError):
        rerank_hits("Question?", [hit(i) for i in range(3)], 1, client=client, usage={})


def test_transport_failure_is_an_operational_error():
    def create(**_):
        raise APIConnectionError(
            request=httpx.Request("POST", "https://api.openai.com")
        )

    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    with pytest.raises(RerankingRequestError):
        rerank_hits("Question?", [hit(i) for i in range(3)], 1, client=client, usage={})


def test_reranking_usage_survives_later_tools_without_model_calls():
    from atrag.chat.core.usage import add_usage, new_usage

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


def test_embedding_counters_remain_independent_from_reranking():
    from atrag.chat.core.usage import add_usage

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
