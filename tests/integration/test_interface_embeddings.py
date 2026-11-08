"""The UI's service factory must supply embeddings to semantic chat search."""

import json
from uuid import uuid4

import httpx
import pytest
import streamlit as st
import test_chat_turns
from fixtures.chat import ScriptedModel, call
from langchain_core.messages import AIMessage

from atrag.chat.api.execution import LangGraphBackend
from atrag.interface import resources
from atrag.schemas.settings import Settings

pytestmark = pytest.mark.integration
pytest_plugins = ["test_chat_tools"]
clean_checkpoints = test_chat_turns.clean_checkpoints


def test_ui_factory_completes_semantic_search(chat, materials, engine, monkeypatch):
    requests = []

    def handle(transport, request):
        requests.append(request)
        payload = json.loads(request.content)
        if request.url.path == "/v1/responses":
            count = json.loads(payload["input"])["count"]
            return httpx.Response(
                200,
                json={
                    "id": "resp-reranking",
                    "object": "response",
                    "created_at": 0,
                    "status": "completed",
                    "model": payload["model"],
                    "output": [
                        {
                            "id": "msg-reranking",
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": json.dumps({"ids": list(range(count))}),
                                    "annotations": [],
                                }
                            ],
                        }
                    ],
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "total_tokens": 25,
                    },
                },
            )
        return httpx.Response(
            200,
            json={
                "model": payload["model"],
                "data": [{"index": 0, "embedding": [0.6, 0.8] + [0.0] * 1534}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    def script():
        yield call(args={"query": "residência", "limit": 2})
        source = chat[0].load_history(chat[1], chat[2])[-1].sources[0]
        yield AIMessage(content=f"Resposta com fonte [{source['citation_key']}].")

    model = ScriptedModel(messages=script(), disable_streaming=True)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle)
    monkeypatch.setattr(resources, "get_engine", lambda: engine)
    monkeypatch.setattr(resources, "get_model", lambda: model)
    monkeypatch.setattr(
        resources, "load_settings", lambda: Settings(openai_api_key="test-key")
    )
    st.cache_resource.clear()
    try:
        turns = resources.get_turn_service()
        _, user, conversation = chat
        events = list(
            turns.stream_turn(user, conversation, uuid4(), "Residência?", "pt")
        )
        assert events[-1].kind == "completed"
        turn = events[-1].turn
        assert turn.sources
        assert turn.embedding_usage["requests"] == 1
        assert turn.embedding_usage["usage"]["total_tokens"] == 2
        assert len(requests) == 2
        assert str(requests[0].url) == "https://api.openai.com/v1/embeddings"
        assert requests[0].headers["authorization"] == "Bearer test-key"
        assert json.loads(requests[0].content)["input"] == ["residência"]
        assert str(requests[1].url) == "https://api.openai.com/v1/responses"
        assert json.loads(requests[1].content)["store"] is False
        assert turn.embedding_usage["reranking"]["total_tokens"] == 25
        cached = resources.get_turn_service().backend
        assert isinstance(cached, LangGraphBackend)
        assert isinstance(turns.backend, LangGraphBackend)
        assert cached.client is turns.backend.client
    finally:
        st.cache_resource.clear()
