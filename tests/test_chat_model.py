"""Exercise Responses serialization and SDK retry policy without paid calls."""

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage
from openai import APIStatusError, APITimeoutError

from atrag.chat.client.model import create_chat_model
from atrag.schemas.settings import Settings


def response_payload(text="No evidence for this tax year."):
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    }


def test_responses_request_uses_configured_model_and_limits():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(200, json=response_payload())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        model = create_chat_model(
            Settings(openai_api_key="test-key", chat_model="configured-model"),
            http_client=client,
        )
        answer = model.invoke("Question")
    assert answer.text == "No evidence for this tax year."
    assert answer.usage_metadata == {
        "input_tokens": 3,
        "output_tokens": 2,
        "total_tokens": 5,
        "input_token_details": {},
        "output_token_details": {},
    }
    assert str(requests[0].url) == "https://api.openai.com/v1/responses"
    assert requests[0].extensions["timeout"] == {
        "connect": 60,
        "read": 60,
        "write": 60,
        "pool": 60,
    }
    payload = json.loads(requests[0].content)
    assert payload["model"] == "configured-model"
    assert payload["reasoning"] == {"effort": "low"}
    assert payload["max_output_tokens"] == 4096
    assert payload["store"] is False
    assert "previous_response_id" not in payload


@pytest.mark.parametrize(
    "failure,expected_attempts",
    [(429, 3), (500, 3), (400, 1), (401, 1), ("timeout", 3)],
)
def test_temporary_errors_have_at_most_two_retries(
    monkeypatch, failure, expected_attempts
):
    requests = []
    monkeypatch.setattr(
        "openai._base_client.BaseClient._calculate_retry_timeout", lambda *a, **kw: 0
    )

    def handle(request):
        requests.append(request)
        if failure == "timeout":
            raise httpx.ReadTimeout("scripted", request=request)
        return httpx.Response(
            failure, json={"error": {"message": "scripted", "type": "test_error"}}
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        model = create_chat_model(
            Settings(openai_api_key="test-key"), http_client=client
        )
        with pytest.raises(APITimeoutError if failure == "timeout" else APIStatusError):
            model.invoke("Question")
    assert len(requests) == expected_attempts


def test_missing_key_is_reported_before_constructing_a_model():
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        create_chat_model(Settings())


def test_followup_sends_local_history_without_a_server_side_response_pointer():
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json=response_payload())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        model = create_chat_model(
            Settings(openai_api_key="test-key"), http_client=client
        )
        question = HumanMessage("Original question")
        answer = model.invoke([question])
        model.invoke([question, answer, HumanMessage("Followup question")])
    assert "previous_response_id" not in payloads[1]
    assert len(payloads[1]["input"]) == 3
    assert payloads[1]["input"][0]["content"] == "Original question"
