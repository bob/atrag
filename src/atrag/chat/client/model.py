"""Configure the chat transport once; SDK retries cover temporary errors only."""

from langchain_openai import ChatOpenAI

from atrag.schemas.settings import Settings


def create_chat_model(settings: Settings, *, http_client=None) -> ChatOpenAI:
    if settings.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY is required for the chat model")
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url="https://api.openai.com/v1",
        use_responses_api=True,
        use_previous_response_id=False,
        reasoning_effort="low",
        max_completion_tokens=4096,
        timeout=60,
        max_retries=2,
        http_client=http_client,
        store=False,
    )
