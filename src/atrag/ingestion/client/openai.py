"""Create the OpenAI client from application settings."""

from openai import OpenAI

from atrag.schemas.settings import Settings


def create_openai_client(settings: Settings) -> OpenAI | None:
    if settings.openai_api_key is None:
        return None
    return OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        base_url="https://api.openai.com/v1",
        max_retries=0,
        timeout=30,
    )
