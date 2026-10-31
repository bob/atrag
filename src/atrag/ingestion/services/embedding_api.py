"""One retry policy; the SDK's implicit retries are disabled by the caller."""

import random

from openai import APIConnectionError, APIError, APIStatusError

from atrag.schemas.ingestion.embeddings import APIUsage

MAX_ATTEMPTS = 3


def _temporary(error):
    if isinstance(error, APIConnectionError):
        return True
    if error.code == "insufficient_quota":
        return False
    return error.status_code in (408, 409, 429) or error.status_code >= 500


def request_embeddings(client, space, batch, report, sleep):
    if client is None:
        raise ValueError("OPENAI_API_KEY is required for uncached embeddings")
    for attempt in range(MAX_ATTEMPTS):
        report["requests"] += 1
        try:
            response = client.embeddings.create(
                model=space.model,
                dimensions=space.dimensions,
                input=[item.text for item in batch],
                encoding_format="float",
            )
            break
        except (APIConnectionError, APIStatusError) as error:
            if not _temporary(error) or attempt == MAX_ATTEMPTS - 1:
                raise ValueError("Embedding request failed; see run report") from None
            report["retries"] += 1
            sleep(2**attempt + random.random())
        except APIError:
            raise ValueError("Embedding request failed; invalid API response") from None
    raw = response.model_dump()
    usage = APIUsage.model_validate(raw.get("usage"))
    report["usage"]["prompt_tokens"] += usage.prompt_tokens
    report["usage"]["total_tokens"] += usage.total_tokens
    return raw
