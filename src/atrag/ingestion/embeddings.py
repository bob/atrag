"""Pure embedding preparation and response matching, shared with future queries."""

import hashlib
from collections.abc import Callable, Iterable, Iterator

from atrag.schemas.ingestion.embeddings import (
    APIResponse,
    EmbeddingInput,
    EmbeddingRecord,
    EmbeddingSpace,
)

MAX_INPUT_TOKENS = 8192
MAX_BATCH_INPUTS = 2048
MAX_BATCH_TOKENS = 300_000


def prepare_input(text: str, count_tokens: Callable[[str], int]) -> EmbeddingInput:
    tokens = count_tokens(text)
    if not text.strip() or tokens > MAX_INPUT_TOKENS:
        raise ValueError("Embedding input must be nonempty and within the token limit")
    return EmbeddingInput(
        hashlib.sha256(text.encode("utf-8")).hexdigest(), text, tokens
    )


def input_batches(
    inputs: Iterable[EmbeddingInput], batch_size: int
) -> Iterator[list[EmbeddingInput]]:
    if not 1 <= batch_size <= MAX_BATCH_INPUTS:
        raise ValueError("Embedding batch size must be between 1 and 2048")
    batch, tokens = [], 0
    for item in inputs:
        if len(batch) == batch_size or tokens + item.token_count > MAX_BATCH_TOKENS:
            yield batch
            batch, tokens = [], 0
        batch.append(item)
        tokens += item.token_count
    if batch:
        yield batch


def match_response(
    raw: dict, space: EmbeddingSpace, inputs: list[EmbeddingInput]
) -> list[EmbeddingRecord]:
    response = APIResponse.model_validate(raw)
    if response.model != space.model:
        raise ValueError("Embedding response belongs to another model")
    by_index = {row.index: row for row in response.data}
    if len(response.data) != len(inputs) or set(by_index) != set(range(len(inputs))):
        raise ValueError("Embedding response indices do not match the inputs")
    return [
        EmbeddingRecord(
            **space.model_dump(),
            input_hash=item.input_hash,
            embedding=by_index[i].embedding,
        )
        for i, item in enumerate(inputs)
    ]
