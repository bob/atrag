"""Retrieval usage counters; embeddings and reranking are accounted separately."""


def new_usage():
    return {
        "requests": 0,
        "retries": 0,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def add_usage(previous, current):
    result = {
        "requests": previous.get("requests", 0) + current["requests"],
        "retries": previous.get("retries", 0) + current["retries"],
        "usage": {
            key: previous.get("usage", {}).get(key, 0) + value
            for key, value in current["usage"].items()
        },
    }
    if "reranking" in previous or "reranking" in current:
        result["reranking"] = {
            key: previous.get("reranking", {}).get(key, 0)
            + current.get("reranking", {}).get(key, 0)
            for key in ("requests", "input_tokens", "output_tokens", "total_tokens")
        }
    return result
