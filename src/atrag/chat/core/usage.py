"""Embedding usage counters independent of transport and storage."""


def new_usage():
    return {
        "requests": 0,
        "retries": 0,
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def add_usage(previous, current):
    return {
        "requests": previous.get("requests", 0) + current["requests"],
        "retries": previous.get("retries", 0) + current["retries"],
        "usage": {
            key: previous.get("usage", {}).get(key, 0) + value
            for key, value in current["usage"].items()
        },
    }
