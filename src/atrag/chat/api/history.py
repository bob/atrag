"""Bound model input without editing the full checkpoint transcript."""

import json
from collections.abc import Sequence

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class HistoryBudgetExceeded(ValueError):
    """The current turn cannot fit; silently cutting its evidence is unsafe."""


def count_tokens(messages: Sequence[BaseMessage]) -> int:
    """Conservative local count, including message metadata and tool arguments."""
    encoding = tiktoken.get_encoding("o200k_base")
    return (
        sum(
            len(
                encoding.encode_ordinary(
                    json.dumps(m.model_dump(mode="json"), ensure_ascii=False)
                )
            )
            + 4
            for m in messages
        )
        + 2
    )


def _pending_calls(message):
    calls = {call["id"] for call in message.tool_calls}
    if len(calls) != len(message.tool_calls) or message.invalid_tool_calls:
        raise ValueError("Invalid tool calls in history")
    return calls


def _check_tool_pairs(messages):
    pending = set()
    for message in messages:
        if isinstance(message, ToolMessage):
            if message.tool_call_id not in pending:
                raise ValueError("Orphan tool result in history")
            pending.remove(message.tool_call_id)
        else:
            if pending:
                raise ValueError("Missing tool results in history")
            if isinstance(message, AIMessage):
                pending = _pending_calls(message)
    if pending:
        raise ValueError("Missing tool results in history")


def _turns(messages):
    if not messages or not isinstance(messages[0], HumanMessage):
        raise ValueError("History must start with a question")
    starts = [
        i for i, message in enumerate(messages) if isinstance(message, HumanMessage)
    ]
    groups = [
        messages[start:end] for start, end in zip(starts, starts[1:] + [len(messages)])
    ]
    for group in groups[:-1]:
        if not isinstance(group[-1], AIMessage) or group[-1].tool_calls:
            raise ValueError("Only completed older turns may enter model history")
    _check_tool_pairs(messages)
    return groups


def bounded_messages(messages, *, fixed_tokens=0, max_tokens=32000):
    groups = _turns(messages)
    retained = list(messages)
    for group in groups[:-1]:
        if fixed_tokens + count_tokens(retained) <= max_tokens:
            return retained
        retained = retained[len(group) :]
    if fixed_tokens + count_tokens(retained) > max_tokens:
        raise HistoryBudgetExceeded("Current turn exceeds the model input budget")
    return retained
