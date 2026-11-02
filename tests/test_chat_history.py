"""History is bounded by dropping whole completed turns, never tool pairs."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from atrag.chat.api.history import HistoryBudgetExceeded, bounded_messages, count_tokens


def exchange(question="old question"):
    return [
        HumanMessage(question),
        AIMessage(
            "",
            tool_calls=[
                {
                    "name": "search_tax_documents",
                    "args": {"query": "CIRS 16"},
                    "id": "c",
                }
            ],
        ),
        ToolMessage("evidence", tool_call_id="c"),
        AIMessage("Old answer [1]"),
    ]


def test_multilingual_message_budget_includes_metadata_and_framing():
    # Fixed fixture for the locked tokenizer/message format, independent of the
    # boundary calculations that exercise bounded_messages below.
    assert count_tokens([HumanMessage("Olá, мир?"), AIMessage("Answer")]) == 93


def test_trimming_removes_a_whole_turn_and_keeps_original_history():
    old = exchange()
    current = [HumanMessage("current")]
    messages = old + current
    budget = count_tokens(messages) - 1
    assert bounded_messages(messages, max_tokens=budget) == current
    assert messages == old + current
    assert bounded_messages(messages, max_tokens=count_tokens(messages)) == messages


def test_newest_completed_turns_are_retained_as_a_suffix():
    old, recent, current = (
        exchange("x " * 100),
        exchange("recent"),
        [HumanMessage("now")],
    )
    budget = count_tokens(recent + current) + 7
    assert (
        bounded_messages(old + recent + current, fixed_tokens=7, max_tokens=budget)
        == recent + current
    )
    with pytest.raises(HistoryBudgetExceeded):
        bounded_messages(current, max_tokens=count_tokens(current) - 1)
    with pytest.raises(HistoryBudgetExceeded):
        bounded_messages(current, fixed_tokens=1, max_tokens=count_tokens(current))
    assert bounded_messages(current, max_tokens=count_tokens(current)) == current


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [AIMessage("no question")],
        [HumanMessage("Q"), ToolMessage("orphan", tool_call_id="x")],
        [
            HumanMessage("Q"),
            AIMessage("", tool_calls=[{"name": "search", "args": {}, "id": "c"}]),
        ],
        [HumanMessage("unfinished"), HumanMessage("next")],
        [
            HumanMessage("Q"),
            AIMessage("", tool_calls=[{"name": "search", "args": {}, "id": "c"}]),
            AIMessage("missing result"),
        ],
        exchange()[:-1] + [HumanMessage("next")],
        [
            HumanMessage("Q"),
            AIMessage(
                "",
                tool_calls=[
                    {"name": "search", "args": {}, "id": "c"},
                    {"name": "search", "args": {}, "id": "c"},
                ],
            ),
            ToolMessage("duplicate", tool_call_id="c"),
        ],
    ],
)
def test_malformed_or_incomplete_history_is_not_sent(messages):
    with pytest.raises(ValueError):
        bounded_messages(messages)
