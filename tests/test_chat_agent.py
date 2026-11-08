"""The real create_agent graph with a scripted, tool-capable model."""

import json
from types import SimpleNamespace

import pytest
from fixtures.chat import ScriptedModel, call
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool

from atrag.chat.api.agent import AgentLimitExceeded, build_agent
from atrag.chat.core.answers import InvalidAnswer


@pytest.fixture
def rag():
    registry, calls = [], []

    def search(query: str):
        calls.append(query)
        if not registry:
            registry.append({"citation_id": 1, "text_original": "São residentes."})
        return {"status": "ok", "sources": registry}

    tool = StructuredTool.from_function(
        search, name="search_tax_documents", description="Search tax documents"
    )
    return SimpleNamespace(
        tools=(tool,),
        service=SimpleNamespace(sources=lambda: list(registry)),
        calls=calls,
    )


def test_graph_searches_and_returns_only_registered_sources(rag):
    model = ScriptedModel(messages=iter([call(), AIMessage("Residência [1].")]))
    graph = build_agent(model, rag, "pt")
    result = graph.invoke({"messages": [HumanMessage("Residência?")]})
    assert result["answer"] == "Residência [1]."
    assert result["sources"] == rag.service.sources()
    assert result["model_usage"] == {
        "requests": 2,
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }
    assert rag.calls == ["CIRS 16"]
    assert isinstance(model.seen[1][-1], ToolMessage)
    assert (
        json.loads(model.seen[1][-1].content)["sources"][0]["text_original"]
        == "São residentes."
    )


@pytest.mark.parametrize(
    "language,name",
    [("en", "English"), ("pt", "Português"), ("uk", "Українська"), ("ru", "Русский")],
)
def test_model_receives_an_unambiguous_language_name(rag, language, name):
    model = ScriptedModel(messages=iter([AIMessage("No evidence")]))
    build_agent(model, rag, language).invoke({"messages": [HumanMessage("Question")]})
    assert f"Answer entirely in {name}." in model.seen[0][0].text


@pytest.mark.parametrize(
    "answer",
    [
        "Invented [2]",
        "Bad [0]",
        "Range [1, 2]",
        "Zero [01]",
        "Missing reference",
        "",
        "   ",
    ],
)
def test_invalid_answer_never_returns_success(rag, answer):
    model = ScriptedModel(messages=iter([call(), AIMessage(answer)]))
    with pytest.raises(InvalidAnswer):
        build_agent(model, rag, "en").invoke({"messages": [HumanMessage("Question")]})


def test_empty_evidence_allows_a_clear_gap_without_citations(rag):
    model = ScriptedModel(
        messages=iter([AIMessage("There is no evidence; which tax year?")])
    )
    result = build_agent(model, rag, "en").invoke(
        {"messages": [HumanMessage("Unknown tax?")]}
    )
    assert result["sources"] == []
    assert result["answer"] == "There is no evidence; which tax year?"


def test_unknown_citation_is_rejected_even_without_tools(rag):
    model = ScriptedModel(messages=iter([AIMessage("Claim [1]")]))
    with pytest.raises(InvalidAnswer):
        build_agent(model, rag, "ru").invoke({"messages": [HumanMessage("Вопрос")]})


def test_answer_repair_uses_feedback_and_preserves_the_original_question(rag):
    model = ScriptedModel(
        messages=iter([call(), AIMessage("Missing reference"), AIMessage("Answer [1]")])
    )
    result = build_agent(model, rag, "en", repair_answers=True).invoke(
        {"messages": [HumanMessage("Question")]}
    )
    assert result["answer"] == "Answer [1]"
    assert result["sources"] == rag.service.sources()
    assert result["model_usage"]["requests"] == 3
    assert sum(isinstance(m, HumanMessage) for m in result["messages"]) == 1
    assert "Answer omits citations" in model.seen[-1][-1].text


def test_answer_repair_is_bounded_and_never_accepts_an_invalid_second_draft(rag):
    model = ScriptedModel(
        messages=iter(
            [call(), AIMessage("Missing reference"), AIMessage("Still missing")]
        )
    )
    with pytest.raises(InvalidAnswer, match="omits citations"):
        build_agent(model, rag, "en", repair_answers=True).invoke(
            {"messages": [HumanMessage("Question")]}
        )
    assert len(model.seen) == 3


def test_tool_limit_stops_the_graph(rag):
    model = ScriptedModel(messages=iter([call(identifier=str(i)) for i in range(7)]))
    with pytest.raises(AgentLimitExceeded, match="Tool"):
        build_agent(model, rag, "en").invoke({"messages": [HumanMessage("Loop")]})
    assert len(model.seen) == 7
    assert len(rag.calls) == 6


def test_model_limit_stops_before_the_ninth_call(rag, monkeypatch):
    # Isolate the model gate: normally the stricter tool gate stops this loop first.
    monkeypatch.setattr("atrag.chat.api.agent.TOOL_CALL_LIMIT", 10)
    model = ScriptedModel(messages=iter([call(identifier=str(i)) for i in range(9)]))
    graph = build_agent(model, rag, "en")
    with pytest.raises(AgentLimitExceeded, match="Model"):
        graph.invoke({"messages": [HumanMessage("Loop")]})
    assert len(model.seen) == 8


def test_parallel_tool_batch_is_checked_before_any_tool_runs(rag):
    batch = call()
    batch.tool_calls = [call(identifier=str(i)).tool_calls[0] for i in range(7)]
    with pytest.raises(AgentLimitExceeded, match="Tool"):
        build_agent(ScriptedModel(messages=iter([batch])), rag, "en").invoke(
            {"messages": [HumanMessage("Q")]}
        )
    assert rag.calls == []


def test_model_errors_propagate_without_a_success_result(rag):
    def fail():
        raise TimeoutError("scripted timeout")
        yield

    with pytest.raises(TimeoutError, match="scripted timeout"):
        build_agent(ScriptedModel(messages=fail()), rag, "en").invoke(
            {"messages": [HumanMessage("Q")]}
        )


@pytest.mark.parametrize(
    "metadata",
    [
        {"finish_reason": "length"},
        {"finish_reason": "content_filter"},
        {"status": "incomplete"},
        {"status": "failed"},
    ],
)
def test_truncated_or_failed_response_is_not_a_completed_answer(rag, metadata):
    model = ScriptedModel(
        messages=iter([AIMessage("Partial answer", response_metadata=metadata)])
    )
    with pytest.raises(InvalidAnswer):
        build_agent(model, rag, "en").invoke({"messages": [HumanMessage("Q")]})


def test_bounded_model_input_preserves_the_full_checkpoint_and_tool_pairs(rag):
    from langgraph.checkpoint.memory import InMemorySaver

    from atrag.chat.api.history import HistoryBudgetExceeded, count_tokens

    old = [HumanMessage("old " * 18000), AIMessage("Older answer")]
    recent = [
        HumanMessage("recent " * 15000),
        call(),
        ToolMessage("source", tool_call_id="call-1"),
        AIMessage("Recent answer"),
    ]
    current = HumanMessage("Current question")
    model = ScriptedModel(messages=iter([AIMessage("Need more evidence.")]))
    graph = build_agent(model, rag, "en", checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "bounded"}}
    result = graph.invoke({"messages": old + recent + [current]}, config)
    assert len(model.seen[0]) == len(recent) + 2
    assert model.seen[0][1:] == recent + [current]
    assert count_tokens(model.seen[0]) <= 32000
    assert result["messages"] == old + recent + [current, result["messages"][-1]]
    assert len(graph.get_state(config).values["messages"]) == 8
    model = ScriptedModel(messages=iter([]))
    with pytest.raises(HistoryBudgetExceeded):
        build_agent(model, rag, "en").invoke(
            {"messages": [HumanMessage("oversized " * 32000)]}
        )
    assert model.seen == []


def test_only_public_text_and_cited_evidence_leave_the_agent(rag):
    model = ScriptedModel(
        messages=iter(
            [
                call(),
                AIMessage(
                    content=[
                        {"type": "reasoning", "reasoning": "private reasoning"},
                        {"type": "text", "text": "Answer [1] [1]."},
                    ]
                ),
            ]
        )
    )
    result = build_agent(model, rag, "en").invoke({"messages": [HumanMessage("Q")]})
    assert result["answer"] == "Answer [1] [1]."
    assert len(result["sources"]) == 1


def test_unfinished_tool_request_cannot_be_a_final_answer(rag):
    model = ScriptedModel(messages=iter([call()]))
    rag.tools = ()
    with pytest.raises(InvalidAnswer):
        build_agent(model, rag, "en").invoke({"messages": [HumanMessage("Q")]})


def test_tool_schemas_share_the_input_budget_with_the_question(rag):
    from atrag.chat.api.history import HistoryBudgetExceeded

    rag.tools[0].description = "schema " * 6000
    model = ScriptedModel(messages=iter([AIMessage("Gap")]))
    with pytest.raises(HistoryBudgetExceeded):
        build_agent(model, rag, "en").invoke(
            {"messages": [HumanMessage("question " * 28000)]}
        )
    assert model.seen == []


def test_invalid_tool_json_is_not_a_final_answer(rag):
    model = ScriptedModel(
        messages=iter(
            [
                AIMessage(
                    "Answer",
                    invalid_tool_calls=[
                        {
                            "name": "search_tax_documents",
                            "args": "{",
                            "id": "bad",
                            "error": "Invalid JSON",
                        }
                    ],
                )
            ]
        )
    )
    with pytest.raises(InvalidAnswer):
        build_agent(model, rag, "en").invoke({"messages": [HumanMessage("Q")]})
