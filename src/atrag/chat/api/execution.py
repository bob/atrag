"""Adapt checkpointed LangGraph streams to public text and tool activity."""

from contextlib import contextmanager

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.postgres import PostgresSaver

from atrag.chat.api.agent import _turn_usage, build_agent
from atrag.chat.api.tools import RagTools
from atrag.chat.core.answers import CitationStream
from atrag.chat.db.execution import checkpoint_connection
from atrag.schemas.chat.execution import ExecutionState, TextDelta, ToolActivity


def _activities(update):
    for value in update.values():
        if value is None:
            continue
        for message in value.get("messages", []):
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    yield ToolActivity(call["name"], call["id"], "started")
            elif isinstance(message, ToolMessage):
                yield ToolActivity(
                    message.name or "", message.tool_call_id, "completed"
                )


class LangGraphSession:
    def __init__(self, graph, conversation_id, turn_id, *, sources=list):
        self.graph = graph
        self.turn_id = str(turn_id)
        self.config = {"configurable": {"thread_id": str(conversation_id)}}
        self.citations = CitationStream(sources)

    def state(self):
        snapshot = self.graph.get_state(self.config)
        values = snapshot.values
        if values.get("turn_id") != self.turn_id:
            return ExecutionState(
                values.get("turn_id"), bool(snapshot.next), "", "", {}
            )
        messages = values.get("messages", [])
        latest = messages[-1] if messages else None
        return ExecutionState(
            turn_id=values.get("turn_id"),
            pending=bool(snapshot.next),
            answer=values.get("answer", ""),
            text=latest.text if isinstance(latest, AIMessage) else "",
            model_usage=_turn_usage(messages) if messages else {},
        )

    def stream(self, turn):
        saved = self.state()
        if saved.turn_id == str(turn.id):
            payload = None
        else:
            if saved.pending:
                raise RuntimeError("Another turn has unfinished checkpoints")
            payload = {
                "turn_id": str(turn.id),
                "messages": [HumanMessage(content=turn.question, id=str(turn.id))],
                "answer": "",
                "sources": [],
                "model_usage": {},
            }
        for mode, data in self.graph.stream(
            payload, self.config, stream_mode=["messages", "updates"], durability="sync"
        ):
            if mode == "messages":
                message, metadata = data
                if (
                    metadata.get("langgraph_node") == "model"
                    and message.text
                    and (text := self.citations.feed(message.text))
                ):
                    yield TextDelta(text)
            else:
                yield from _activities(data)
        if text := self.citations.finish():
            yield TextDelta(text)


class LangGraphBackend:
    """Shared model/client, fresh graph and PostgreSQL session for each execution."""

    def __init__(self, engine, model, *, client=None):
        self.engine, self.model, self.client = engine, model, client

    @contextmanager
    def open(self, user, conversation_id, turn):
        rag = RagTools(self.engine, user, conversation_id, turn.id, client=self.client)
        with checkpoint_connection(self.engine) as connection:
            graph = build_agent(
                self.model,
                rag,
                turn.language,
                checkpointer=PostgresSaver(connection),
                repair_answers=turn.status == "failed",
            )
            yield LangGraphSession(
                graph, conversation_id, turn.id, sources=rag.service.sources
            )
