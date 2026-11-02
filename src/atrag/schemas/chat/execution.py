"""Public execution events and the boundary to a checkpointed agent backend."""

from collections.abc import Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Literal, Protocol
from uuid import UUID

from atrag.schemas.chat.conversations import Turn, User


@dataclass(frozen=True)
class TextDelta:
    text: str
    kind: Literal["text"] = field(default="text", init=False)


@dataclass(frozen=True)
class ToolActivity:
    name: str
    call_id: str
    phase: Literal["started", "completed"]
    kind: Literal["tool"] = field(default="tool", init=False)


@dataclass(frozen=True)
class TurnCompleted:
    turn: Turn
    kind: Literal["completed"] = field(default="completed", init=False)


@dataclass(frozen=True)
class TurnError:
    code: Literal[
        "conversation_busy",
        "resume_required",
        "unfinished_turn",
        "execution_failed",
        "turn_not_found",
    ]
    kind: Literal["error"] = field(default="error", init=False)


AgentEvent = TextDelta | ToolActivity
TurnEvent = AgentEvent | TurnCompleted | TurnError


@dataclass(frozen=True)
class ExecutionState:
    turn_id: str | None
    pending: bool
    answer: str
    text: str
    model_usage: dict


class ExecutionSession(Protocol):
    def state(self) -> ExecutionState: ...

    def stream(self, turn: Turn) -> Iterator[AgentEvent]: ...


class TurnBackend(Protocol):
    def open(
        self, user: User, conversation_id: UUID, turn: Turn
    ) -> AbstractContextManager[ExecutionSession]: ...
