"""Own a turn's lifecycle independently of UI reruns and agent implementation."""

import logging
from collections.abc import Generator
from uuid import UUID

from atrag.chat.db.execution import ExecutionStore
from atrag.chat.services.chat import LANGUAGE, QUESTION
from atrag.schemas.chat.conversations import User
from atrag.schemas.chat.execution import (
    TurnBackend,
    TurnCompleted,
    TurnError,
    TurnEvent,
)
from atrag.schemas.settings import SupportedLanguage

logger = logging.getLogger(__name__)


class TurnService:
    def __init__(self, engine, backend: TurnBackend):
        self.store = ExecutionStore(engine)
        self.backend = backend

    def stream_turn(
        self,
        user: User,
        conversation_id: UUID,
        turn_id: UUID,
        text: str,
        language: SupportedLanguage,
        *,
        resume: bool = False,
    ) -> Generator[TurnEvent]:
        question = QUESTION.validate_python(text, strict=True)
        language = LANGUAGE.validate_python(language)
        with self.store.conversation_lock(user, conversation_id) as acquired:
            if not acquired:
                yield TurnError("conversation_busy")
                return
            turn = self.store.get_turn(user, conversation_id, turn_id)
            if turn is not None:
                if (turn.question, turn.language) != (question, language):
                    raise ValueError("Turn UUID is already assigned to another request")
                if turn.status == "completed":
                    yield TurnCompleted(turn)
                    return
                if not resume:
                    yield TurnError("resume_required")
                    return
            elif resume:
                yield TurnError("turn_not_found")
                return
            if any(
                item.id != turn_id and item.status != "completed"
                for item in self.store.load_history(user, conversation_id)
            ):
                yield TurnError("unfinished_turn")
                return
            self.store.start_turn(user, conversation_id, turn_id, question, language)
            turn = self.store.get_turn(user, conversation_id, turn_id)
            yield from self._execute(user, conversation_id, turn)

    def _execute(self, user, conversation_id, turn):
        self.store.save_execution(
            user, conversation_id, turn.id, status="running", error=None
        )
        try:
            with self.backend.open(user, conversation_id, turn) as session:
                saved = yield from self._stream_session(
                    user, conversation_id, turn, session
                )
        except BaseException as error:
            self.store.save_execution(
                user,
                conversation_id,
                turn.id,
                status="failed" if isinstance(error, Exception) else "interrupted",
                error=type(error).__name__,
            )
            if not isinstance(error, Exception):
                raise
            logger.exception("Chat turn execution failed: %s", turn.id)
            yield TurnError("execution_failed")
        else:
            yield TurnCompleted(saved)

    def _stream_session(self, user, conversation_id, turn, session):
        text = ""
        try:
            for event in session.stream(turn):
                if event.kind == "text":
                    text += event.text
                yield event
            state = session.state()
            if state.pending or not state.answer:
                raise RuntimeError("Agent did not finish")
            saved = self.store.finish(user, conversation_id, turn.id, state)
        except BaseException:
            state = session.state()
            self.store.save_execution(
                user,
                conversation_id,
                turn.id,
                answer=text or state.text,
                model_usage=state.model_usage,
            )
            raise
        return saved
