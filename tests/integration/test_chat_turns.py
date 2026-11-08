"""Persisted execution through real LangGraph/PostgresSaver, without paid APIs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event
from uuid import uuid4

import pytest
from fixtures.chat import ScriptedModel, call
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select, text, update

from atrag.chat.api.execution import LangGraphBackend, LangGraphSession
from atrag.chat.services.turns import TurnService
from atrag.corpus import publish_corpus
from atrag.db_models import ChatTurnRow
from atrag.schemas.chat.execution import ExecutionState

pytestmark = pytest.mark.integration
pytest_plugins = ["test_chat_tools"]


@pytest.fixture(autouse=True)
def clean_checkpoints(chat, engine):
    def clear():
        with engine.begin() as connection:
            connection.execute(
                text("TRUNCATE checkpoint_writes, checkpoint_blobs, checkpoints")
            )

    clear()
    yield
    clear()


def executor(engine, messages):
    model = ScriptedModel(messages=iter(messages), disable_streaming=True)
    return TurnService(engine, LangGraphBackend(engine, model)), model


def request(chat, turns, turn_id=None, **options):
    _, user, conversation = chat
    return list(
        turns.stream_turn(
            user, conversation, turn_id or uuid4(), "Question?", "pt", **options
        )
    )


def test_completed_turn_replays_without_model_and_keeps_history(chat, engine):
    turns, model = executor(
        engine, [AIMessage(content="First answer"), AIMessage(content="Second answer")]
    )
    turn_id = uuid4()
    events = request(chat, turns, turn_id)
    assert events[-1].kind == "completed"
    saved = events[-1].turn
    assert (saved.id, saved.question, saved.answer, saved.status, saved.error) == (
        turn_id,
        "Question?",
        "First answer",
        "completed",
        None,
    )
    assert (
        "".join(event.text for event in events if event.kind == "text") == saved.answer
    )
    assert saved.model_usage["requests"] == 1
    assert request(chat, turns, turn_id)[-1].turn == saved
    assert len(model.seen) == 1
    engine.dispose()
    request(chat, turns)
    assert len(model.seen) == 2
    assert [m.content for m in model.seen[-1] if isinstance(m, HumanMessage)] == [
        "Question?",
        "Question?",
    ]
    service, user, conversation = chat
    assert [t.answer for t in service.load_history(user, conversation)] == [
        "First answer",
        "Second answer",
    ]
    with engine.connect() as connection:
        assert (
            connection.scalar(
                text("SELECT count(*) FROM checkpoints WHERE thread_id = :id"),
                {"id": str(conversation)},
            )
            > 0
        )


def test_stream_and_saved_answer_use_server_numbers_for_named_evidence(
    chat, materials, engine
):
    service, user, conversation = chat

    def script():
        yield call(args={"query": "CIRS 16", "source_ids": ["first"], "limit": 1})
        key = service.load_history(user, conversation)[-1].sources[0]["citation_key"]
        yield AIMessage(f"Evidence [{key}].")

    turns, _ = executor(engine, script())
    events = request(chat, turns)
    assert events[-1].kind == "completed"
    assert "".join(e.text for e in events if e.kind == "text") == "Evidence [1]."
    assert events[-1].turn.answer == "Evidence [1]."


def test_stream_preserves_an_unclosed_ordinary_bracket(chat, engine):
    turns, _ = executor(engine, [AIMessage("Draft [note")])
    events = request(chat, turns)
    assert events[-1].kind == "completed"
    assert "".join(e.text for e in events if e.kind == "text") == "Draft [note"
    assert events[-1].turn.answer == "Draft [note"


def test_error_requires_explicit_resume_without_duplicate_question(chat, engine):
    def script():
        yield call()
        raise TimeoutError("private transport details")

    turns, model = executor(engine, script())
    turn_id = uuid4()
    failed = request(chat, turns, turn_id)
    assert failed[-1].kind == "error"
    assert failed[-1].code == "execution_failed"
    assert not [event for event in failed if event.kind == "text"]
    assert [
        (event.name, event.call_id, event.phase)
        for event in failed
        if event.kind == "tool"
    ] == [
        ("search_tax_documents", "call-1", "started"),
        ("search_tax_documents", "call-1", "completed"),
    ]
    service, user, conversation = chat
    saved = service.load_history(user, conversation)[0]
    assert saved.status == "failed"
    assert saved.error == "TimeoutError"
    assert saved.model_usage == {
        "requests": 1,
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
    }
    assert request(chat, turns, turn_id)[-1].code == "resume_required"
    assert request(chat, turns)[-1].code == "unfinished_turn"
    assert len(model.seen) == 2
    resumed, next_model = executor(engine, [AIMessage(content="No evidence")])
    done = request(chat, resumed, turn_id, resume=True)[-1]
    assert done.kind == "completed"
    assert done.turn.error is None
    assert done.turn.model_usage["requests"] == 2
    assert len(next_model.seen) == 1
    assert len([m for m in next_model.seen[0] if isinstance(m, HumanMessage)]) == 1
    assert len(service.load_history(user, conversation)) == 1


def test_resume_keeps_old_corpus_and_original_sources(chat, materials, engine):
    def script():
        yield call(
            "load_unit_text",
            {
                "address": {
                    "kind": "unit",
                    "source_id": "first",
                    "snapshot_id": materials["first"],
                    "block_id": "pt:cirs:code:article:16",
                }
            },
        )
        raise TimeoutError()

    turns, _ = executor(engine, script())
    turn_id = uuid4()
    assert request(chat, turns, turn_id)[-1].kind == "error"
    service, user, conversation = chat
    before = service.load_history(user, conversation)[0]
    assert before.sources and before.retrieval_tokens > 0
    publish_corpus(engine, {"first": None, "second": None})
    resumed, _ = executor(
        engine,
        [AIMessage(content=f"Original evidence [{before.sources[0]['citation_key']}]")],
    )
    done = request(chat, resumed, turn_id, resume=True)[-1].turn
    assert done.corpus_release_id == before.corpus_release_id
    assert done.sources == before.sources
    assert done.retrieval_tokens == before.retrieval_tokens
    assert done.embedding_usage == before.embedding_usage
    assert done.sources[0]["address"]["snapshot_id"] == materials["first"]
    following, _ = executor(engine, [AIMessage(content="Empty corpus")])
    assert request(chat, following)[-1].turn.corpus_release_id != done.corpus_release_id


def test_owner_is_checked_before_backend_and_no_uuid_reassignment(
    chat, engine, monkeypatch
):
    turns, model = executor(engine, [AIMessage(content="Private")])
    service, user, conversation = chat
    turn_id = uuid4()
    request(chat, turns, turn_id)

    def forbidden(*args):
        pytest.fail("Backend/checkpoints must not be accessed")

    monkeypatch.setattr(turns.backend, "open", forbidden)
    for stranger in (replace(user, subject="stranger"), replace(user, id=uuid4())):
        with pytest.raises(PermissionError):
            list(
                turns.stream_turn(
                    stranger, conversation, turn_id, "Question?", "pt", resume=True
                )
            )
    with pytest.raises(PermissionError):
        list(turns.stream_turn(user, uuid4(), turn_id, "Question?", "pt"))
    with pytest.raises(ValueError, match="UUID"):
        list(turns.stream_turn(user, conversation, turn_id, "Changed", "pt"))
    with pytest.raises(ValueError, match="UUID"):
        list(turns.stream_turn(user, conversation, turn_id, "Question?", "en"))
    second = service.create_conversation(user)
    with pytest.raises(ValueError, match="UUID"):
        list(turns.stream_turn(user, second.id, turn_id, "Question?", "pt"))
    assert len(model.seen) == 1


def test_unknown_resume_and_invalid_inputs_do_not_create_turns(chat, engine):
    from pydantic import ValidationError

    turns, model = executor(engine, [])
    assert request(chat, turns, resume=True)[-1].code == "turn_not_found"
    service, user, conversation = chat
    for question, language in ((" ", "ru"), ("Question", "es")):
        with pytest.raises(ValidationError):
            list(turns.stream_turn(user, conversation, uuid4(), question, language))
    assert service.load_history(user, conversation) == ()
    assert not model.seen


def test_backend_open_failure_is_saved_and_retry_is_explicit(chat, engine, monkeypatch):
    turns, model = executor(engine, [AIMessage(content="Recovered")])
    original = turns.backend.open

    def fail(*args):
        raise OSError("private connection failure")

    monkeypatch.setattr(turns.backend, "open", fail)
    turn_id = uuid4()
    assert request(chat, turns, turn_id)[-1].code == "execution_failed"
    service, user, conversation = chat
    saved = service.load_history(user, conversation)[0]
    assert (saved.status, saved.error) == ("failed", "OSError")
    assert not model.seen
    monkeypatch.setattr(turns.backend, "open", original)
    assert request(chat, turns, turn_id)[-1].code == "resume_required"
    assert request(chat, turns, turn_id, resume=True)[-1].kind == "completed"


def test_failure_before_new_checkpoint_cannot_copy_previous_answer_or_usage(
    chat, engine, monkeypatch
):
    turns, _ = executor(engine, [AIMessage(content="Previous answer")])
    assert request(chat, turns)[-1].kind == "completed"

    def fail_before_input(self, turn):
        raise OSError("Failed before graph input was checkpointed")

    monkeypatch.setattr(LangGraphSession, "stream", fail_before_input)
    assert request(chat, turns)[-1].code == "execution_failed"
    service, user, conversation = chat
    saved = service.load_history(user, conversation)[-1]
    assert (saved.answer, saved.model_usage) == ("", {})


def test_backend_open_interruption_is_saved_and_completed_event_can_be_closed(
    chat, engine, monkeypatch
):
    turns, _ = executor(engine, [AIMessage(content="Completed")])
    original = turns.backend.open

    def stop(*args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(turns.backend, "open", stop)
    turn_id = uuid4()
    with pytest.raises(KeyboardInterrupt):
        request(chat, turns, turn_id)
    service, user, conversation = chat
    assert service.load_history(user, conversation)[0].status == "interrupted"
    monkeypatch.setattr(turns.backend, "open", original)
    stream = turns.stream_turn(
        user, conversation, turn_id, "Question?", "pt", resume=True
    )
    for event in stream:
        if event.kind == "completed":
            break
    stream.close()
    assert service.load_history(user, conversation)[0].status == "completed"


def test_invalid_citations_are_failed_and_keep_public_text_and_usage(chat, engine):
    message = AIMessage(
        content=[
            {"type": "reasoning", "reasoning": "Secret reasoning"},
            {"type": "text", "text": "Wrong source [9]"},
        ],
        usage_metadata={"input_tokens": 30, "output_tokens": 5, "total_tokens": 35},
    )
    turns, model = executor(engine, [message])
    turn_id = uuid4()
    events = request(chat, turns, turn_id)
    assert events[-1].code == "execution_failed"
    assert (
        "".join(event.text for event in events if event.kind == "text")
        == "Wrong source [9]"
    )
    service, user, conversation = chat
    saved = service.load_history(user, conversation)[0]
    assert (saved.status, saved.error, saved.answer) == (
        "failed",
        "InvalidAnswer",
        "Wrong source [9]",
    )
    assert saved.model_usage == {
        "requests": 1,
        "input_tokens": 30,
        "output_tokens": 5,
        "total_tokens": 35,
    }
    assert request(chat, turns, turn_id, resume=True)[-1].kind == "error"
    assert service.load_history(user, conversation)[0].answer == "Wrong source [9]"
    assert len(model.seen) == 1


def test_real_token_stream_preserves_order_and_never_emits_reasoning(chat, engine):
    model = ScriptedModel(messages=iter([AIMessage(content="Tokens in order")]))
    turns = TurnService(engine, LangGraphBackend(engine, model))
    events = request(chat, turns)
    deltas = [event.text for event in events if event.kind == "text"]
    assert len(deltas) > 1
    assert "".join(deltas) == events[-1].turn.answer == "Tokens in order"


@pytest.mark.parametrize("question", [b"Question?", 123, None])
def test_question_requires_text_at_the_boundary(chat, engine, question):
    from pydantic import ValidationError

    turns, _ = executor(engine, [])
    _, user, conversation = chat
    with pytest.raises(ValidationError):
        list(turns.stream_turn(user, conversation, uuid4(), question, "pt"))


@pytest.mark.parametrize("pending,answer", [(True, "Unfinished"), (False, "")])
def test_incomplete_backend_never_marks_success(chat, engine, pending, answer):
    state = ExecutionState(None, pending, answer, "Partial answer", {"requests": 1})

    class IncompleteSession:
        def stream(self, turn):
            return iter(())

        def state(self):
            return state

    class IncompleteBackend:
        @contextmanager
        def open(self, user, conversation_id, turn):
            yield IncompleteSession()

    turns = TurnService(engine, IncompleteBackend())
    events = request(chat, turns)
    assert events[-1].code == "execution_failed"
    service, user, conversation = chat
    saved = service.load_history(user, conversation)[0]
    assert (saved.status, saved.answer, saved.model_usage) == (
        "failed",
        "Partial answer",
        {"requests": 1},
    )


def test_backend_refuses_to_replace_another_unfinished_checkpoint(chat, engine):
    def script():
        raise TimeoutError()
        yield

    turns, model = executor(engine, script())
    assert request(chat, turns)[-1].code == "execution_failed"
    service, user, conversation = chat
    failed = service.load_history(user, conversation)[0]
    assert (failed.error, failed.answer) == ("TimeoutError", "")
    assert failed.model_usage == {
        "requests": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    turn_id = uuid4()
    service.start_turn(user, conversation, turn_id, "Another?", "en")
    turn = next(
        item for item in service.load_history(user, conversation) if item.id == turn_id
    )
    with (
        turns.backend.open(user, conversation, turn) as session,
        pytest.raises(RuntimeError, match="unfinished checkpoints"),
    ):
        list(session.stream(turn))
    assert len(model.seen) == 1


def test_two_tabs_are_serialized_but_other_conversations_are_independent(chat, engine):
    entered, release = Event(), Event()

    def script():
        entered.set()
        assert release.wait(10)
        yield AIMessage(content="One execution")

    turns, model = executor(engine, script())
    turn_id = uuid4()
    service, user, conversation = chat
    with ThreadPoolExecutor(max_workers=1) as workers:
        running = workers.submit(request, chat, turns, turn_id)
        try:
            assert entered.wait(10)
            with engine.connect() as connection:
                assert (
                    connection.scalar(
                        text("SELECT count(*) FROM checkpoints WHERE thread_id = :id"),
                        {"id": str(conversation)},
                    )
                    > 0
                )
            other, other_model = executor(engine, [AIMessage(content="Independent")])
            assert request(chat, other, turn_id)[-1].code == "conversation_busy"
            assert request(chat, other)[-1].code == "conversation_busy"
            second = service.create_conversation(user)
            assert (
                list(other.stream_turn(user, second.id, uuid4(), "Other?", "en"))[
                    -1
                ].kind
                == "completed"
            )
            assert len(other_model.seen) == 1
        finally:
            release.set()
        assert running.result(timeout=10)[-1].kind == "completed"
    assert len(model.seen) == 1
    assert len(service.load_history(user, conversation)) == 1


def test_abandoned_generator_requires_resume_and_releases_lock(chat, engine):
    turns, _ = executor(engine, [AIMessage(content="Interrupted answer")])
    service, user, conversation = chat
    turn_id = uuid4()
    stream = turns.stream_turn(user, conversation, turn_id, "Question?", "pt")
    assert next(stream).kind == "text"
    stream.close()
    saved = service.load_history(user, conversation)[0]
    assert saved.status == "interrupted"
    assert request(chat, turns, turn_id)[-1].code == "resume_required"
    continued, _ = executor(engine, [AIMessage(content="Resumed answer")])
    assert request(chat, continued, turn_id, resume=True)[-1].kind == "completed"


def test_crash_after_graph_completion_does_not_repeat_model(chat, engine, monkeypatch):
    turns, model = executor(engine, [AIMessage(content="Durable answer")])
    turn_id = uuid4()
    original = turns.store.finish

    def fail(*args, **kwargs):
        raise RuntimeError("process stopped before saving result")

    monkeypatch.setattr(turns.store, "finish", fail)
    assert request(chat, turns, turn_id)[-1].kind == "error"
    monkeypatch.setattr(turns.store, "finish", original)
    resumed, next_model = executor(engine, [])
    done = request(chat, resumed, turn_id, resume=True)[-1]
    assert done.kind == "completed" and done.turn.answer == "Durable answer"
    assert len(model.seen) == 1 and not next_model.seen


def test_pending_and_running_rows_are_never_automatically_resumed(chat, engine):
    service, user, conversation = chat
    turn_id = uuid4()
    service.start_turn(user, conversation, turn_id, "Question?", "pt")
    turns, model = executor(engine, [AIMessage(content="Explicit start")])
    assert request(chat, turns, turn_id)[-1].code == "resume_required"
    with engine.begin() as connection:
        connection.execute(
            update(ChatTurnRow)
            .where(ChatTurnRow.id == turn_id)
            .values(status="running")
        )
    assert request(chat, turns, turn_id)[-1].code == "resume_required"
    assert not model.seen
    assert request(chat, turns, turn_id, resume=True)[-1].kind == "completed"
    with engine.connect() as connection:
        assert (
            connection.scalar(
                select(ChatTurnRow.status).where(ChatTurnRow.id == turn_id)
            )
            == "completed"
        )
