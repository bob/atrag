"""AppTest through real PostgreSQL services, checkpoints and a local model."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
import test_chat_turns
from fixtures.chat import ScriptedModel
from langchain_core.messages import AIMessage
from streamlit.testing.v1 import AppTest

from atrag.chat.api.execution import LangGraphBackend
from atrag.chat.services.turns import TurnService
from atrag.interface import resources

pytestmark = pytest.mark.integration
pytest_plugins = ["test_chat_tools"]
clean_checkpoints = test_chat_turns.clean_checkpoints
APP = Path(__file__).resolve().parents[2] / "streamlit_app.py"


@pytest.mark.parametrize(
    "language,question,subtitle",
    [
        ("pt", "Faça uma pergunta", "Perguntas sobre documentos"),
        ("uk", "Поставте запитання", "Питання щодо документів"),
    ],
)
def test_ui_persists_reopens_and_rejects_another_owner(
    chat, engine, monkeypatch, language, question, subtitle
):
    service, user, conversation = chat
    claims = {
        "is_logged_in": True,
        "iss": user.issuer,
        "sub": user.subject,
        "email": user.email,
        "email_verified": True,
    }
    monkeypatch.setattr(
        st, "user", SimpleNamespace(get=claims.get, to_dict=lambda: dict(claims))
    )
    monkeypatch.setattr(resources, "get_engine", lambda: engine)
    model = ScriptedModel(
        messages=iter([AIMessage(content="Resposta guardada")]), disable_streaming=True
    )
    turns = TurnService(engine, LangGraphBackend(engine, model))
    monkeypatch.setattr(resources, "get_turn_service", lambda: turns)

    def open_app():
        app = AppTest.from_file(str(APP), default_timeout=20)
        app.secrets["access"] = {"allowed_emails": [user.email]}
        app.query_params["conversation"] = str(conversation)
        return app.run()

    app = open_app()
    assert app.sidebar.selectbox[0].value == "en"
    assert app.caption[0].value == "Questions about documents"
    assert app.sidebar.selectbox[0].options == [
        "Português",
        "English",
        "Українська",
        "Русский",
    ]
    app.sidebar.selectbox[0].set_value(language).run()
    app.chat_input[0].set_value("Question from UI").run()
    assert not app.exception
    turn = service.load_history(user, conversation)[0]
    assert turn.answer == "Resposta guardada"
    assert turn.language == language
    reopened = open_app()
    assert not reopened.exception
    assert reopened.chat_input[0].placeholder == question
    assert reopened.caption[0].value == subtitle
    assert [m.markdown[0].value for m in reopened.chat_message] == [
        "Question from UI",
        "Resposta guardada",
    ]
    assert len(model.seen) == 1
    claims["sub"] = "another-owner"
    denied = open_app()
    assert not denied.exception
    assert not denied.chat_message
    assert not denied.chat_input
    assert denied.error[0].value == "Access denied"
