"""Drive the real entrypoint with detached service values and no network."""

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from streamlit.testing.v1 import AppTest

from atrag.chat.core.identity import parse_google_identity
from atrag.corpus import CorpusComposition
from atrag.schemas.chat.conversations import Conversation, Turn, User
from atrag.schemas.chat.execution import (
    TextDelta,
    ToolActivity,
    TurnCompleted,
    TurnError,
)

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
NOW = datetime.now(UTC)
CLAIMS = {
    "is_logged_in": True,
    "iss": "https://accounts.google.com",
    "sub": "owner",
    "email": "a@b",
    "email_verified": True,
}
SOURCE = {
    "citation_id": 1,
    "address": {"source_id": "cirs", "snapshot_id": "old-version", "block_id": "art16"},
    "document_title": "CIRS",
    "citation_label": "Artigo 16.º",
    "text_original": "São residentes em território português.",
    "locations": [{"kind": "pdf", "page": 12}],
    "consolidation_label": "Redação 2025",
    "revision_publication_date": "2025-01-01",
    "editorial_notes": [{"raw_text": "Nota original."}],
    "issues": ["missing_continuation"],
    "complete": False,
}


class MemoryChat:
    def __init__(self):
        self.user = User(
            uuid4(), "https://accounts.google.com", "owner", "a@b", "ru", NOW
        )
        self.conversations = {}
        self.history = {}

    def authenticate(self, claims, allowed, *, check_allowed_emails=True):
        parse_google_identity(
            claims, allowed, check_allowed_emails=check_allowed_emails
        )
        return self.user

    def set_language(self, user, language):
        self.user = replace(user, language=language)
        return self.user

    def create_conversation(self, user):
        conversation = Conversation(uuid4(), user.id, "", NOW)
        self.conversations[conversation.id] = conversation
        self.history[conversation.id] = []
        return conversation

    def list_conversations(self, user):
        return tuple(self.conversations.values())

    def get_conversation(self, user, conversation_id):
        if conversation_id not in self.conversations:
            raise PermissionError("Access denied")
        return self.conversations[conversation_id]

    def load_history(self, user, conversation_id):
        self.get_conversation(user, conversation_id)
        return tuple(self.history[conversation_id])


@pytest.fixture
def ui(monkeypatch):
    from atrag.interface import resources

    chat = MemoryChat()
    calls = []
    outcome = {"fail": False}

    def stream(user, conversation_id, turn_id, text, language, *, resume=False):
        chat.get_conversation(user, conversation_id)
        calls.append((conversation_id, turn_id, text, language, resume))
        history = chat.history[conversation_id]
        if not resume:
            history.append(
                Turn(
                    turn_id,
                    conversation_id,
                    language,
                    text,
                    "",
                    [],
                    "running",
                    {},
                    {},
                    None,
                    None,
                    "release",
                    0,
                    NOW,
                )
            )
        yield ToolActivity("search_tax_documents", "call1", "started")
        yield ToolActivity("search_tax_documents", "call1", "completed")
        yield TextDelta("Resposta [1]")
        if outcome["fail"]:
            history[-1] = replace(
                history[-1], answer="Partial", status="failed", error="private-secret"
            )
            yield TurnError("execution_failed")
        else:
            history[-1] = replace(
                history[-1], answer="Resposta [1]", sources=[SOURCE], status="completed"
            )
            chat.conversations[conversation_id] = replace(
                chat.conversations[conversation_id], title=text
            )
            yield TurnCompleted(history[-1])

    monkeypatch.setattr(resources, "get_chat_service", lambda: chat)
    monkeypatch.setattr(
        resources, "get_turn_service", lambda: SimpleNamespace(stream_turn=stream)
    )
    composition = Mock(
        return_value=CorpusComposition("release", (("cirs", "current-version"),))
    )
    monkeypatch.setattr(resources, "get_composition", composition)
    app = AppTest.from_file(str(APP), default_timeout=10)

    class Claims(dict):
        def to_dict(self):
            return dict(self)

    claims = Claims(CLAIMS)
    monkeypatch.setattr("streamlit.user", claims)
    monkeypatch.setattr("streamlit.logout", lambda: claims.update(is_logged_in=False))
    app.secrets["access"] = {"allowed_emails": ["a@b"]}
    app.secrets["auth"] = {"google": {"client_id": "test"}}
    return SimpleNamespace(
        claims=claims,
        app=app,
        chat=chat,
        calls=calls,
        outcome=outcome,
        composition=composition,
    )


def content(app):
    assert not app.exception
    return "\n".join(
        str(element.value)
        for kind in ("markdown", "caption", "warning", "error", "text")
        for element in app.get(kind)
    )


def test_login_uses_google_and_does_not_read_corpus(ui, monkeypatch):
    login = Mock()
    monkeypatch.setattr("streamlit.login", login)
    ui.claims["is_logged_in"] = False
    ui.app.run()
    assert "Google" in ui.app.button(key="login").label
    ui.app.button(key="login").click().run()
    login.assert_called_once_with("google")
    assert not ui.app.chat_input
    ui.composition.assert_not_called()


def test_first_page_load_defaults_to_english(ui):
    ui.claims["is_logged_in"] = False
    ui.app.run()
    assert not ui.app.exception
    assert ui.app.caption[0].value == "Questions about documents"
    assert ui.app.button(key="login").label == "Sign in with Google"
    assert ui.app.sidebar.selectbox(key="public_language").value == "en"
    ui.app.run()
    assert ui.app.sidebar.selectbox(key="public_language").value == "en"


@pytest.mark.parametrize(
    "claims,allowed",
    [
        ({"email_verified": False}, ["a@b"]),
        ({"iss": "https://evil.test"}, ["a@b"]),
        ({}, []),
        ({}, "a@b"),
    ],
)
def test_denied_identity_cannot_open_chat(ui, claims, allowed):
    ui.claims.update(claims)
    ui.app.secrets["access"] = {"allowed_emails": allowed}
    ui.app.run()
    assert "Access denied" in content(ui.app)
    assert not ui.app.chat_input
    ui.composition.assert_not_called()


def test_allowlist_can_be_disabled_and_reenabled_on_rerun(ui):
    ui.claims["email"] = "new-user@gmail.com"
    ui.app.secrets["access"] = {"check_allowed_emails": False}
    ui.app.run().button(key="new_conversation").click().run()
    assert not ui.app.exception
    assert ui.app.chat_input
    assert not ui.app.error
    ui.app.secrets["access"] = {"check_allowed_emails": True}
    ui.app.run()
    assert "Доступ запрещён" in content(ui.app)
    assert not ui.app.chat_input


def test_disabled_allowlist_still_requires_google_login(ui):
    ui.claims["is_logged_in"] = False
    ui.app.secrets["access"] = {"check_allowed_emails": False}
    ui.app.run()
    assert not ui.app.exception
    assert ui.app.button(key="login")
    assert not ui.app.chat_input
    ui.composition.assert_not_called()


@pytest.mark.parametrize(
    "claims", [{"email_verified": False}, {"iss": "https://evil.test"}, {"sub": ""}]
)
def test_disabled_allowlist_still_rejects_invalid_identity(ui, claims):
    ui.app.session_state["ui_language"] = "en"
    ui.claims.update(claims)
    ui.app.secrets["access"] = {"check_allowed_emails": False}
    ui.app.run()
    assert "Access denied" in content(ui.app)
    assert not ui.app.chat_input
    ui.composition.assert_not_called()


@pytest.mark.parametrize("check_allowed_emails", [None, "false", 0])
def test_invalid_allowlist_switch_cannot_open_chat(ui, check_allowed_emails):
    ui.app.session_state["ui_language"] = "en"
    ui.app.secrets["access"] = {
        "check_allowed_emails": check_allowed_emails,
        "allowed_emails": ["a@b"],
    }
    ui.app.run()
    assert "Access denied" in content(ui.app)
    assert not ui.app.chat_input
    ui.composition.assert_not_called()


def test_create_stream_sources_rerun_and_reopen_url(ui):
    ui.app.run().button(key="new_conversation").click().run()
    conversation = next(iter(ui.chat.conversations))
    assert ui.app.query_params["conversation"] == str(conversation)
    ui.app.chat_input[0].set_value("Residência?").run()
    text = content(ui.app)
    for expected in (
        "[1] CIRS",
        "Residência?",
        "Resposta [1]",
        "São residentes",
        "old-version",
        "current-version",
        "Redação 2025",
        "12",
        "Nota original.",
        "missing_continuation",
        "неполный",
    ):
        assert expected in text
    assert len(ui.calls) == 1
    assert [item.name for item in ui.app.chat_message] == ["user", "assistant"]
    ui.app.run()
    assert len(ui.calls) == 1
    ui.app.session_state["conversation"] = None
    ui.app.query_params["conversation"] = str(conversation)
    ui.app.run()
    assert "Resposta [1]" in content(ui.app)


@pytest.mark.parametrize(
    "language,label",
    [
        ("pt", "Faça uma pergunta"),
        ("en", "Ask a question"),
        ("uk", "Поставте запитання"),
        ("ru", "Задайте вопрос"),
    ],
)
def test_language_persists_and_only_changes_new_turns(ui, language, label):
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.chat_input[0].set_value("Old question").run()
    ui.app.sidebar.selectbox[0].set_value(language).run()
    assert ui.chat.user.language == language
    assert ui.app.chat_input[0].placeholder == label
    assert "Resposta [1]" in content(ui.app)
    ui.app.chat_input[0].set_value("Next question").run()
    assert ui.calls[-1][3] == language
    assert ui.calls[0][3] == "ru"


def test_switching_conversations_does_not_mix_history(ui):
    first = ui.chat.create_conversation(ui.chat.user)
    second = ui.chat.create_conversation(ui.chat.user)
    ui.app.query_params["conversation"] = str(first.id)
    ui.chat.load_history = Mock(wraps=ui.chat.load_history)
    ui.app.run()
    assert ui.app.selectbox(key="conversation").options == [
        "1. Без названия",
        "2. Без названия",
    ]
    ui.app.chat_input[0].set_value("Only first").run()
    assert ui.calls[0][0] == first.id
    assert ui.chat.history[second.id] == []
    ui.app.selectbox(key="conversation").select(str(second.id)).run()
    assert ui.app.query_params["conversation"] == str(second.id)
    assert ui.chat.load_history.call_args.args[1] == second.id
    assert "Only first" not in content(ui.app)
    ui.app.selectbox(key="conversation").select(str(first.id)).run()
    assert "Only first" in content(ui.app)


@pytest.mark.parametrize("value", ["bad-uuid", str(uuid4())])
def test_unknown_url_never_loads_history(ui, value):
    ui.chat.load_history = Mock(side_effect=AssertionError("Unauthorized history read"))
    ui.app.query_params["conversation"] = value
    ui.app.run()
    assert not ui.app.exception
    assert not ui.app.chat_input
    ui.chat.load_history.assert_not_called()


def test_access_is_checked_before_history(ui):
    conversation = ui.chat.create_conversation(ui.chat.user)
    ui.chat.get_conversation = Mock(side_effect=PermissionError("private details"))
    ui.chat.load_history = Mock()
    ui.app.query_params["conversation"] = str(conversation.id)
    ui.app.run()
    assert "Доступ запрещён" in content(ui.app)
    assert "private details" not in content(ui.app)
    ui.chat.load_history.assert_not_called()


def test_failure_requires_explicit_resume_with_same_uuid_and_language(ui):
    ui.outcome["fail"] = True
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.chat_input[0].set_value("Question").run()
    assert "private-secret" not in content(ui.app)
    assert ui.app.chat_input[0].disabled
    ui.app.run()
    assert len(ui.calls) == 1
    ui.app.sidebar.selectbox[0].set_value("pt").run()
    ui.outcome["fail"] = False
    resume = next(b for b in ui.app.button if b.key.startswith("resume_"))
    resume.click().run()
    assert len(ui.calls) == 2
    assert ui.calls[0][:4] == ui.calls[1][:4]
    assert ui.calls[1][4] is True
    assert len(next(iter(ui.chat.history.values()))) == 1
    assert "Resposta [1]" in content(ui.app)
    assert not ui.app.chat_input[0].disabled


def test_allowlist_rechecked_and_logout_clears_access(ui):
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.secrets["access"] = {"allowed_emails": []}
    ui.app.run()
    assert "Доступ запрещён" in content(ui.app)
    ui.app.button(key="logout").click().run()
    assert not ui.claims["is_logged_in"]
    assert not ui.app.chat_input


@pytest.mark.parametrize("language", ["pt", "en", "uk", "ru"])
def test_login_locales_and_missing_auth_configuration(ui, language):
    from atrag.interface.locales import tr

    ui.claims["is_logged_in"] = False
    ui.app.secrets["auth"] = {}
    ui.app.run().sidebar.selectbox(key="public_language").set_value(language).run()
    assert not ui.app.exception
    assert ui.app.sidebar.selectbox(key="public_language").options == [
        "Português",
        "English",
        "Українська",
        "Русский",
    ]
    assert ui.app.button(key="login").label == tr("login", language)
    assert ui.app.button(key="login").disabled
    assert ui.app.warning[0].value == tr("auth_setup", language)


def test_empty_corpus_refreshes_and_blank_question_does_not_run(ui):
    ui.composition.return_value = CorpusComposition("empty", ())
    ui.app.run().button(key="new_conversation").click().run()
    assert "пока нет документов" in ui.app.info[0].value
    ui.app.chat_input[0].set_value("   ").run()
    assert not ui.calls
    ui.composition.return_value = CorpusComposition(
        "new-release", (("html-doc", "new-version"),)
    )
    ui.app.run()
    assert "new-version" in content(ui.app)


def test_non_pdf_sources_tables_and_optional_metadata(ui):
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.chat_input[0].set_value("Question").run()
    history = next(iter(ui.chat.history.values()))
    history[-1] = replace(
        history[-1],
        sources=[
            {
                **SOURCE,
                "locations": [
                    {"kind": "html", "anchor": "residencia", "section": "Residência"},
                    {"kind": "docx", "paragraph": 3},
                ],
                "complete": True,
                "tables": [
                    {
                        "header_rows": [["Escalão", "Taxa"]],
                        "rows": [["A", "10%"]],
                        "review_note": "Transcrição revista.",
                    }
                ],
            },
            {"citation_id": 2, "text_original": "Sem localização."},
        ],
    )
    ui.app.run()
    text = content(ui.app)
    for expected in (
        "residencia",
        "Residência",
        "Абзац: 3",
        "Sem localização.",
        "Transcrição revista.",
    ):
        assert expected in text
    assert "неполный" not in text
    assert ui.app.table[0].value.iloc[0, 0] == "Escalão"


@pytest.mark.parametrize(
    "language,law_label,faq_label,open_label,candidate_note,live_note",
    [
        (
            "pt",
            "Lei",
            "Esclarecimento da AT",
            "Abrir fonte oficial",
            "O PDF oficial",
            "A página oficial",
        ),
        (
            "en",
            "Law",
            "AT explanation",
            "Open official source",
            "The official PDF",
            "The official page",
        ),
        (
            "uk",
            "Закон",
            "Роз’яснення AT",
            "Відкрити офіційне джерело",
            "Офіційний PDF",
            "Офіційна сторінка",
        ),
        (
            "ru",
            "Закон",
            "Разъяснение AT",
            "Открыть официальный источник",
            "Официальный PDF",
            "Официальная страница",
        ),
    ],
)
def test_saved_source_links_and_types_in_each_language(
    ui, language, law_label, faq_label, open_label, candidate_note, live_note
):
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.chat_input[0].set_value("Question").run()
    history = next(iter(ui.chat.history.values()))
    law = {
        **SOURCE,
        "document_type": "law",
        "source_format": "pdf",
        "source_url_status": "candidate",
        "url": "https://example.org/CIRS.pdf#page=12",
    }
    faq = {
        **SOURCE,
        "citation_id": 2,
        "document_title": "Questões Frequentes",
        "document_type": "faq",
        "source_format": "html_collection",
        "source_url_status": "verified",
        "url": "https://example.org/topic.aspx#heading20",
        "faq_id": "3286",
        "question": "Quem é residente?",
        "topic_path": ["IRS", "Residentes"],
        "locations": [{"kind": "html", "anchor": "heading20"}],
    }
    history[-1] = replace(history[-1], answer="Lei [1]; FAQ [2]", sources=[law, faq])
    ui.chat.user = replace(ui.chat.user, language=language)
    ui.app.run()
    assert not ui.app.exception
    buttons = ui.app.get("link_button")
    assert [b.proto.url for b in buttons] == [law["url"], faq["url"]]
    assert [b.proto.label for b in buttons] == [open_label, open_label]
    text = content(ui.app)
    for expected in (
        law_label,
        faq_label,
        "FAQ AT",
        "3286",
        "IRS",
        "Residentes",
        "Quem é residente?",
    ):
        assert expected in text
    assert any("12" in caption.value for caption in ui.app.caption)
    assert len(ui.app.info) == 2  # Live links do not promise immutable copies.
    assert ui.app.info[0].value.startswith(candidate_note)
    assert ui.app.info[1].value.startswith(live_note)
    assert any(t.value == "[1] CIRS" for t in ui.app.text)
    # Reopening and a newer composition cannot replace the saved URLs.
    ui.composition.return_value = CorpusComposition("new", (("cirs", "new-version"),))
    ui.app.run()
    assert [b.proto.url for b in ui.app.get("link_button")] == [law["url"], faq["url"]]


def test_legacy_history_has_no_invented_link(ui):
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.chat_input[0].set_value("Question").run()
    assert not ui.app.get("link_button")
    assert "old-version" in content(ui.app)
    history = next(iter(ui.chat.history.values()))
    history[-1] = replace(
        history[-1],
        sources=[
            {"citation_id": 2, "address": {"source_id": "legacy-source"}},
            {
                "citation_id": 3,
                "document_type": "faq",
                "locations": [
                    {
                        "kind": "html",
                        "page_url": "https://example.org/new",
                        "anchor": "heading99",
                    }
                ],
            },
        ],
    )
    ui.app.run()
    assert any(t.value == "[2] legacy-source" for t in ui.app.text)
    assert not ui.app.get("link_button")


def test_repeated_url_values_are_denied(ui):
    first = ui.chat.create_conversation(ui.chat.user)
    ui.app.query_params["conversation"] = [str(first.id), str(first.id)]
    ui.app.run()
    assert "Доступ запрещён" in content(ui.app)
    assert not ui.app.chat_input


def test_clearing_selection_closes_history(ui):
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.selectbox(key="conversation").set_value(None).run()
    assert not ui.app.exception
    assert not ui.app.chat_input
    assert "conversation" not in ui.app.query_params


def test_connection_errors_are_localized_without_details(ui, monkeypatch):
    from atrag.interface import resources

    monkeypatch.setattr(
        resources, "get_chat_service", Mock(side_effect=ValueError("private-dsn"))
    )
    ui.app.run()
    assert "The service is unavailable" in content(ui.app)
    assert "private-dsn" not in content(ui.app)


def test_logout_on_chat_page(ui):
    ui.app.run().button(key="new_conversation").click().run()
    ui.app.button(key="logout").click().run()
    assert not ui.app.exception
    assert not ui.app.chat_input
    assert not ui.claims["is_logged_in"]


def test_stream_displays_only_text_and_closes_generator(ui):
    from atrag.interface.views import text_stream

    closed = []

    def events():
        try:
            yield ToolActivity("load_unit_text", "id", "started")
            yield TextDelta("original")
            yield TurnCompleted(Mock())
        finally:
            closed.append(True)

    status = Mock()
    assert list(text_stream(events(), status, "en")) == ["original"]
    assert closed == [True]
    assert status.update.call_args_list[0].kwargs == {"label": "Reading material"}
    assert status.update.call_args_list[-1].kwargs == {
        "label": "Done",
        "state": "complete",
    }


def test_partial_stream_close_releases_backend(ui):
    from atrag.interface.views import text_stream

    closed = []

    def events():
        try:
            yield TextDelta("first")
            yield TextDelta("second")
        finally:
            closed.append(True)

    stream = text_stream(events(), Mock(), "pt")
    assert next(stream) == "first"
    stream.close()
    assert closed == [True]


@pytest.mark.parametrize("logged_in", [False, True])
def test_missing_secrets_closes_access(ui, monkeypatch, logged_in):
    class MissingSecrets:
        def get(self, key, default=None):
            raise FileNotFoundError("No secrets found")

    ui.claims["is_logged_in"] = logged_in
    ui.app.secrets = {}
    monkeypatch.setattr("streamlit.secrets", MissingSecrets())
    ui.app.run()
    assert not ui.app.exception
    assert not ui.app.chat_input
    if logged_in:
        assert ui.app.error[0].value == "Access denied"
    else:
        assert ui.app.button(key="login").disabled
        assert "secrets.toml" in ui.app.warning[0].value


@pytest.mark.parametrize("selected", [None, "es"])
def test_invalid_language_is_rejected_before_use(monkeypatch, selected):
    from atrag.interface.views import language_picker

    monkeypatch.setattr("streamlit.selectbox", lambda *args, **kwargs: selected)
    with pytest.raises(ValueError, match="Unsupported language"):
        language_picker("pt", "language")


def test_another_session_language_is_not_overwritten_on_rerun(ui):
    ui.app.run()
    ui.chat.user = replace(ui.chat.user, language="pt")
    ui.app.run()
    assert ui.chat.user.language == "pt"
    assert ui.app.sidebar.selectbox[0].value == "pt"
    assert ui.app.caption[0].value == "Perguntas sobre documentos"
