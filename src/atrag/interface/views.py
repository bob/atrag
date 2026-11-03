"""Native Streamlit widgets over authenticated chat and execution services."""

from collections.abc import Callable, Generator
from contextlib import closing
from uuid import UUID, uuid4

import streamlit as st

from atrag.interface import resources
from atrag.interface.locales import LANGUAGE_NAMES, LANGUAGES, tr
from atrag.schemas.chat.conversations import Turn, User
from atrag.schemas.chat.execution import TurnEvent
from atrag.schemas.settings import SupportedLanguage


def language_picker(
    language: SupportedLanguage,
    key: str,
    *,
    on_change: Callable[[], None] | None = None,
) -> SupportedLanguage:
    st.session_state.setdefault(key, language)
    selected = st.segmented_control(
        tr("language", language),
        LANGUAGES,
        format_func=LANGUAGE_NAMES.__getitem__,
        key=key,
        required=True,
        on_change=on_change,
    )
    if selected not in LANGUAGES:
        raise ValueError("Unsupported language")
    return selected


def logout(language: SupportedLanguage) -> None:
    if st.button(tr("logout", language), key="logout"):
        st.session_state.clear()
        st.logout()
        st.stop()


def login(language: SupportedLanguage) -> None:
    with st.sidebar:
        selected = language_picker(language, "public_language")
    if selected != language:
        st.session_state["ui_language"] = selected
        st.rerun()
    st.info(tr("login_hint", language))
    try:
        configured = bool(st.secrets.get("auth", {}).get("google"))
    except FileNotFoundError:
        configured = False
    if not configured:
        st.warning(tr("auth_setup", language))
    if st.button(tr("login", language), key="login", disabled=not configured):
        st.login("google")


def authenticate(language: SupportedLanguage):
    try:
        access = st.secrets.get("access", {})
    except FileNotFoundError:
        access = {}
    service = resources.get_chat_service()
    try:
        user = service.authenticate(
            st.user.to_dict(),
            access.get("allowed_emails", []),
            check_allowed_emails=access.get("check_allowed_emails", True),
        )
    except PermissionError:
        st.error(tr("denied", language))
        logout(language)
        st.stop()
    return service, user


def sidebar(service, user: User):
    language = user.language
    if "language_selection" in st.session_state:
        selected = st.session_state.pop("language_selection")
        if selected not in LANGUAGES:
            raise ValueError("Unsupported language")
        service.set_language(user, selected)
        st.rerun()
    with st.sidebar:
        key = f"language_{user.id}"
        st.session_state[key] = language
        language_picker(language, key, on_change=lambda: select_language(key))
        if st.button(tr("new", language), key="new_conversation", width="stretch"):
            conversation = service.create_conversation(user)
            st.session_state["route_selection"] = str(conversation.id)
        conversations = {str(c.id): c for c in service.list_conversations(user)}
        labels = {
            value: f"{position}. {item.title or tr('untitled', language)}"
            for position, (value, item) in enumerate(conversations.items(), 1)
        }
        conversation = restore_conversation(service, user)
        st.session_state["conversation"] = (
            str(conversation.id) if conversation is not None else None
        )
        st.selectbox(
            tr("conversations", language),
            tuple(conversations),
            index=None,
            format_func=labels.__getitem__,
            key="conversation",
            on_change=select_conversation,
        )
        corpus_view(language)
        logout(language)
    if conversation is None:
        st.info(tr("choose", language))
        st.stop()
    return conversation


def select_language(key: str) -> None:
    st.session_state["language_selection"] = st.session_state[key]


def restore_conversation(service, user: User):
    if "route_selection" in st.session_state:
        chosen = st.session_state.pop("route_selection")
        if chosen is None:
            st.query_params.pop("conversation", None)
        else:
            st.query_params["conversation"] = chosen
    values = st.query_params.get_all("conversation")
    if not values:
        return None
    if len(values) != 1:
        raise PermissionError("Access denied")
    try:
        conversation_id = UUID(values[0])
    except ValueError:
        raise PermissionError("Access denied") from None
    return service.get_conversation(user, conversation_id)


def select_conversation() -> None:
    # Streamlit's native bind serializes format_func labels instead of UUIDs.
    # Titles change after the first question and can coincide across dialogs.
    st.session_state["route_selection"] = st.session_state["conversation"]


def corpus_view(language: SupportedLanguage) -> None:
    composition = resources.get_composition()
    with st.expander(tr("corpus", language)):
        st.caption(f"{tr('release', language)}: {composition.release_id}")
        for source, version in composition.members:
            st.text(source)
            st.caption(f"{tr('version', language)}: {version}")
        if not composition.members:
            st.info(tr("empty_corpus", language))


def source_view(source: dict, language: SupportedLanguage) -> None:
    address = source.get("address", {})
    st.text(
        f"[{source['citation_id']}] {source.get('document_title') or address.get('source_id', '')}"
    )
    st.caption(source.get("citation_label", ""))
    st.caption(
        f"{address.get('source_id', '')} · {tr('version', language)}: {address.get('snapshot_id', '')} · {address.get('block_id', '')}"
    )
    source_metadata(source, language)
    st.text(source.get("text_original", ""))
    for table in source.get("tables", []):
        st.table(table["header_rows"] + table["rows"], alt=tr("sources", language))
        if table.get("review_note"):
            st.warning(table["review_note"])
    source_notes(source, language)


def source_metadata(source: dict, language: SupportedLanguage) -> None:
    for location in source.get("locations", []):
        parts = [
            f"{tr(key, language)}: {location[field]}"
            for key, field in (
                ("pdf", "page"),
                ("section", "section"),
                ("anchor", "anchor"),
                ("paragraph", "paragraph"),
            )
            if location.get(field) is not None
        ]
        st.caption(" · ".join(parts))
    for field in (
        "consolidation_label",
        "publication_date",
        "revision_publication_date",
    ):
        if source.get(field):
            st.caption(f"{tr('revision', language)}: {source[field]}")


def source_notes(source: dict, language: SupportedLanguage) -> None:
    if source.get("complete") is False:
        st.warning(tr("incomplete", language))
    for note in source.get("editorial_notes", []):
        st.text(note["raw_text"])
    for issue in source.get("issues", []):
        st.warning(str(issue))


def history_view(history: tuple[Turn, ...], language: SupportedLanguage) -> Turn | None:
    resume = None
    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn.question)
        with st.chat_message("assistant"):
            st.markdown(turn.answer)
            if turn.sources:
                with st.expander(tr("sources", language)):
                    for source in turn.sources:
                        source_view(source, language)
            if turn.status != "completed":
                st.warning(tr("execution_failed", language))
                if st.button(tr("resume", language), key=f"resume_{turn.id}"):
                    resume = turn
    return resume


def text_stream(
    events: Generator[TurnEvent], status, language: SupportedLanguage
) -> Generator[str]:
    with closing(events):
        for event in events:
            match event.kind:
                case "text":
                    status.update(label=tr("preparing", language))
                    yield event.text
                case "tool":
                    label = (
                        "search" if event.name == "search_tax_documents" else "reading"
                    )
                    status.update(label=tr(label, language))
                case "completed":
                    status.update(label=tr("done", language), state="complete")
                case "error":
                    st.session_state["turn_error"] = event.code
                    status.update(label=tr(event.code, language), state="error")


def execute(
    user: User,
    conversation_id: UUID,
    turn_id: UUID,
    text: str,
    language: SupportedLanguage,
    *,
    resume: bool = False,
) -> None:
    with st.chat_message("assistant"):
        status = st.status(tr("preparing", user.language), type="compact")
        events = resources.get_turn_service().stream_turn(
            user, conversation_id, turn_id, text, language, resume=resume
        )
        st.write_stream(text_stream(events, status, user.language))
    st.rerun()


def chat(service, user: User, conversation_id: UUID) -> None:
    history = service.load_history(user, conversation_id)
    with st.container(key=f"messages_{conversation_id}"):
        resume = history_view(history, user.language)
    if error := st.session_state.pop("turn_error", None):
        st.error(tr(error, user.language))
    prompt = st.chat_input(
        tr("question", user.language),
        key="question",
        max_chars=32000,
        submit_mode="disable",
        disabled=any(t.status != "completed" for t in history),
    )
    if resume is not None:
        execute(
            user,
            conversation_id,
            resume.id,
            resume.question,
            resume.language,
            resume=True,
        )
    if prompt and prompt.strip():
        with st.chat_message("user"):
            st.markdown(prompt)
        execute(user, conversation_id, uuid4(), prompt, user.language)
