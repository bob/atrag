"""Run with: uv run streamlit run streamlit_app.py."""

import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

from atrag.interface import views
from atrag.interface.locales import tr

st.set_page_config(page_title="ATRAG", page_icon=":material/forum:")
st.session_state.setdefault("ui_language", "ru")
language = st.session_state["ui_language"]
st.title("ATRAG")

try:
    if not st.user.get("is_logged_in", False):
        st.caption(tr("subtitle", language))
        views.login(language)
        st.stop()
    service, user = views.authenticate(language)
    language = user.language
    st.session_state["ui_language"] = language
    st.caption(tr("subtitle", language))
    conversation = views.sidebar(service, user)
    views.chat(service, user, conversation.id)
except PermissionError:
    st.error(tr("denied", language))
except (SQLAlchemyError, OSError, ValueError):
    st.error(tr("unavailable", language))
