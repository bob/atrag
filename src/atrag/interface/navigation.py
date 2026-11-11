"""Native page routing with a shared sidebar and a retained chat selection."""

from pathlib import Path

import streamlit as st

from atrag.interface import resources, views
from atrag.interface.locales import tr
from atrag.schemas.chat.conversations import User

PAGES = Path(__file__).with_name("app_pages")


def switch_mode() -> None:
    destination = "users" if st.session_state["app_mode"] == "admin" else "chat"
    st.session_state["mode_destination"] = PAGES / f"{destination}.py"


def run(service, user: User) -> None:
    # Hidden routes still run their service authorization for direct URLs.
    page = st.navigation(
        [
            st.Page(
                PAGES / "chat.py",
                title=tr("chat_mode", user.language),
                icon=":material/forum:",
                default=True,
            ),
            st.Page(PAGES / "users.py", title="Users", icon=":material/group:"),
            st.Page(
                PAGES / "documents.py",
                title="Documents",
                icon=":material/description:",
            ),
        ],
        position="hidden",
    )
    if destination := st.session_state.pop("mode_destination", None):
        st.switch_page(destination)
    st.set_page_config(layout="centered" if page.url_path == "" else "wide")
    views.common_sidebar(service, user)
    settings = resources.get_settings()
    if user.email in settings.admin_emails and resources.get_admin_service().is_admin(
        user
    ):
        with st.sidebar:
            st.session_state["app_mode"] = "chat" if page.url_path == "" else "admin"
            mode = st.segmented_control(
                tr("mode", user.language),
                ("chat", "admin"),
                key="app_mode",
                format_func=lambda value: tr(f"{value}_mode", user.language),
                on_change=switch_mode,
                selection_mode="single",
            )
            if mode == "admin":
                st.page_link(PAGES / "users.py", label="Users", icon=":material/group:")
                st.page_link(
                    PAGES / "documents.py",
                    label="Documents",
                    icon=":material/description:",
                )
    page.run()
