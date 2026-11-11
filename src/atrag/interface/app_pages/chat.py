"""The selected conversation survives navigation to administrative pages."""

import streamlit as st

from atrag.interface import resources, views

user = st.session_state["current_user"]
service = resources.get_chat_service()
conversation = views.sidebar(service, user)
views.chat(service, user, conversation.id)
