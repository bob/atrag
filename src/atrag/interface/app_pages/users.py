"""Account status and recorded LLM usage, authorized by the service."""

import streamlit as st

from atrag.interface import admin_views, resources

admin_views.users(resources.get_admin_service(), st.session_state["current_user"])
