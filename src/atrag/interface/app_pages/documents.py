"""Loaded logical documents and their immutable source versions."""

import streamlit as st

from atrag.interface import admin_views, resources

admin_views.documents(resources.get_admin_service(), st.session_state["current_user"])
