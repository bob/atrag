"""A native table click keeps the identity of the row shown to the user."""

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
import streamlit as st

from atrag.admin.schemas import UserSummary
from atrag.interface.admin_views import select_user_action


@pytest.mark.parametrize("position", [0, 1])
def test_row_action_selects_the_clicked_account(monkeypatch, position):
    rows = tuple(
        UserSummary(uuid4(), email, datetime.now(UTC), None, True, 0, 0, 0)
        for email in ("first@example.com", "second@example.com")
    )
    state = {"admin_user_action": SimpleNamespace(row=position, label="Deactivate")}
    monkeypatch.setattr(st, "session_state", state)
    select_user_action(rows)
    assert state["admin_pending_user"].id == rows[position].id
    assert state["admin_pending_user"].email == rows[position].email
