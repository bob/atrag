"""Native pages preserve chat selection and enforce server-side permissions."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import streamlit as st
from sqlalchemy import update
from streamlit.testing.v1 import AppTest

from atrag.db_models import ChatUserRow
from atrag.ingestion.services.index import index_package
from atrag.ingestion.services.packages import build_package
from atrag.interface import resources

pytestmark = pytest.mark.integration
pytest_plugins = ["test_admin"]
APP = Path(__file__).resolve().parents[2] / "streamlit_app.py"


@pytest.fixture
def admin_app(administration, engine, monkeypatch):
    admin, chat, actor, reader = administration
    claims = {
        "is_logged_in": True,
        "iss": actor.issuer,
        "sub": actor.subject,
        "email": actor.email,
        "email_verified": True,
    }
    monkeypatch.setattr(
        st, "user", SimpleNamespace(get=claims.get, to_dict=lambda: claims)
    )
    monkeypatch.setattr(resources, "get_engine", lambda: engine)
    monkeypatch.setattr(resources, "get_settings", lambda: admin.settings)
    app = AppTest.from_file(str(APP), default_timeout=20)
    app.secrets["access"] = {"allowed_emails": [actor.email, reader.email]}
    return app, admin, chat, actor, reader, claims


def test_admin_navigation_status_and_login_reruns(admin_app):
    app, _, chat, actor, _, _ = admin_app
    conversation = chat.create_conversation(actor)
    app.query_params["conversation"] = str(conversation.id)
    app.run()
    logged_in = chat.get_user(actor).last_login_at
    assert logged_in is not None
    app.get("segmented_control")[0].set_value("admin").run()
    assert not app.exception
    assert app.subheader[0].value == "Users"
    assert len(app.dataframe[0].value) == 2
    assert list(app.dataframe[0].value["Actions"]) == ["Deactivate", "Deactivate"]
    assert not any(widget.key == "admin_selected_user" for widget in app.selectbox)
    app.switch_page("src/atrag/interface/app_pages/documents.py").run()
    assert not app.exception
    assert app.subheader[0].value == "Documents"
    app.get("segmented_control")[0].set_value("chat").run()
    assert not app.exception
    assert app.selectbox(key="conversation").value == str(conversation.id)
    assert chat.get_user(actor).last_login_at == logged_in


def test_direct_admin_page_denies_reader_and_new_session_updates_login(admin_app):
    app, _, chat, actor, reader, claims = admin_app
    app.run()
    first_login = chat.get_user(actor).last_login_at
    new_session = AppTest.from_file(str(APP), default_timeout=20)
    new_session.secrets["access"] = app.secrets["access"]
    new_session.run()
    assert chat.get_user(actor).last_login_at > first_login
    claims.update(sub=reader.subject, email=reader.email)
    app.run().switch_page("src/atrag/interface/app_pages/users.py").run()
    assert not app.exception
    assert app.error[0].value == "Access denied"
    assert not app.dataframe
    app.switch_page("src/atrag/interface/app_pages/documents.py").run()
    assert app.error[0].value == "Access denied"


def test_document_details_keep_table_when_source_disappears(
    admin_app, prepared_document, engine, monkeypatch, tmp_path
):
    app, admin, _, actor, _, _ = admin_app
    data, parsed, chunked, embedded = prepared_document
    package = build_package(parsed, chunked, embedded, data, tmp_path / "package")
    index_package(package, engine, admin.settings.data_dir)
    app.run().switch_page("src/atrag/interface/app_pages/documents.py").run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    assert len(app.get("download_button")) == 1
    assert app.get("link_button")
    version = admin.list_documents(actor)[0].versions[0]
    (admin.settings.data_dir / version.local_path).unlink()
    app.run()
    assert not app.exception
    assert len(app.dataframe[0].value) == 1
    assert not app.get("download_button")
    assert app.warning[0].value == "Source file unavailable"


def test_user_filter_and_page_reset(admin_app):
    app, _, _, _, _, _ = admin_app
    app.run().switch_page("src/atrag/interface/app_pages/users.py").run()
    app.number_input(key="admin_user_page").set_value(2).run()
    assert len(app.dataframe[0].value) == 0
    app.text_input(key="admin_email_search").set_value("reader").run()
    assert app.number_input(key="admin_user_page").value == 1
    assert list(app.dataframe[0].value["Email"]) == ["reader@example.com"]
    app.get_by_key("admin_status").set_value("inactive").run()
    assert len(app.dataframe[0].value) == 0


def confirmation_page(service, actor, row):
    from atrag.interface.admin_views import show_user_confirmation

    show_user_confirmation(service, actor, row)


def test_user_action_requires_confirmation_and_can_be_cancelled(administration):
    admin, _, actor, reader = administration
    row = admin.list_users(actor, search="reader")[0]

    def open_dialog(target):
        return AppTest.from_function(
            confirmation_page, args=(admin, actor, target)
        ).run()

    dialog = open_dialog(row)
    assert not dialog.exception
    assert reader.email in dialog.text[0].value
    assert admin.list_users(actor, active=False) == ()
    dialog.button(key="admin_cancel_status").click().run()
    assert admin.list_users(actor, active=False) == ()
    dialog = open_dialog(row)
    dialog.button(key="admin_confirm_status").click().run()
    blocked = admin.list_users(actor, active=False)[0]
    assert blocked.id == reader.id
    dialog = open_dialog(blocked)
    dialog.button(key="admin_confirm_status").click().run()
    assert admin.list_users(actor, active=False) == ()
    own_row = admin.list_users(actor, search="admin")[0]
    assert open_dialog(own_row).button(key="admin_confirm_status").disabled


def test_status_confirmation_rechecks_administrator_access(administration, engine):
    admin, chat, actor, reader = administration
    row = admin.list_users(actor, search="reader")[0]
    dialog = AppTest.from_function(confirmation_page, args=(admin, actor, row)).run()
    with engine.begin() as connection:
        connection.execute(
            update(ChatUserRow)
            .where(ChatUserRow.id == actor.id)
            .values(is_active=False)
        )
    dialog.button(key="admin_confirm_status").click().run()
    assert not dialog.exception
    assert dialog.error[0].value == "Access denied"
    assert chat.get_user(reader).is_active
