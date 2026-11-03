"""The email policy controls admission before a Google user is persisted."""

import pytest
from sqlalchemy import select

from atrag.chat.services.chat import ChatService
from atrag.db_models import ChatUserRow
from atrag.migrations import migrate_database

pytestmark = pytest.mark.integration


def test_verified_google_user_can_enter_without_allowlist(empty_corpus_schema, engine):
    migrate_database(engine)
    service = ChatService(engine)
    claims = {
        "iss": "https://accounts.google.com",
        "sub": "new-google-user",
        "email": "new-user@gmail.com",
        "email_verified": True,
    }
    with pytest.raises(PermissionError):
        service.authenticate(claims, [])
    user = service.authenticate(claims, [], check_allowed_emails=False)
    assert service.get_user(user) == user
    assert user.email == "new-user@gmail.com"
    assert service.authenticate(claims, [user.email]) == user
    with pytest.raises(PermissionError):
        service.authenticate(claims, [], check_allowed_emails=True)
    with pytest.raises(PermissionError):
        service.authenticate(
            {**claims, "email_verified": False}, [], check_allowed_emails=False
        )
    with engine.connect() as connection:
        assert len(connection.execute(select(ChatUserRow)).all()) == 1
