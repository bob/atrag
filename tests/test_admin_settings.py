"""Administrator emails have one validated configuration boundary."""

import pytest
from pydantic import ValidationError

from atrag.schemas.settings import Settings
from atrag.settings import load_settings


@pytest.mark.parametrize("value", ["", "  ", [], ()])
def test_no_administrators_by_default(value):
    assert Settings().admin_emails == frozenset()
    assert Settings(admin_emails=value).admin_emails == frozenset()


def test_admin_emails_are_normalized_at_settings_boundary(project_root, monkeypatch):
    (project_root / ".env").write_text(
        "ATRAG_ADMIN_EMAILS= First@Example.com,second@example.com,first@example.com \n"
    )
    assert load_settings(project_root).admin_emails == frozenset(
        {"first@example.com", "second@example.com"}
    )
    monkeypatch.setenv("ATRAG_ADMIN_EMAILS", "third@example.com")
    assert load_settings(project_root).admin_emails == frozenset({"third@example.com"})


@pytest.mark.parametrize("value", ["invalid", "a@b,", "a@b,,c@d", [False], None])
def test_invalid_administrator_configuration_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(admin_emails=value)
