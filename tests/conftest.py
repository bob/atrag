import os

import pytest


@pytest.fixture(autouse=True)
def clean_settings_environment(monkeypatch):
    for name in os.environ:
        if name == "ATRAG_TEST_DATABASE_URL":
            continue
        if name.startswith(("ATRAG_", "OPENAI_")) or name == "DATABASE_URL":
            monkeypatch.delenv(name)


@pytest.fixture
def project_root(tmp_path):
    (tmp_path / ".env").write_text("", encoding="utf-8")
    return tmp_path
