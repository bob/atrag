"""Shared factories own infrastructure, not a user's mutable chat state."""

from unittest.mock import MagicMock, Mock

from atrag.chat.api.execution import LangGraphBackend
from atrag.interface import resources
from atrag.schemas.settings import Settings


def test_resources_share_clients(monkeypatch):
    resources.get_engine.clear()
    resources.get_model.clear()
    engine, model = MagicMock(), Mock()
    create_engine, create_model = Mock(return_value=engine), Mock(return_value=model)
    monkeypatch.setattr(resources, "load_settings", Mock(return_value=Settings()))
    monkeypatch.setattr(resources, "create_db_engine", create_engine)
    monkeypatch.setattr(resources, "create_chat_model", create_model)
    try:
        first, second = resources.get_chat_service(), resources.get_chat_service()
        assert first is not second
        assert first.engine is second.engine is engine
        a, b = resources.get_turn_service(), resources.get_turn_service()
        assert a is not b
        assert a.backend is not b.backend
        assert isinstance(a.backend, LangGraphBackend)
        assert isinstance(b.backend, LangGraphBackend)
        assert a.backend.model is b.backend.model is model
        create_engine.assert_called_once_with(Settings())
        create_model.assert_called_once_with(Settings())
    finally:
        resources.get_engine.clear()
        resources.get_model.clear()
