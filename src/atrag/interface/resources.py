"""Cache shared infrastructure only, never identities, history or agent state."""

import streamlit as st

from atrag.chat.api.execution import LangGraphBackend
from atrag.chat.client.model import create_chat_model
from atrag.chat.services.chat import ChatService
from atrag.chat.services.turns import TurnService
from atrag.corpus import CorpusComposition, resolve_corpus
from atrag.db import create_db_engine
from atrag.settings import load_settings


@st.cache_resource(show_spinner=False)
def get_engine():
    return create_db_engine(load_settings())


@st.cache_resource(show_spinner=False)
def get_model():
    return create_chat_model(load_settings())


def get_chat_service() -> ChatService:
    return ChatService(get_engine())


def get_turn_service() -> TurnService:
    engine = get_engine()
    return TurnService(engine, LangGraphBackend(engine, get_model()))


def get_composition() -> CorpusComposition:
    with get_engine().connect() as connection:
        return resolve_corpus(connection)
