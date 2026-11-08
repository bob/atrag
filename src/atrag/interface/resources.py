"""Cache shared infrastructure only, never identities, history or agent state."""

from dataclasses import dataclass
from datetime import datetime

import streamlit as st
from sqlalchemy import select

from atrag.chat.api.execution import LangGraphBackend
from atrag.chat.client.model import create_chat_model
from atrag.chat.services.chat import ChatService
from atrag.chat.services.turns import TurnService
from atrag.db import create_db_engine
from atrag.db_models import CorpusMemberRow, CorpusStateRow, SnapshotRow
from atrag.ingestion.client.openai import create_openai_client
from atrag.settings import load_settings


@st.cache_resource(show_spinner=False)
def get_engine():
    return create_db_engine(load_settings())


@st.cache_resource(show_spinner=False)
def get_model():
    return create_chat_model(load_settings())


@st.cache_resource(show_spinner=False)
def get_embedding_client():
    return create_openai_client(load_settings())


def get_chat_service() -> ChatService:
    return ChatService(get_engine())


def get_turn_service() -> TurnService:
    engine = get_engine()
    return TurnService(
        engine,
        LangGraphBackend(engine, get_model(), client=get_embedding_client()),
    )


@dataclass(frozen=True)
class PublishedDocument:
    source_id: str
    ingested_at: datetime


def get_published_documents() -> tuple[PublishedDocument, ...]:
    query = (
        select(SnapshotRow.source_id, SnapshotRow.ingested_at)
        .join(CorpusMemberRow, CorpusMemberRow.snapshot_id == SnapshotRow.snapshot_id)
        .join(CorpusStateRow, CorpusStateRow.release_id == CorpusMemberRow.release_id)
        .order_by(SnapshotRow.source_id)
    )
    with get_engine().connect() as connection:
        return tuple(PublishedDocument(*row) for row in connection.execute(query))
