"""Queries for locked turn budgets, saved citations and fragment addresses."""

from sqlalchemy import select, update

from atrag.chat.db.access import owned_conversation, required_row
from atrag.db_models import ChatTurnRow, FragmentRow


def lock_turn(connection, user, conversation_id, turn_id):
    required_row(connection.execute(owned_conversation(user, conversation_id)))
    return required_row(
        connection.execute(
            select(ChatTurnRow)
            .where(
                ChatTurnRow.id == turn_id,
                ChatTurnRow.conversation_id == conversation_id,
            )
            .with_for_update()
        )
    )


def save_evidence(connection, turn_id, *, sources, retrieval_tokens, embedding_usage):
    connection.execute(
        update(ChatTurnRow)
        .where(ChatTurnRow.id == turn_id)
        .values(
            sources=sources,
            retrieval_tokens=retrieval_tokens,
            embedding_usage=embedding_usage,
        )
    )


def citation_sources(connection, conversation_id, turn_id):
    return connection.scalar(
        select(ChatTurnRow.sources).where(
            ChatTurnRow.id == turn_id,
            ChatTurnRow.conversation_id == conversation_id,
        )
    )


def fragment_unit_id(connection, address):
    return connection.scalar(
        select(FragmentRow.legal_unit_id).where(
            FragmentRow.snapshot_id == address.snapshot_id,
            FragmentRow.fragment_id == address.block_id,
        )
    )
