"""Shared ownership predicates and required-row access checks."""

from uuid import UUID

from sqlalchemy import and_, select

from atrag.db_models import ChatConversationRow, ChatUserRow
from atrag.schemas.chat.conversations import User


def user_identity(user: User):
    return and_(
        ChatUserRow.id == user.id,
        ChatUserRow.issuer == user.issuer,
        ChatUserRow.subject == user.subject,
    )


def required_row(result):
    row = result.mappings().one_or_none()
    if row is None:
        raise PermissionError("Access denied")
    return row


def owned_conversation(user: User, conversation_id: UUID):
    return (
        select(ChatConversationRow)
        .join(ChatUserRow)
        .where(ChatConversationRow.id == conversation_id, user_identity(user))
    )
