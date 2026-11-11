"""Shared ownership predicates and required-row access checks."""

from uuid import UUID

from sqlalchemy import and_, select, true

from atrag.db_models import ChatConversationRow, ChatUserRow
from atrag.schemas.chat.conversations import User


def user_identity(user: User, *, require_active: bool = True):
    return and_(
        ChatUserRow.id == user.id,
        ChatUserRow.issuer == user.issuer,
        ChatUserRow.subject == user.subject,
        ChatUserRow.is_active.is_(True) if require_active else true(),
    )


def required_row(result):
    row = result.mappings().one_or_none()
    if row is None:
        raise PermissionError("Access denied")
    return row


def owned_conversation(
    user: User, conversation_id: UUID, *, require_active: bool = True
):
    return (
        select(ChatConversationRow)
        .join(ChatUserRow)
        .where(
            ChatConversationRow.id == conversation_id,
            user_identity(user, require_active=require_active),
        )
    )
