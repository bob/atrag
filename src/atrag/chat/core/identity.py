"""Authorize claims supplied by the application's trusted OIDC authentication layer.

This boundary does not decode or verify raw tokens. The OIDC adapter must verify
the token signature, audience, expiry and session before passing its claims here.
"""

from collections.abc import Mapping

from pydantic import TypeAdapter, ValidationError

from atrag.schemas.chat.identity import Email, GoogleIdentity

ALLOWLIST = TypeAdapter(frozenset[Email])


def parse_google_identity(
    claims: Mapping[str, object], allowed_emails: object
) -> GoogleIdentity:
    """Fail closed; email grants admission, while issuer/subject identify the user."""
    try:
        identity = GoogleIdentity.model_validate(claims)
        allowed = ALLOWLIST.validate_python(allowed_emails)
    except ValidationError:
        raise PermissionError("Access denied") from None
    if not identity.email_verified or identity.email not in allowed:
        raise PermissionError("Access denied")
    return identity
