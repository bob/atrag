"""Authorize claims supplied by the application's trusted OIDC authentication layer.

This boundary does not decode or verify raw tokens. The OIDC adapter must verify
the token signature, audience, expiry and session before passing its claims here.
"""

from collections.abc import Mapping

from pydantic import StrictBool, TypeAdapter, ValidationError

from atrag.schemas.chat.identity import Email, GoogleIdentity

ALLOWLIST = TypeAdapter(frozenset[Email])
CHECK_ALLOWED_EMAILS = TypeAdapter(StrictBool)


def parse_google_identity(
    claims: Mapping[str, object],
    allowed_emails: object,
    *,
    check_allowed_emails: object = True,
) -> GoogleIdentity:
    """Validate Google claims and enforce the optional email allowlist."""
    try:
        identity = GoogleIdentity.model_validate(claims)
        check_allowed = CHECK_ALLOWED_EMAILS.validate_python(check_allowed_emails)
        allowed = ALLOWLIST.validate_python(allowed_emails) if check_allowed else ()
    except ValidationError:
        raise PermissionError("Access denied") from None
    if not identity.email_verified or (check_allowed and identity.email not in allowed):
        raise PermissionError("Access denied")
    return identity
