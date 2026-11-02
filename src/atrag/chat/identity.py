"""Authorize claims supplied by the application's trusted OIDC authentication layer.

This boundary does not decode or verify raw tokens. The OIDC adapter must verify
the token signature, audience, expiry and session before passing its claims here.
"""

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    TypeAdapter,
    ValidationError,
    field_validator,
)

Email = Annotated[
    str,
    StringConstraints(
        strict=True, strip_whitespace=True, to_lower=True, pattern=r"^[^@\s]+@[^@\s]+$"
    ),
]
ALLOWLIST = TypeAdapter(frozenset[Email])


class GoogleIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    issuer: Literal["https://accounts.google.com", "accounts.google.com"] = Field(
        alias="iss"
    )
    subject: str = Field(
        alias="sub",
        strict=True,
        min_length=1,
        max_length=255,
        pattern=r"^[\x21-\x7e]+$",
    )
    email: Email
    email_verified: StrictBool

    @field_validator("issuer")
    @classmethod
    def canonical_issuer(cls, value: str) -> Literal["https://accounts.google.com"]:
        return "https://accounts.google.com"


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
