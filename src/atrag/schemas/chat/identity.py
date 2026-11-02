"""Validated claims supplied by the trusted Google OIDC adapter."""

from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
)

Email = Annotated[
    str,
    StringConstraints(
        strict=True, strip_whitespace=True, to_lower=True, pattern=r"^[^@\s]+@[^@\s]+$"
    ),
]


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
