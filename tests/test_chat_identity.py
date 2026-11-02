"""Only verified Google identities on the configured allowlist enter the service."""

import pytest

from atrag.chat.identity import parse_google_identity


def claims(**changes):
    return {
        "iss": "https://accounts.google.com",
        "sub": "Google-Subject",
        "email": "Owner@Example.com",
        "email_verified": True,
        **changes,
    }


@pytest.mark.parametrize(
    "issuer", ["accounts.google.com", "https://accounts.google.com"]
)
def test_google_identity_uses_subject_and_canonical_issuer(issuer):
    identity = parse_google_identity(
        claims(iss=issuer, name="Untrusted display name", user_id="untrusted"),
        [" OWNER@example.com "],
    )
    assert (identity.issuer, identity.subject, identity.email) == (
        "https://accounts.google.com",
        "Google-Subject",
        "owner@example.com",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"iss": "https://evil.example"},
        {"iss": "https://accounts.google.com/"},
        {"iss": None},
        {"sub": ""},
        {"sub": " "},
        {"sub": 123},
        {"sub": "é"},
        {"sub": "a" * 256},
        {"email": ""},
        {"email": "invalid"},
        {"email": "other@example.com"},
        {"email_verified": False},
        {"email_verified": "true"},
        {"email_verified": 1},
        {"email_verified": None},
    ],
)
def test_invalid_or_disallowed_claims_are_denied(changes):
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(claims(**changes), ["owner@example.com"])


@pytest.mark.parametrize("missing", ["iss", "sub", "email", "email_verified"])
def test_required_claims_cannot_be_omitted(missing):
    values = claims()
    del values[missing]
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(values, ["owner@example.com"])


@pytest.mark.parametrize("allowlist", [[], "owner@example.com", [None], [1]])
def test_empty_or_malformed_allowlist_denies_access(allowlist):
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(claims(), allowlist)
