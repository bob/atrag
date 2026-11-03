"""Verified Google identities obey the configured email access policy."""

import pytest

from atrag.chat.core.identity import parse_google_identity


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
        {"email_verified": False},
        {"email_verified": "true"},
        {"email_verified": 1},
        {"email_verified": None},
    ],
)
@pytest.mark.parametrize("check_allowed_emails", [True, False])
def test_invalid_claims_are_denied(changes, check_allowed_emails):
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(
            claims(**changes),
            ["owner@example.com"],
            check_allowed_emails=check_allowed_emails,
        )


@pytest.mark.parametrize("missing", ["iss", "sub", "email", "email_verified"])
@pytest.mark.parametrize("check_allowed_emails", [True, False])
def test_required_claims_cannot_be_omitted(missing, check_allowed_emails):
    values = claims()
    del values[missing]
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(
            values, ["owner@example.com"], check_allowed_emails=check_allowed_emails
        )


@pytest.mark.parametrize("allowlist", [[], "owner@example.com", [None], [1]])
def test_empty_or_malformed_allowlist_denies_access(allowlist):
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(claims(), allowlist)


@pytest.mark.parametrize("policy", [{}, {"check_allowed_emails": True}])
def test_allowlist_is_enforced_by_default_and_when_enabled(policy):
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(claims(), ["other@example.com"], **policy)


@pytest.mark.parametrize("allowlist", [[], ["other@example.com"], None, [None], "bad"])
def test_disabled_allowlist_admits_any_verified_google_identity(allowlist):
    identity = parse_google_identity(claims(), allowlist, check_allowed_emails=False)
    assert (identity.issuer, identity.subject, identity.email) == (
        "https://accounts.google.com",
        "Google-Subject",
        "owner@example.com",
    )


@pytest.mark.parametrize("check_allowed_emails", [None, 0, 1, "false", "true", [], {}])
def test_allowlist_switch_requires_a_boolean(check_allowed_emails):
    with pytest.raises(PermissionError, match="Access denied"):
        parse_google_identity(
            claims(), ["owner@example.com"], check_allowed_emails=check_allowed_emails
        )
