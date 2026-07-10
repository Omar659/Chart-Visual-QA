"""Tests for Google ID-token verification — the real external boundary
(``google.oauth2.id_token.verify_oauth2_token``) is disabled/mocked, never our own
``verify_google_token``, per the project's testing convention."""
import pytest

import auth


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_ENABLED", True)
    monkeypatch.setattr(auth, "GOOGLE_CLIENT_ID", "test-client-id.apps.googleusercontent.com")


def test_empty_token_rejected():
    assert auth.verify_google_token("") is None


def test_valid_token_returns_claims(monkeypatch):
    claims = {"email": "user@example.com", "sub": "1234567890", "name": "Test User"}
    monkeypatch.setattr(auth.google_id_token, "verify_oauth2_token", lambda *a, **k: claims)
    assert auth.verify_google_token("a.valid.jwt") == claims


def test_expired_or_bad_signature_rejected(monkeypatch):
    def boom(*a, **k):
        raise ValueError("Token expired")
    monkeypatch.setattr(auth.google_id_token, "verify_oauth2_token", boom)
    assert auth.verify_google_token("an.expired.jwt") is None


def test_wrong_audience_rejected(monkeypatch):
    def boom(*a, **k):
        raise ValueError("Wrong audience")
    monkeypatch.setattr(auth.google_id_token, "verify_oauth2_token", boom)
    assert auth.verify_google_token("a.jwt.for.someone.else") is None


def test_verify_passes_configured_audience(monkeypatch):
    seen = {}

    def fake_verify(token, request, audience):
        seen["audience"] = audience
        return {"sub": "1"}

    monkeypatch.setattr(auth.google_id_token, "verify_oauth2_token", fake_verify)
    auth.verify_google_token("a.jwt")
    assert seen["audience"] == "test-client-id.apps.googleusercontent.com"
