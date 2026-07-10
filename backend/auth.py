"""Required Google sign-in for the public API (Phase 3.7 abuse control).

Stateless: the frontend sends a Google-issued **ID token** (from Google Identity
Services — see ``frontend/src/App.jsx``) as ``Authorization: Bearer <token>`` on every
call to a gated endpoint, and this module verifies its signature **on every request** —
no session cookie, no server-side session store, no new Postgres table. This is a
deliberate simplicity trade-off: a real Google account is a much higher bar against
abuse than an anonymous IP (accounts aren't free to mint at scale), and per-request
verification needs zero new data-layer work.

**Toggle, same convention as GUARD_ENABLED**: ``AUTH_ENABLED=0`` (the default) turns
this off entirely — `--dev` / docker-compose / tests never need a real Google Client ID.
Cloud Run sets ``AUTH_ENABLED=1`` + a real ``GOOGLE_CLIENT_ID``.
"""

from __future__ import annotations

import logging

from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token

from env_config import env_bool, env_str

log = logging.getLogger("auth")

AUTH_ENABLED = env_bool("AUTH_ENABLED")
GOOGLE_CLIENT_ID = env_str("GOOGLE_CLIENT_ID")

# Reused across calls: caches Google's signing certs internally (short-lived, refetched
# on expiry by the library) instead of a fresh HTTP round-trip per verification.
_transport = google_requests.Request()


def verify_google_token(token: str) -> dict | None:
    """Return the decoded claims (``email``, ``sub``, ``name``) for a valid Google ID
    token, or ``None`` if it's missing, expired, malformed, or for the wrong audience.

    Never raises — every failure mode the underlying library can produce (bad signature,
    expired, wrong audience, malformed JWT, ...) collapses to "not authenticated" rather
    than a 500, since an invalid token here is ordinary traffic, not a server bug.
    """
    if not token:
        return None
    try:
        return google_id_token.verify_oauth2_token(token, _transport, audience=GOOGLE_CLIENT_ID)
    except ValueError as exc:
        log.info("Google ID token rejected: %s", exc)
        return None
