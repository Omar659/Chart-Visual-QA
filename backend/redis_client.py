"""Shared, fail-open Redis connection for the backend (Phase 3.6 data layer).

ONE place so the answer cache, the rate limiter, and the VLM budget breaker share a
single client and a single "is Redis available" decision — instead of three modules
each opening their own connection. ``REDIS_URL`` empty ⇒ Redis is off everywhere and
every consumer uses its in-memory fallback (right for ``--dev`` / a single worker).

**Fail-open by design, same spirit as the guards and the answer cache:** Redis is an
optimization (shared cache/counters across gunicorn workers + restart survival), never a
hard dependency. Missing package, empty URL, or an unreachable server ⇒ ``client()``
returns ``None`` and callers fall back; a per-call error inside a consumer is that
consumer's problem to swallow. The connection is resolved once and memoized.

Config (.env): ``REDIS_URL`` (empty = disabled; compose sets ``redis://redis:6379/0``).
"""
from __future__ import annotations

import logging

from env_config import env_str

log = logging.getLogger("redis")

try:
    import redis as _redis
except Exception:  # noqa: BLE001 — package absent ⇒ Redis simply off
    _redis = None

_client = None
_resolved = False


def client():
    """Return a live Redis client, or ``None`` if Redis is disabled/unavailable.

    Resolved once (ping-checked) and memoized. Consumers still wrap their own ops in
    try/except and fall back — a server that dies *after* this succeeds must not break a
    request.
    """
    global _client, _resolved
    if _resolved:
        return _client
    _resolved = True

    url = env_str("REDIS_URL").strip()
    if not url:
        return None
    if _redis is None:
        log.warning("REDIS_URL is set but the 'redis' package isn't installed; "
                    "using in-memory fallbacks.")
        return None
    try:
        c = _redis.Redis.from_url(
            url, socket_connect_timeout=2, socket_timeout=2, decode_responses=True
        )
        c.ping()
        _client = c
        log.info("Connected to Redis at %s", url)
    except Exception as exc:  # noqa: BLE001 — unreachable ⇒ fall back, don't crash boot
        log.warning("Redis at %s is unreachable (%s); using in-memory fallbacks.", url, exc)
        _client = None
    return _client


def reset() -> None:
    """Test hook: forget the memoized client so the next ``client()`` re-resolves."""
    global _client, _resolved
    _client = None
    _resolved = False
