"""Per-IP rate limiting for the public path (Phase 3.7 cost/abuse control).

A **fixed-window** limiter: at most ``RATELIMIT_PER_MINUTE`` requests per client IP per
calendar minute. Backed by **Redis** (``INCR`` + ``EXPIRE`` — atomic and shared across
gunicorn workers, so the limit is global, not per-process) when ``REDIS_URL`` is set;
otherwise an in-process counter (right for ``--dev`` / a single worker).

**Fail-open by design:** if the limiter itself errors (Redis blip, bug), it ALLOWS the
request — a broken limiter must never take the API down. It only ever *blocks* on a
confirmed over-limit count. This is the front-line defense against a bot hammering
``/api/ask`` and keeping the GPU warm (see docs/REVIEW_AND_ROADMAP.md §3.7).

Config (.env): ``RATELIMIT_ENABLED``, ``RATELIMIT_PER_MINUTE``.
"""
from __future__ import annotations

import time
from threading import Lock

import redis_client
from env_config import env_bool, env_int

_ENABLED = env_bool("RATELIMIT_ENABLED")
_PER_MINUTE = env_int("RATELIMIT_PER_MINUTE")

_RPREFIX = "ratelimit:"

# In-memory fallback: {(ip, window_start_minute): count}. Pruned opportunistically.
_counts: dict[tuple[str, int], int] = {}
_lock = Lock()


def allow(client_ip: str) -> bool:
    """True if this IP may proceed; False if it has exceeded the per-minute limit."""
    if not _ENABLED or _PER_MINUTE <= 0:
        return True
    try:
        window = int(time.time() // 60)
        r = redis_client.client()
        if r is not None:
            return _redis_allow(r, client_ip, window)
        return _mem_allow(client_ip, window)
    except Exception:  # noqa: BLE001 — a broken limiter must never break the API
        return True


def _redis_allow(r, client_ip: str, window: int) -> bool:
    key = f"{_RPREFIX}{client_ip}:{window}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, 60)  # first hit in this window — set the TTL so keys self-clean
    return count <= _PER_MINUTE


def _mem_allow(client_ip: str, window: int) -> bool:
    with _lock:
        # Prune stale windows so the dict can't grow unbounded across minutes.
        for k in [k for k in _counts if k[1] < window]:
            del _counts[k]
        key = (client_ip, window)
        count = _counts.get(key, 0) + 1
        _counts[key] = count
        return count <= _PER_MINUTE


def reset() -> None:
    """Test hook: clear the in-memory window counts."""
    with _lock:
        _counts.clear()
