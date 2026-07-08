"""Best-effort answer cache — short-circuits the guard+chart+VLM path on repeat asks.

Keyed by ``(sha256(image_bytes), normalized_question)``. This is the project's
production-efficiency principle in action: run the cheapest check first and skip the
expensive path (guard encoders + Layer-3 LLM + the GPU VLM) when the exact same
``(image, question)`` was already answered.

In-memory LRU with an optional TTL — right for dev / a single worker. A **Redis** backend
(Phase 3.6) is the multi-worker/persistent prod store; this module is the seam it slots
behind. **Fail-open by design:** any cache error is swallowed and treated as a miss, so
the cache can never break a request. Only real answers are cached (never the mock
disclaimer).

Config (.env): ``ANSWER_CACHE_ENABLED``, ``ANSWER_CACHE_MAX`` (entries),
``ANSWER_CACHE_TTL_S`` (0 = never expire).
"""
from __future__ import annotations

import hashlib
import re
import time
from collections import OrderedDict
from threading import Lock

from env_config import env_bool, env_float, env_int

_ENABLED = env_bool("ANSWER_CACHE_ENABLED")
_MAX = env_int("ANSWER_CACHE_MAX")
_TTL = env_float("ANSWER_CACHE_TTL_S")

# key -> (stored_at_epoch, value dict). OrderedDict gives O(1) LRU move/evict.
_store: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
_lock = Lock()
_WS = re.compile(r"\s+")


def _key(image_bytes: bytes, question: str) -> str:
    """Stable key: image content hash + normalized (lowered, whitespace-collapsed) question."""
    norm_q = _WS.sub(" ", question.strip().lower())
    h = hashlib.sha256()
    h.update(hashlib.sha256(image_bytes).digest())
    h.update(b"\x00")
    h.update(norm_q.encode("utf-8"))
    return h.hexdigest()


def get(image_bytes: bytes, question: str) -> dict | None:
    """Return the cached value dict for this (image, question), or None on miss/disabled."""
    if not _ENABLED:
        return None
    try:
        k = _key(image_bytes, question)
        with _lock:
            entry = _store.get(k)
            if entry is None:
                return None
            stored_at, value = entry
            if _TTL > 0 and (time.time() - stored_at) > _TTL:
                _store.pop(k, None)
                return None
            _store.move_to_end(k)  # mark most-recently-used
            return dict(value)
    except Exception:  # noqa: BLE001 — a cache must never break a request
        return None


def put(image_bytes: bytes, question: str, value: dict) -> None:
    """Store a value dict for this (image, question); evicts the oldest past ANSWER_CACHE_MAX."""
    if not _ENABLED:
        return
    try:
        k = _key(image_bytes, question)
        with _lock:
            _store[k] = (time.time(), dict(value))
            _store.move_to_end(k)
            while len(_store) > _MAX:
                _store.popitem(last=False)  # evict least-recently-used
    except Exception:  # noqa: BLE001
        pass


def clear() -> None:
    """Drop all entries (used by tests)."""
    with _lock:
        _store.clear()
