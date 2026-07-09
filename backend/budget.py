"""Daily VLM budget circuit breaker (Phase 3.7 cost control).

The GPU VLM is the only real cost in the stack. This caps how many **real** VLM answers
the public demo produces per UTC day: once ``VLM_DAILY_BUDGET`` is reached, ``/api/ask``
short-circuits with **HTTP 429** *before* touching the GPU, while the cheap paths keep
working — the answer cache still serves repeats, the guard still screens, mock mode is
unaffected. **The demo degrades; the bill doesn't.**

Counts only *successful* real invocations (``record()`` is called after a VLM answer),
so a burst of guard-blocked or cached requests never eats the budget. Backed by **Redis**
(a per-day key shared across workers) when available, else an in-process counter.

**Fail-open by design:** if the counter errors, ``over_budget()`` returns ``False``
(allow) — a broken breaker must not take the demo down; the provider-level spend cap
(§3.7) is the hard backstop. ``VLM_DAILY_BUDGET=0`` disables the breaker (unlimited).

Config (.env): ``VLM_DAILY_BUDGET`` — a COUNT of real VLM answers/day, NOT a dollar
amount (0 = unlimited). This module has no idea what a GPU-second costs; it's an
app-level soft cap on request volume. For an actual $ ceiling, set a GCP Billing
Budget + alert on the project (see docs/REVIEW_AND_ROADMAP.md §3.7) — that's the only
thing here that looks at money.
"""
from __future__ import annotations

import time
from threading import Lock

import redis_client
from env_config import env_int

_BUDGET = env_int("VLM_DAILY_BUDGET")

_RPREFIX = "vlm_budget:"

# In-memory fallback: {utc_day: count}.
_counts: dict[str, int] = {}
_lock = Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def over_budget() -> bool:
    """True if today's real-VLM count has already reached the daily budget."""
    if _BUDGET <= 0:
        return False
    try:
        day = _today()
        r = redis_client.client()
        if r is not None:
            raw = r.get(f"{_RPREFIX}{day}")
            used = int(raw) if raw else 0
        else:
            with _lock:
                used = _counts.get(day, 0)
        return used >= _BUDGET
    except Exception:  # noqa: BLE001 — a broken breaker must not block the demo
        return False


def record() -> None:
    """Count one successful real VLM invocation against today's budget."""
    if _BUDGET <= 0:
        return
    try:
        day = _today()
        r = redis_client.client()
        if r is not None:
            key = f"{_RPREFIX}{day}"
            count = r.incr(key)
            if count == 1:
                r.expire(key, 60 * 60 * 48)  # keep ~2 days so the key self-cleans
            return
        with _lock:
            _counts[day] = _counts.get(day, 0) + 1
    except Exception:  # noqa: BLE001
        pass


def reset() -> None:
    """Test hook: clear the in-memory day counts."""
    with _lock:
        _counts.clear()
