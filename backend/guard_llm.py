"""Layer-3 input boundary filter — Llama Guard 3 as a classifier.

See docs/TASK_B_LAYER3.md. This is the **semantic** boundary check that runs after the
cheap Layers 1-2, screening the question for unsafe content / policy / out-of-scope
(jailbreak & prompt-injection stay primarily on Layer 2). It calls a small local guard
model over the OpenAI-compatible HTTP API (``/v1/chat/completions``) — served by both
Ollama (dev) and vLLM (prod), so dev->prod is a ``GUARD_LLM_URL`` change only.

**ON by default.** It only falls back to allowing the request (fail-open) when the guard is
genuinely unavailable — the dependency is missing, or the service is unreachable / slow /
returns junk — and it **logs a WARNING every time it does** so the gap is visible. Set
``GUARD_LLM_ENABLED=0`` to run intentionally without it (silent, no attempt).

Llama Guard returns plain text: ``safe`` or ``unsafe\\n<comma-separated category codes>``.
We map the codes to our ``{unsafe, jailbreak, off_topic}`` categories + a user-safe reason.
"""

from __future__ import annotations

import logging

from env_config import env_bool, env_float, env_str
from guard import GuardResult

log = logging.getLogger("guard.layer3")

try:
    import requests
except Exception:  # noqa: BLE001 — requests not installed -> fail-open
    requests = None

# --- Config (required in .env; loaded from .env by the backend at startup) ---
GUARD_LLM_ENABLED = env_bool("GUARD_LLM_ENABLED")
GUARD_LLM_URL = env_str("GUARD_LLM_URL")
GUARD_LLM_MODEL = env_str("GUARD_LLM_MODEL")
# Set generously in .env to survive a cold model load (~15-18s) so the first request
# actually runs the guard instead of failing open; warm calls are ~3s.
_TIMEOUT = env_float("GUARD_LLM_TIMEOUT")

# Llama Guard hazard code -> (our category, user-safe reason). S99 is our custom off-topic
# category (requires a custom-taxonomy prompt — see docs/TASK_B_LAYER3.md §3/§5).
_CODES = {
    "S1": ("unsafe", "This request involves violent content."),
    "S2": ("unsafe", "This request involves criminal activity."),
    "S3": ("unsafe", "This request involves sexual content."),
    "S4": ("unsafe", "This request involves content harmful to minors."),
    "S6": ("unsafe", "This request involves specialized advice we don't provide."),
    "S9": ("unsafe", "This request involves weapons or mass harm."),
    "S10": ("unsafe", "This request involves hateful content."),
    "S11": ("unsafe", "This request involves self-harm."),
    "S14": ("jailbreak", "That looks like an attempt to abuse the tool."),
    "S99": ("off_topic", "Please ask a question about the chart."),
}


def _warn_fallback(reason: str) -> None:
    log.warning(
        "Layer-3 guard (Llama Guard) UNAVAILABLE: %s — request ALLOWED without Layer-3 "
        "screening. Start Ollama and `ollama pull %s`, or set GUARD_LLM_ENABLED=0 to run "
        "without it intentionally.",
        reason, GUARD_LLM_MODEL,
    )


def _chat(content: str, timeout: float):
    # OpenAI-compatible Chat Completions — served by BOTH Ollama (dev) and vLLM
    # (prod), so dev->prod is a GUARD_LLM_URL change only. See docs/ARCHITECTURE.md.
    resp = requests.post(
        f"{GUARD_LLM_URL}/v1/chat/completions",
        timeout=timeout,
        json={
            "model": GUARD_LLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
            "temperature": 0,
            "max_tokens": 32,  # Llama Guard answers with "safe" / "unsafe\nS..": tiny output
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()


def llm_classify(question: str):
    """Return a GuardResult, or None if the guard is unavailable (fail-open + warn)."""
    if not GUARD_LLM_ENABLED:
        return None
    if requests is None:
        _warn_fallback("the 'requests' package is not installed")
        return None
    try:
        out = _chat(question, _TIMEOUT)
    except Exception as exc:  # noqa: BLE001 — service down / slow / bad payload
        _warn_fallback(f"{type(exc).__name__}: {exc}")
        return None

    if not out or out.lower().startswith("safe"):
        return GuardResult(True)

    lines = out.splitlines()
    codes = lines[1].replace(" ", "").upper().split(",") if len(lines) > 1 else []
    for code in codes:
        if code in _CODES:
            category, reason = _CODES[code]
            return GuardResult(False, category, reason)
    # Flagged unsafe but no recognized code — block conservatively.
    return GuardResult(False, "unsafe", "Blocked by the safety classifier.")


def warmup() -> None:
    """Pre-load the guard model (off the request path) so the first request is warm.

    Called from a background thread at backend boot. No-op when disabled or unavailable.
    """
    if not GUARD_LLM_ENABLED or requests is None:
        return
    try:
        log.info("Warming up Layer-3 guard model %s ...", GUARD_LLM_MODEL)
        _chat("ok", max(_TIMEOUT, 60))  # cold load can take ~15-18s
        log.info("Layer-3 guard model is warm.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Layer-3 guard warmup failed (%s); first request may fail open.", exc)


def is_available() -> bool:
    """Best-effort check that the guard service is reachable (for /api/health / debugging)."""
    if not GUARD_LLM_ENABLED or requests is None:
        return False
    try:
        return requests.get(f"{GUARD_LLM_URL}/v1/models", timeout=2).ok
    except Exception:  # noqa: BLE001
        return False
