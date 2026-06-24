"""Layer-3 input boundary filter — Llama Guard 3 as a classifier.

See docs/TASK_B_LAYER3.md. This is the **semantic** boundary check that runs after the
cheap Layers 1-2, screening the question for unsafe content / policy / out-of-scope
(jailbreak & prompt-injection stay primarily on Layer 2). It calls a small local guard
model over HTTP — Ollama in dev, vLLM in prod — same code, only ``GUARD_LLM_URL`` changes.

**Off by default and fail-open:** if disabled, the dependency is missing, or the guard
service is unreachable / returns junk, ``llm_classify`` returns ``None`` so the request is
allowed. The app and CI therefore run with no guard server.

Llama Guard returns plain text: ``safe`` or ``unsafe\\n<comma-separated category codes>``.
We map the codes to our ``{unsafe, jailbreak, off_topic}`` categories + a user-safe reason.
"""

from __future__ import annotations

import os

from guard import GuardResult

try:
    import requests
except Exception:  # noqa: BLE001 — requests not installed -> fail-open
    requests = None

# --- Config (env-overridable) ----------------------------------------------
GUARD_LLM_ENABLED = os.environ.get("GUARD_LLM_ENABLED", "0").lower() in ("1", "true", "yes", "on")
GUARD_LLM_URL = os.environ.get("GUARD_LLM_URL", "http://localhost:11434")
GUARD_LLM_MODEL = os.environ.get("GUARD_LLM_MODEL", "llama-guard3:1b")
_TIMEOUT = float(os.environ.get("GUARD_LLM_TIMEOUT", "5"))

# Llama Guard hazard code -> (our category, user-safe reason). Trimmed to what matters here;
# S99 is our custom off-topic category (requires a custom-taxonomy prompt — see TASK_B doc).
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


def llm_classify(question: str):
    """Return a GuardResult, or None if the guard is unavailable (fail-open)."""
    if not GUARD_LLM_ENABLED or requests is None:
        return None
    try:
        resp = requests.post(
            f"{GUARD_LLM_URL}/api/chat",
            timeout=_TIMEOUT,
            json={
                "model": GUARD_LLM_MODEL,
                "messages": [{"role": "user", "content": question}],
                "stream": False,
            },
        )
        resp.raise_for_status()
        out = (resp.json().get("message", {}).get("content") or "").strip()
    except Exception:  # noqa: BLE001 — service down / bad payload -> fail-open
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


def is_available() -> bool:
    """Best-effort check that the guard service is reachable (for /api/health / debugging)."""
    if not GUARD_LLM_ENABLED or requests is None:
        return False
    try:
        return requests.get(f"{GUARD_LLM_URL}/api/tags", timeout=2).ok
    except Exception:  # noqa: BLE001
        return False
