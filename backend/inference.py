"""Inference seam between the webapp and the model team's work.

This is the ONE place the backend goes through to get an answer. For now it
returns a deterministic mock so the full UI + API can be built and tested
end-to-end. When Susanne & Omar's model is ready, they implement
``model_adapter.predict`` and flip the mock off — no backend or frontend code
here needs to change.

Contract (see docs/PLAN.md §5):
    run_inference(image_bytes: bytes, question: str) -> str

Switching to the real model:
    1. Implement ``predict(image_bytes, question) -> str`` in model_adapter.py.
    2. Run the backend with USE_MOCK=0 (env var) — no code edit required.
"""

from __future__ import annotations

import hashlib
import os
import time

# Default to mock. Override at runtime with the USE_MOCK env var, e.g.
#   USE_MOCK=0 python app.py        # use the real model
# Accepts 0/1, false/true, no/yes, off/on (case-insensitive).
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _env_use_mock(default: bool = True) -> bool:
    raw = os.environ.get("USE_MOCK")
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    return default


USE_MOCK = _env_use_mock()

# Artificial delay (seconds) for the mock so the UI's loading state is visible.
# Real model latency replaces this once wired in. Override with MOCK_DELAY_S.
MOCK_DELAY_S = float(os.environ.get("MOCK_DELAY_S", "1.2"))

# Canned answers the mock picks from. Deterministic per question so the same
# question always yields the same answer (stable for demos and tests).
_MOCK_ANSWERS = ["4.2B", "2018", "37%", "Yes", "No", "About 1,200", "Q3", "12.5%"]


def run_inference(image_bytes: bytes, question: str) -> str:
    """Return a short answer (1-10 words) for a chart image + question.

    Args:
        image_bytes: Raw bytes of the uploaded chart image.
        question: The natural-language question about the chart.

    Returns:
        A short string answer.
    """
    if USE_MOCK:
        return _mock_answer(image_bytes, question)
    return _real_answer(image_bytes, question)


def _real_answer(image_bytes: bytes, question: str) -> str:
    """Call the model team's adapter. Imported lazily so the backend boots
    in mock mode without any heavy ML dependencies installed."""
    from model_adapter import predict  # local import on purpose

    return predict(image_bytes, question)


def _mock_answer(image_bytes: bytes, question: str) -> str:
    """Deterministic mock: same question -> same canned answer.

    Sleeps for ``MOCK_DELAY_S`` to emulate model latency so the frontend's
    "Processing model…" state is actually visible during demos.
    """
    if MOCK_DELAY_S > 0:
        time.sleep(MOCK_DELAY_S)
    digest = hashlib.sha1(question.strip().lower().encode("utf-8")).digest()
    return _MOCK_ANSWERS[digest[0] % len(_MOCK_ANSWERS)]


def is_mock() -> bool:
    """Whether the backend is currently serving mock answers."""
    return USE_MOCK
