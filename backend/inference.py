"""Inference seam between the webapp and the model team's work.

This is the ONE function the backend calls. For now it returns a mock answer so
the full UI + API can be built and tested end-to-end. When Susanne & Omar's model
is ready, replace the body of ``run_inference`` with the real call — no other
backend or frontend code needs to change.

Contract (see docs/PLAN.md §5):
    run_inference(image_bytes: bytes, question: str) -> str
"""

from __future__ import annotations

import hashlib

# Flip to False (or set env USE_MOCK=0) once a real model is wired in.
USE_MOCK = True

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

    # --- Real model goes here (model team) -------------------------------
    # from .model import predict
    # return predict(image_bytes, question)
    raise NotImplementedError("Real model not wired in yet; set USE_MOCK=True.")


def _mock_answer(image_bytes: bytes, question: str) -> str:
    """Deterministic mock: same question -> same canned answer."""
    digest = hashlib.sha1(question.strip().lower().encode("utf-8")).digest()
    return _MOCK_ANSWERS[digest[0] % len(_MOCK_ANSWERS)]


def is_mock() -> bool:
    """Whether the backend is currently serving mock answers."""
    return USE_MOCK
