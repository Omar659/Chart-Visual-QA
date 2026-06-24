"""Make the flat backend modules (app, inference) importable from tests."""

import pathlib
import sys

import pytest

# backend/ is the parent of this tests/ dir.
BACKEND_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(autouse=True)
def _disable_real_llm_guard(monkeypatch):
    """Never hit a real Ollama / LLM during tests. Layer-3 is ON by default in prod, so
    without this every guard() call would reach the running guard service. Tests that
    exercise Layer 3 opt in by monkeypatching GUARD_LLM_ENABLED / llm_classify themselves."""
    import guard_llm

    monkeypatch.setattr(guard_llm, "GUARD_LLM_ENABLED", False)
