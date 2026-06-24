"""Unit tests for the Layer-3 LLM guard (guard_llm) and its wiring into guard().

Monkeypatch the HTTP layer so nothing here needs Ollama / a guard server.
"""

import guard as guard_mod
import guard_llm as g
from guard import GuardResult


class _Resp:
    """Minimal stand-in for a requests.Response with an Ollama /api/chat body."""

    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": self._content}}


def _fake_requests(content=None, boom=False):
    class _R:
        @staticmethod
        def post(*_a, **_k):
            if boom:
                raise RuntimeError("connection refused")
            return _Resp(content)

    return _R


def _enable(monkeypatch, requests_stub):
    monkeypatch.setattr(g, "GUARD_LLM_ENABLED", True)
    monkeypatch.setattr(g, "requests", requests_stub)


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(g, "GUARD_LLM_ENABLED", False)
    assert g.llm_classify("What was revenue in 2024?") is None


def test_safe_allows(monkeypatch):
    _enable(monkeypatch, _fake_requests("safe"))
    r = g.llm_classify("Which year was highest?")
    assert r is not None and r.allowed


def test_unsafe_blocks_with_category(monkeypatch):
    _enable(monkeypatch, _fake_requests("unsafe\nS10"))
    r = g.llm_classify("<hateful text>")
    assert not r.allowed and r.category == "unsafe" and r.reason


def test_offtopic_custom_category(monkeypatch):
    _enable(monkeypatch, _fake_requests("unsafe\nS99"))
    r = g.llm_classify("write me a poem")
    assert not r.allowed and r.category == "off_topic"


def test_unknown_code_blocks_generic(monkeypatch):
    _enable(monkeypatch, _fake_requests("unsafe\nS77"))
    r = g.llm_classify("...")
    assert not r.allowed and r.category == "unsafe"


def test_service_error_fails_open(monkeypatch):
    _enable(monkeypatch, _fake_requests(boom=True))
    assert g.llm_classify("...") is None


def test_guard_invokes_layer3(monkeypatch):
    # Layers 1-2 unavailable (None) -> not blocked; Layer 3 blocks.
    monkeypatch.setattr(guard_mod, "GUARD_ENABLED", True)
    monkeypatch.setattr(guard_mod, "toxicity_score", lambda q: None)
    monkeypatch.setattr(guard_mod, "injection_score", lambda q: None)
    monkeypatch.setattr(guard_mod, "pii_hits", lambda q: None)
    monkeypatch.setattr(g, "llm_classify", lambda q: GuardResult(False, "unsafe", "blocked"))
    r = guard_mod.guard("anything")
    assert not r.allowed and r.category == "unsafe"


def test_guard_allows_when_layer3_unavailable(monkeypatch):
    monkeypatch.setattr(guard_mod, "GUARD_ENABLED", True)
    monkeypatch.setattr(guard_mod, "toxicity_score", lambda q: None)
    monkeypatch.setattr(guard_mod, "injection_score", lambda q: None)
    monkeypatch.setattr(guard_mod, "pii_hits", lambda q: None)
    monkeypatch.setattr(g, "llm_classify", lambda q: None)  # fail-open
    assert guard_mod.guard("a normal chart question").allowed
