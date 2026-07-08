"""Tests for the best-effort answer cache (real in-memory store; config via module consts)."""
import time

import pytest

import answer_cache as ac


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Clean store + deterministic small config for each test."""
    ac.clear()
    monkeypatch.setattr(ac, "_ENABLED", True)
    monkeypatch.setattr(ac, "_MAX", 3)
    monkeypatch.setattr(ac, "_TTL", 0.0)
    yield
    ac.clear()


def test_miss_then_hit():
    assert ac.get(b"img", "q") is None
    ac.put(b"img", "q", {"answer": "42"})
    assert ac.get(b"img", "q") == {"answer": "42"}


def test_question_is_normalized():
    ac.put(b"img", "What is  the MAX?", {"answer": "9"})
    assert ac.get(b"img", "  what is the max?  ") == {"answer": "9"}


def test_different_image_misses():
    ac.put(b"img1", "q", {"answer": "1"})
    assert ac.get(b"img2", "q") is None


def test_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(ac, "_ENABLED", False)
    ac.put(b"img", "q", {"answer": "x"})
    assert ac.get(b"img", "q") is None


def test_lru_eviction():
    for i in range(4):  # _MAX == 3
        ac.put(f"img{i}".encode(), "q", {"answer": str(i)})
    assert ac.get(b"img0", "q") is None            # oldest evicted
    assert ac.get(b"img3", "q") == {"answer": "3"}


def test_returned_value_is_a_copy():
    ac.put(b"img", "q", {"answer": "1"})
    got = ac.get(b"img", "q")
    got["answer"] = "mutated"
    assert ac.get(b"img", "q") == {"answer": "1"}   # store not affected


def test_ttl_expiry(monkeypatch):
    monkeypatch.setattr(ac, "_TTL", 0.05)
    ac.put(b"img", "q", {"answer": "x"})
    assert ac.get(b"img", "q") == {"answer": "x"}
    time.sleep(0.08)
    assert ac.get(b"img", "q") is None
