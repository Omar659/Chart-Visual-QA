"""Tests for the per-IP rate limiter — in-memory fallback path (no Redis in CI).

Disables Redis at the real boundary (``redis_client.client -> None``) so the REAL
in-memory code path runs, per the project's testing convention — never monkeypatch the
unit under test with a constant.
"""
import pytest

import ratelimit


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    # Force the in-memory path: no Redis available.
    monkeypatch.setattr(ratelimit.redis_client, "client", lambda: None)
    monkeypatch.setattr(ratelimit, "_ENABLED", True)
    monkeypatch.setattr(ratelimit, "_PER_MINUTE", 3)
    ratelimit.reset()
    yield
    ratelimit.reset()


def test_allows_up_to_the_limit():
    assert [ratelimit.allow("1.2.3.4") for _ in range(3)] == [True, True, True]


def test_blocks_over_the_limit():
    for _ in range(3):
        ratelimit.allow("1.2.3.4")
    assert ratelimit.allow("1.2.3.4") is False


def test_limit_is_per_ip():
    for _ in range(3):
        ratelimit.allow("1.1.1.1")
    assert ratelimit.allow("2.2.2.2") is True  # a different IP has its own budget


def test_disabled_always_allows(monkeypatch):
    monkeypatch.setattr(ratelimit, "_ENABLED", False)
    assert all(ratelimit.allow("1.2.3.4") for _ in range(10))


def test_zero_limit_means_unlimited(monkeypatch):
    monkeypatch.setattr(ratelimit, "_PER_MINUTE", 0)
    assert all(ratelimit.allow("1.2.3.4") for _ in range(10))


def test_window_rolls_over(monkeypatch):
    """A new calendar minute resets the count (windows are keyed by minute)."""
    now = [1_000_000]
    monkeypatch.setattr(ratelimit.time, "time", lambda: now[0])
    for _ in range(3):
        ratelimit.allow("1.2.3.4")
    assert ratelimit.allow("1.2.3.4") is False   # exhausted this window
    now[0] += 61                                  # next minute
    assert ratelimit.allow("1.2.3.4") is True     # fresh budget


def test_fail_open_on_error(monkeypatch):
    """If the limiter backend raises, the request is ALLOWED (never break the API)."""
    def boom():
        raise RuntimeError("redis exploded")
    monkeypatch.setattr(ratelimit.redis_client, "client", boom)
    assert ratelimit.allow("1.2.3.4") is True
