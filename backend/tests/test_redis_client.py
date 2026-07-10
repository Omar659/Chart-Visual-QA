"""Tests for the shared Redis client resolver — disabled/absent cases (no live Redis in CI)."""
import pytest

import redis_client


@pytest.fixture(autouse=True)
def _reset():
    redis_client.reset()
    yield
    redis_client.reset()


def test_empty_url_means_disabled(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "")
    assert redis_client.client() is None


def test_url_set_but_package_absent_returns_none(monkeypatch):
    """REDIS_URL set but the redis package missing → None (fall back), not a crash."""
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setattr(redis_client, "_redis", None)
    assert redis_client.client() is None


def test_result_is_memoized(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "")
    assert redis_client.client() is None
    # A second call must not re-resolve — flip the env and confirm it's still cached None.
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    assert redis_client.client() is None   # still the memoized result until reset()
    redis_client.reset()
