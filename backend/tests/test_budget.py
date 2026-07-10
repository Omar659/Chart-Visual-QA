"""Tests for the daily VLM budget circuit breaker — in-memory fallback path."""
import pytest

import budget


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(budget.redis_client, "client", lambda: None)  # force in-memory
    monkeypatch.setattr(budget, "_BUDGET", 2)
    budget.reset()
    yield
    budget.reset()


def test_under_budget_allows():
    assert budget.over_budget() is False
    budget.record()
    assert budget.over_budget() is False   # 1 < 2


def test_reaching_budget_trips_breaker():
    budget.record()
    budget.record()
    assert budget.over_budget() is True    # 2 >= 2


def test_zero_budget_is_unlimited(monkeypatch):
    monkeypatch.setattr(budget, "_BUDGET", 0)
    for _ in range(100):
        budget.record()
    assert budget.over_budget() is False


def test_budget_is_per_day(monkeypatch):
    day = ["2026-01-01"]
    monkeypatch.setattr(budget, "_today", lambda: day[0])
    budget.record()
    budget.record()
    assert budget.over_budget() is True
    day[0] = "2026-01-02"                  # new day → fresh budget
    assert budget.over_budget() is False


def test_over_budget_fail_open_on_error(monkeypatch):
    def boom():
        raise RuntimeError("redis exploded")
    monkeypatch.setattr(budget.redis_client, "client", boom)
    assert budget.over_budget() is False   # fail-open: don't block the demo
