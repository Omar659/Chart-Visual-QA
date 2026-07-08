"""Tests for the Prometheus metrics helpers (real prometheus_client when installed)."""
import pytest

import metrics as m


@pytest.mark.skipif(not m._OK, reason="prometheus_client not installed")
def test_render_exposes_metric_names():
    m.count_request("/api/ask", 200)
    m.observe_stage("guard", 0.01)
    m.observe_stage("vlm", 0.5)
    m.count_blocked("toxicity")
    m.count_cache(True)
    m.count_cache(False)
    m.count_vlm()

    body, ctype = m.render()
    text = body.decode()
    for name in (
        "http_requests_total",
        "stage_latency_seconds",
        "blocked_total",
        "answer_cache_hits_total",
        "answer_cache_misses_total",
        "vlm_invocations_total",
    ):
        assert name in text, name
    assert "text/plain" in ctype or "openmetrics" in ctype


def test_helpers_fail_open_when_disabled(monkeypatch):
    # Simulate prometheus_client absent at the module's real flag; helpers no-op, no raise.
    monkeypatch.setattr(m, "_OK", False)
    m.count_request("/x", 500)
    m.observe_stage("stage", 0.1)
    m.count_blocked("pii")
    m.count_cache(True)
    m.count_vlm()
    body, ctype = m.render()
    assert b"disabled" in body
    assert "text/plain" in ctype
