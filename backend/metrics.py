"""Prometheus metrics for the backend — fail-open.

If ``prometheus_client`` isn't installed, every helper is a no-op and ``/metrics``
returns a short notice, so observability is never a hard dependency (same spirit as the
guards and the answer cache). When it IS installed, the backend exposes request counts,
per-stage latency histograms, guard-block reasons, answer-cache hit/miss, and VLM
invocation counts — enough to see **which layer spends the time** and **how often the
expensive VLM path fires** (= cost).
"""
from __future__ import annotations

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
    _OK = True
except Exception:  # noqa: BLE001 — prometheus_client absent -> no-op metrics
    _OK = False

if _OK:
    _REQUESTS = Counter("http_requests_total", "API requests", ["route", "status"])
    _STAGE = Histogram("stage_latency_seconds", "Per-request-stage latency", ["stage"])
    _BLOCKED = Counter("blocked_total", "Requests blocked by the guard", ["reason"])
    _CACHE_HITS = Counter("answer_cache_hits_total", "Answer-cache hits")
    _CACHE_MISSES = Counter("answer_cache_misses_total", "Answer-cache misses")
    _VLM = Counter("vlm_invocations_total", "Real VLM inference calls (the cost proxy)")
    # A guard layer silently failing open in prod is a silent security downgrade — it must
    # be visible (alertable), not just a log line (see docs/REVIEW_AND_ROADMAP.md §3.5/3.7).
    _GUARD_FAIL_OPEN = Counter("guard_fail_open_total", "Guard layer failed open (allowed "
                              "without screening)", ["layer"])
    # Cost/abuse controls firing (§3.7): a per-IP rate-limit rejection, or the daily VLM
    # budget breaker tripping before the GPU is touched.
    _RATE_LIMITED = Counter("rate_limited_total", "Requests rejected by the per-IP rate limit")
    _OVER_BUDGET = Counter("vlm_over_budget_total", "Requests refused by the daily VLM budget breaker")


def count_request(route: str, status: int) -> None:
    if _OK:
        _REQUESTS.labels(route, str(status)).inc()


def observe_stage(stage: str, seconds: float) -> None:
    """Record how long a pipeline stage took (guard / chart_gate / vlm)."""
    if _OK:
        _STAGE.labels(stage).observe(seconds)


def count_blocked(reason: str) -> None:
    if _OK:
        _BLOCKED.labels(reason or "unknown").inc()


def count_cache(hit: bool) -> None:
    if _OK:
        (_CACHE_HITS if hit else _CACHE_MISSES).inc()


def count_vlm() -> None:
    if _OK:
        _VLM.inc()


def count_guard_fail_open(layer: str) -> None:
    """A guard layer allowed a request without screening (dependency/service missing)."""
    if _OK:
        _GUARD_FAIL_OPEN.labels(layer or "unknown").inc()


def count_rate_limited() -> None:
    if _OK:
        _RATE_LIMITED.inc()


def count_over_budget() -> None:
    if _OK:
        _OVER_BUDGET.inc()


def render() -> tuple[bytes, str]:
    """Return ``(body, content_type)`` for the /metrics response."""
    if _OK:
        return generate_latest(), CONTENT_TYPE_LATEST
    return (b"# prometheus_client not installed; metrics disabled\n",
            "text/plain; charset=utf-8")
