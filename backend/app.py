"""Flask backend for Chart-Visual-QA.

Mock-first: the API contract is stable and returns fake answers via
``inference.run_inference`` until the real model lands. See docs/PLAN.md §9.

Endpoints:
    GET  /api/health  -> {"status": "ok", "mock": <bool>}
    POST /api/ask     -> {"answer": <str>, "mock": <bool>, "is_chart": <bool>,
                          "latency_ms": <number>}
                         or, if the guard blocks the question (HTTP 200):
                         {"blocked": true, "category": <str>, "reason": <str>}
                         (multipart/form-data: image=<file>, question=<string>)

``is_chart`` is a cheap Layer-1 heuristic (see chart_check) — a warning signal, not
a hard block: when false, the UI can warn that results may be unreliable.
The Layer-2 guard (see guard.py) screens the question for toxicity / prompt
injection / PII before the model runs.

Run standalone:   python backend/app.py
Or via Flask CLI: flask --app backend/app run --port 5000
Or via the root orchestrator:  python app.py
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

# Load <repo-root>/.env BEFORE importing modules that read env at import time
# (inference.USE_MOCK, guard.* thresholds, guard_llm.GUARD_LLM_*).
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

import answer_cache
import auth
import budget
import metrics
import ratelimit
import vlm_provider
from chart_check import looks_like_chart
from env_config import env_bool, env_float, env_int, env_str
from guard import guard, warmup
from inference import is_mock, run_inference
from uploads import InvalidImage, sanitize_image

logging.basicConfig(level=logging.INFO)

# Demo toggle: when MOCK_REVEAL is on, mock mode returns the canned answer instead
# of the disclaimer — useful for demoing the full UI. Off keeps Rule 3 (no fake
# numbers, disclaimer only).
MOCK_REVEAL = env_bool("MOCK_REVEAL")

app = Flask(__name__)

# Shown instead of a (fake) answer while the backend is in mock mode, so a
# canned value is never mistaken for a real model prediction.
MOCK_DISCLAIMER = (
    "Mock mode: no model is connected yet, so no answer is produced. "
    "This is a placeholder response for building and testing the app."
)

# CORS: pin to the frontend origin(s) in prod via CORS_ORIGINS (comma-separated); '*'
# is allowed for dev/local tools. In dev the Vite proxy makes requests same-origin anyway.
_CORS_ORIGINS = [o.strip() for o in env_str("CORS_ORIGINS").split(",") if o.strip()] or ["*"]
CORS(app, resources={r"/api/*": {"origins": _CORS_ORIGINS}})


@app.after_request
def _security_headers(resp):
    """Baseline response hardening (defense-in-depth; the prod proxy may add more)."""
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    if request.path.startswith("/api/"):
        metrics.count_request(request.path, resp.status_code)
    return resp

# Cap uploads (MB) so a huge file can't exhaust memory. Keep in sync with the
# frontend's MAX_BYTES.
MAX_UPLOAD_MB = env_float("MAX_UPLOAD_MB")
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)


# Min "meaningful" (alphanumeric) chars a question must have to not be junk.
MIN_QUESTION_ALNUM = env_int("MIN_QUESTION_ALNUM")


def _question_too_weak(question: str) -> bool:
    """Reject only near-empty / junk questions (e.g. "?", "hi").

    Counts "meaningful" characters — letters or digits in ANY language, so CJK
    questions (which have no spaces) and short questions are handled the same way.
    This is a light Layer-1 guard against junk, not real NLP.
    """
    meaningful = sum(1 for c in question if c.isalnum())
    return meaningful < MIN_QUESTION_ALNUM


def _client_ip() -> str:
    """Best-effort client IP for rate limiting. Behind nginx / Cloud Run the real client
    is the left-most entry of X-Forwarded-For; fall back to the socket peer.

    NOTE: the left-most XFF entry is client-controlled, so the per-IP rate limit is
    **best-effort** — a determined bot can rotate a forged X-Forwarded-For to dodge it.
    That's acceptable here because the hard GPU-cost backstop is the IP-independent daily
    **budget breaker** (budget.py): a spoofer still hits the VLM_DAILY_BUDGET 429 wall.
    The limiter is defense-in-depth against casual/accidental floods, not the cost cap."""
    xff = request.headers.get("X-Forwarded-For", "")
    if xff:
        return xff.split(",")[0].strip()
    return request.remote_addr or "unknown"


def _authenticated_user() -> dict | None:
    """Verify the request's Google ID token (Authorization: Bearer <token>), or None.

    Always returns None when AUTH_ENABLED=0 (local/--dev/tests) — callers that require
    a signed-in user do so via ``_require_auth()`` below, which is the actual gate;
    this helper just answers "who, if anyone" for both that gate and the rate limiter.
    """
    if not auth.AUTH_ENABLED:
        return None
    header = request.headers.get("Authorization", "")
    token = header[7:] if header.startswith("Bearer ") else ""
    return auth.verify_google_token(token)


def _require_auth():
    """Return a ready-to-return ``(body, 401)`` if sign-in is required and
    missing/invalid, else None (caller does ``if (r := _require_auth()): return r``)."""
    if not auth.AUTH_ENABLED:
        return None
    if _authenticated_user() is None:
        return jsonify(error="Sign in with Google to use this."), 401
    return None


@app.get("/api/health")
def health():
    return jsonify(status="ok", mock=is_mock())


@app.get("/metrics")
def metrics_endpoint():
    """Prometheus scrape target (fail-open: a short notice if prometheus_client is absent)."""
    body, content_type = metrics.render()
    return Response(body, mimetype=content_type)


@app.get("/api/vlm/warm")
def vlm_warm():
    """Fire-and-forget nudge for the remote VLM (see vlm_provider.py). The frontend
    calls this right after sign-in so a cold RunPod dev pod / Cloud Run instance has a
    head start before the user's first real question. Always returns immediately.

    Gated the same as /api/ask (Phase 3.7: required sign-in) — no reason to warm the
    billed GPU for a visitor who hasn't authenticated."""
    if (resp := _require_auth()) is not None:
        return resp
    if not is_mock():
        vlm_provider.warm()
    return jsonify(status="ok")


@app.post("/api/ask")
def ask():
    # --- Auth (Phase 3.7): required sign-in, before anything else — cheapest reject ---
    if (resp := _require_auth()) is not None:
        return resp

    # --- Rate limit: keyed by the signed-in user when auth is on (more precise than an
    # IP a bot can spoof via X-Forwarded-For), else by client IP (AUTH_ENABLED=0). ---
    user = _authenticated_user()
    rate_key = user["sub"] if user else _client_ip()
    if not ratelimit.allow(rate_key):
        metrics.count_rate_limited()
        return jsonify(error="Too many requests — please slow down and try again shortly."), 429

    question = (request.form.get("question") or "").strip()
    image = request.files.get("image")

    # --- Layer-1 guard: cheap rules, no ML (see docs/PLAN.md §6) ---
    if image is None or image.filename == "":
        return jsonify(error="Please upload an image."), 400
    if not question:
        return jsonify(error="Please type a question."), 400
    if _question_too_weak(question):
        return jsonify(error="Please ask a more specific question."), 400

    image_bytes = image.read()
    if not image_bytes:
        return jsonify(error="Uploaded image is empty."), 400

    # Re-encode the upload from its decoded pixels: rejects non-images and strips any
    # embedded/trailing payload, and yields canonical bytes for the cache key.
    try:
        image_bytes = sanitize_image(image_bytes)
    except InvalidImage:
        return jsonify(error="Uploaded file is not a valid image."), 400

    # Answer cache (real mode only): a repeat (image, question) short-circuits the guard,
    # chart gate and VLM entirely. Never caches the mock disclaimer.
    if not is_mock():
        cached = answer_cache.get(image_bytes, question)
        metrics.count_cache(cached is not None)
        if cached is not None:
            return jsonify(mock=False, cached=True, latency_ms=0.0, **cached)

    # --- Layer-2/3 guard: toxicity / prompt-injection / PII + Llama Guard (see guard.py) ---
    t0 = time.perf_counter()
    verdict = guard(question)
    metrics.observe_stage("guard", time.perf_counter() - t0)
    if not verdict.allowed:
        metrics.count_blocked(verdict.category)
        return jsonify(blocked=True, category=verdict.category, reason=verdict.reason)

    # Rule 4: chart gate (CLIP zero-shot, heuristic fallback; see chart_check).
    t0 = time.perf_counter()
    is_chart, chart_confidence = looks_like_chart(image_bytes)
    metrics.observe_stage("chart_gate", time.perf_counter() - t0)

    if not is_mock():
        # Daily VLM budget breaker (Phase 3.7): refuse *before* touching the GPU once the
        # day's budget is spent — the cache/guard above still work, the bill doesn't grow.
        if budget.over_budget():
            metrics.count_over_budget()
            return jsonify(error="The demo's daily model quota has been reached — "
                                 "please try again tomorrow."), 429
        t0 = time.perf_counter()
        if not vlm_provider.ensure_running(env_float("VLM_TIMEOUT")):
            metrics.observe_stage("vlm_start", time.perf_counter() - t0)
            return jsonify(error="The model service is starting up — try again shortly."), 503
        metrics.observe_stage("vlm_start", time.perf_counter() - t0)

    start = time.perf_counter()
    answer = run_inference(image_bytes, question)
    inference_s = time.perf_counter() - start
    latency_ms = round(inference_s * 1000, 1)
    if not is_mock():
        metrics.observe_stage("vlm", inference_s)
        metrics.count_vlm()
        budget.record()  # count this real invocation against today's budget

    # Rule 3: in mock mode return a disclaimer, never a fake answer — unless the
    # MOCK_REVEAL demo toggle is on, in which case show the canned answer.
    if is_mock() and not MOCK_REVEAL:
        return jsonify(
            disclaimer=MOCK_DISCLAIMER,
            mock=True,
            is_chart=is_chart,
            chart_confidence=chart_confidence,
            latency_ms=latency_ms,
        )

    result = {"answer": answer, "is_chart": is_chart, "chart_confidence": chart_confidence}
    if not is_mock():
        answer_cache.put(image_bytes, question, result)
    return jsonify(mock=is_mock(), latency_ms=latency_ms, **result)


if __name__ == "__main__":
    debug = env_bool("FLASK_DEBUG")
    # Pre-warm guard models off the request path so the first /api/ask is fast and the
    # Layer-3 guard is ready. With the debug reloader only the child serves
    # (WERKZEUG_RUN_MAIN=true) — warm there; without the reloader, warm unconditionally.
    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=warmup, daemon=True).start()
    app.run(host=env_str("HOST"), port=env_int("PORT"), debug=debug)
