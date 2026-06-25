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

from flask import Flask, jsonify, request
from flask_cors import CORS

from chart_check import looks_like_chart
from env_config import env_bool, env_float, env_int
from guard import guard, warmup
from inference import is_mock, run_inference

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

# Allow the Vite dev server (and others) to call the API directly. In dev the
# Vite proxy means same-origin requests, but enabling CORS keeps the API usable
# when called straight from the browser/tools.
CORS(app, resources={r"/api/*": {"origins": "*"}})

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


@app.get("/api/health")
def health():
    return jsonify(status="ok", mock=is_mock())


@app.post("/api/ask")
def ask():
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

    # --- Layer-2/3 guard: toxicity / prompt-injection / PII + Llama Guard (see guard.py) ---
    verdict = guard(question)
    if not verdict.allowed:
        return jsonify(blocked=True, category=verdict.category, reason=verdict.reason)

    # Rule 4: chart gate (CLIP zero-shot, heuristic fallback; see chart_check).
    is_chart, chart_confidence = looks_like_chart(image_bytes)

    start = time.perf_counter()
    answer = run_inference(image_bytes, question)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

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

    return jsonify(
        answer=answer,
        mock=is_mock(),
        is_chart=is_chart,
        chart_confidence=chart_confidence,
        latency_ms=latency_ms,
    )


if __name__ == "__main__":
    # Pre-warm guard models off the request path so the first /api/ask is fast and the
    # Layer-3 guard is ready. Under the debug reloader only the child serves
    # (WERKZEUG_RUN_MAIN=true) — warm there, not in the watcher process.
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Thread(target=warmup, daemon=True).start()
    app.run(host="127.0.0.1", port=env_int("PORT"), debug=True)
