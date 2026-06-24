"""Flask backend for Chart-Visual-QA.

Mock-first: the API contract is stable and returns fake answers via
``inference.run_inference`` until the real model lands. See docs/PLAN.md §5.

Endpoints:
    GET  /api/health  -> {"status": "ok", "mock": <bool>}
    POST /api/ask     -> real model:  {"answer": <str>, "mock": false,
                                       "is_chart": <bool>, "latency_ms": <number>}
                         mock mode:   {"disclaimer": <str>, "mock": true,
                                       "is_chart": <bool>, "latency_ms": <number>}
                         (multipart/form-data: image=<file>, question=<string>)

    In mock mode we deliberately return a disclaimer instead of a fake answer so
    nobody mistakes a canned number for a real model prediction. ``is_chart`` is
    a heuristic gate (see chart_check); when false the UI warns that results may
    be unreliable.

Run standalone:   python backend/app.py
Or via Flask CLI: flask --app backend/app run --port 5000
Or via the root orchestrator:  python app.py
"""

from __future__ import annotations

import os
import time

# Demo toggle: when MOCK_REVEAL is set (1/true/yes/on), mock mode returns the
# canned answer instead of the disclaimer — useful for demoing the full UI.
# Default (unset) keeps Rule 3: no fake numbers, disclaimer only.
MOCK_REVEAL = os.environ.get("MOCK_REVEAL", "").lower() in ("1", "true", "yes", "on")

from flask import Flask, jsonify, request
from flask_cors import CORS

from chart_check import looks_like_chart
from inference import is_mock, run_inference

app = Flask(__name__)

# Shown instead of a (fake) answer while the backend is in mock mode, so a
# canned value is never mistaken for a real model prediction.
MOCK_DISCLAIMER = (
    "Mock mode: no model is connected yet, so no answer is produced. "
    "This is a placeholder response for building and testing the app."
)


def _question_too_weak(question: str) -> bool:
    """Reject only near-empty / junk questions (e.g. "?", "hi").

    Counts "meaningful" characters — letters or digits in ANY language, so
    CJK questions (which have no spaces) and short questions are handled the
    same way. This is a light guard against junk, not real NLP.
    """
    meaningful = sum(1 for c in question if c.isalnum())
    return meaningful < 3

# Allow the Vite dev server (and others) to call the API directly. In dev the
# Vite proxy means same-origin requests, but enabling CORS keeps the API usable
# when called straight from the browser/tools.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Cap uploads at 10 MB so a huge file can't exhaust memory.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


@app.get("/api/health")
def health():
    return jsonify(status="ok", mock=is_mock())


@app.post("/api/ask")
def ask():
    question = (request.form.get("question") or "").strip()
    image = request.files.get("image")

    # Rule 1: no image -> error.
    if image is None or image.filename == "":
        return jsonify(error="Please upload an image."), 400
    # Rule 2: question too weak -> error.
    if not question:
        return jsonify(error="Please type a question."), 400
    if _question_too_weak(question):
        return jsonify(error="Please ask a more specific question."), 400

    image_bytes = image.read()
    if not image_bytes:
        return jsonify(error="Uploaded image is empty."), 400

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
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
