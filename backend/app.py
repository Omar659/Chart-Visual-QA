"""Flask backend for Chart-Visual-QA.

Mock-first: the API contract is stable and returns fake answers via
``inference.run_inference`` until the real model lands. See docs/PLAN.md §9.

Endpoints:
    GET  /api/health  -> {"status": "ok", "mock": <bool>}
    POST /api/ask     -> {"answer": <str>, "mock": <bool>, "is_chart": <bool>,
                          "latency_ms": <number>}
                         (multipart/form-data: image=<file>, question=<string>)

``is_chart`` is a cheap Layer-1 heuristic (see chart_check) — a warning signal, not
a hard block: when false, the UI can warn that results may be unreliable.

Run standalone:   python backend/app.py
Or via Flask CLI: flask --app backend/app run --port 5000
Or via the root orchestrator:  python app.py
"""

from __future__ import annotations

import os
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from chart_check import looks_like_chart
from inference import is_mock, run_inference

app = Flask(__name__)

# Allow the Vite dev server (and others) to call the API directly. In dev the
# Vite proxy means same-origin requests, but enabling CORS keeps the API usable
# when called straight from the browser/tools.
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Cap uploads at 10 MB so a huge file can't exhaust memory.
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


def _question_too_weak(question: str) -> bool:
    """Reject only near-empty / junk questions (e.g. "?", "hi").

    Counts "meaningful" characters — letters or digits in ANY language, so CJK
    questions (which have no spaces) and short questions are handled the same way.
    This is a light Layer-1 guard against junk, not real NLP.
    """
    meaningful = sum(1 for c in question if c.isalnum())
    return meaningful < 3


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

    # Cheap "is this a chart?" heuristic — a warning signal, not a block.
    is_chart, _confidence = looks_like_chart(image_bytes)

    start = time.perf_counter()
    answer = run_inference(image_bytes, question)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    return jsonify(answer=answer, mock=is_mock(), is_chart=is_chart, latency_ms=latency_ms)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
