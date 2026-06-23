"""Flask backend for Chart-Visual-QA.

Mock-first: the API contract is stable and returns fake answers via
``inference.run_inference`` until the real model lands. See docs/PLAN.md §5.

Endpoints:
    GET  /api/health  -> {"status": "ok", "mock": <bool>}
    POST /api/ask     -> {"answer": <str>, "mock": <bool>, "latency_ms": <number>}
                         (multipart/form-data: image=<file>, question=<string>)

Run standalone:   python backend/app.py
Or via Flask CLI: flask --app backend/app run --port 5000
Or via the root orchestrator:  python app.py
"""

from __future__ import annotations

import os
import time

from flask import Flask, jsonify, request
from flask_cors import CORS

from inference import is_mock, run_inference

app = Flask(__name__)

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

    if not question:
        return jsonify(error="Missing 'question'."), 400
    if image is None or image.filename == "":
        return jsonify(error="Missing 'image' file."), 400

    image_bytes = image.read()
    if not image_bytes:
        return jsonify(error="Uploaded image is empty."), 400

    start = time.perf_counter()
    answer = run_inference(image_bytes, question)
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    return jsonify(answer=answer, mock=is_mock(), latency_ms=latency_ms)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
