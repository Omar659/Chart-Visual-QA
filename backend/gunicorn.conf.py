"""gunicorn config for the Flask backend (see Dockerfile's CMD).

One worker, multiple threads: the guard/CLIP models are ~1-2GB loaded once per
*process* — threads share that memory, separate worker processes would each load
their own copy (Cloud Run's backend deploy is --memory=2Gi, too tight for more than
one copy). Concurrency comes from ``threads``, not ``workers``.

The warmup hook lives here (post_worker_init), not in app.py's module body, so it
fires exactly when gunicorn actually starts serving a worker — never on a bare
``from app import app`` (which backend/tests/test_api.py does at collection time;
starting a background thread that tries to load real ML models on every test run
would contradict the project's "tests run without any guard models" convention).
"""

from __future__ import annotations

import threading

workers = 1
threads = 8
timeout = 180  # >= VLM_TIMEOUT (120s default) + guard/chart-gate overhead, with margin
graceful_timeout = 30


def post_worker_init(worker):  # noqa: ARG001 — gunicorn's hook signature
    from guard import warmup

    threading.Thread(target=warmup, daemon=True).start()
