"""Tests for the remote VLM serving seam in ``model_adapter`` (the ``VLM_URL`` path).

The new code (``_predict_remote``) is exercised as a REAL HTTP round-trip against a REAL
stub server started in a thread — the client (``requests``) is not mocked. Setting
``VLM_URL`` routes around ``_load_model`` entirely, so no GPU/model is needed; that
routing *is* the point of the seam. The in-process branch is covered by disabling the
model **loader** at its real boundary (``_load_model``), per the repo's test convention
(CLAUDE.md) — never by faking ``predict`` itself.
"""
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import model_adapter

# 1x1 PNG so the in-process branch's PIL.Image.open() has real bytes to decode.
_PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


class _StubVLM(BaseHTTPRequestHandler):
    """Minimal /predict stub: records the request body, returns a canned answer."""

    received: dict = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _StubVLM.received = json.loads(self.rfile.read(length) or b"{}")
        body = json.dumps({"answer": "42"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the test output quiet
        pass


@pytest.fixture
def stub_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _StubVLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    host, port = srv.server_address
    try:
        yield f"http://{host}:{port}/predict"
    finally:
        srv.shutdown()


def test_predict_routes_to_remote_when_vlm_url_set(monkeypatch, stub_url):
    monkeypatch.setenv("VLM_URL", stub_url)
    monkeypatch.setenv("VLM_TIMEOUT", "10")

    answer = model_adapter.predict(_PNG_1x1, "  what is the max?  ")

    assert answer == "42"
    # Real round-trip: the stub got the stripped question + the base64 image verbatim.
    assert _StubVLM.received["question"] == "what is the max?"
    assert base64.b64decode(_StubVLM.received["image"]) == _PNG_1x1


def test_predict_stays_in_process_when_vlm_url_empty(monkeypatch):
    monkeypatch.setenv("VLM_URL", "")
    monkeypatch.setenv("QWEN_ANSWER_SUFFIX", " please answer")
    monkeypatch.setenv("QWEN_MAX_NEW_TOKENS", "8")

    # Disable the heavy model LOADER at its real boundary (not predict()).
    captured = {}

    class _FakeChat:
        def chat(self, image, text, max_new_tokens):
            captured["text"] = text
            captured["max_new_tokens"] = max_new_tokens
            return "Answer: local-7"

    monkeypatch.setattr(model_adapter, "_load_model", lambda: _FakeChat())

    answer = model_adapter.predict(_PNG_1x1, "  q  ")

    assert answer == "local-7"  # real "Answer:" stripping ran
    assert captured["text"] == "q please answer"  # real suffix application ran
    assert captured["max_new_tokens"] == 8
