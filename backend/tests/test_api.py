"""Contract tests for the backend API seam (docs/PLAN.md §5).

These lock the request/response shape that the frontend depends on and that the
real model must keep satisfying. Run with: pytest (from the backend/ dir).
"""

import io

import pytest

from app import app as flask_app


@pytest.fixture()
def client():
    flask_app.config.update(TESTING=True)
    return flask_app.test_client()


def _png_bytes(content=b"fake-png-bytes"):
    return io.BytesIO(b"\x89PNG\r\n\x1a\n" + content)


def _ask(client, question="What was revenue in 2024?", image=True):
    data = {}
    if question is not None:
        data["question"] = question
    if image:
        data["image"] = (_png_bytes(), "chart.png")
    return client.post("/api/ask", data=data, content_type="multipart/form-data")


def test_health_ok(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.get_json()
    assert body["status"] == "ok"
    assert isinstance(body["mock"], bool)


def test_ask_happy_path(client):
    res = _ask(client)
    assert res.status_code == 200
    body = res.get_json()
    assert isinstance(body["answer"], str) and body["answer"]
    assert isinstance(body["mock"], bool)
    assert isinstance(body["latency_ms"], (int, float))


def test_ask_is_deterministic(client):
    a = _ask(client, question="Highest year?").get_json()["answer"]
    b = _ask(client, question="Highest year?").get_json()["answer"]
    assert a == b


def test_ask_missing_question(client):
    res = _ask(client, question=None)
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_ask_blank_question(client):
    res = _ask(client, question="   ")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_ask_missing_image(client):
    res = _ask(client, image=False)
    assert res.status_code == 400
    assert "error" in res.get_json()
