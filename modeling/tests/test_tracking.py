"""Fail-open tests for ``chartqa.tracking`` against a REAL temp MLflow file store.

No model, no network. These exercise the tracking helpers directly and disable
tracking at its **real boundary** (flip ``tracking._MLFLOW_IMPORT_OK`` / the
``MLFLOW_ENABLED`` env), never by monkeypatching the public helpers — mirroring the
fail-open convention in ``backend/test_chart_check.py``.

Run: ``python -m pytest modeling/tests/test_tracking.py`` (needs ``mlflow`` installed).
"""
import os

import pytest

from chartqa import tracking


@pytest.fixture(autouse=True)
def _isolate_tracking(monkeypatch, tmp_path):
    """Fresh warning-dedup + a temp file store so no test touches a real ./mlruns."""
    tracking._warned_keys.clear()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.delenv("MLFLOW_ENABLED", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT", raising=False)
    original = tracking._MLFLOW_IMPORT_OK
    yield
    tracking._MLFLOW_IMPORT_OK = original
    tracking._warned_keys.clear()


def test_run_records_params_and_metrics():
    """A real run lands in the file store with the expected params/metrics; None and
    bool values are dropped (the helpers filter them)."""
    mlflow = pytest.importorskip("mlflow")
    with tracking.track_run("unit-test-run", tags={"hardware": "cpu"},
                            experiment="chartqa-test") as active:
        assert active is not None  # mlflow present + enabled -> a real run
        tracking.log_params({"model": "blip2", "quantization": "4bit", "dropped": None})
        tracking.log_metrics({"relaxed_accuracy": 0.124, "n": 20, "skip_bool": True})

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    runs = mlflow.search_runs(experiment_names=["chartqa-test"])
    assert len(runs) == 1
    row = runs.iloc[0]
    assert row["params.model"] == "blip2"
    assert row["params.quantization"] == "4bit"
    assert "params.dropped" not in runs.columns          # None param dropped
    assert row["metrics.relaxed_accuracy"] == pytest.approx(0.124)
    assert row["metrics.n"] == 20
    assert "metrics.skip_bool" not in runs.columns        # bool metric dropped


def test_fail_open_when_mlflow_absent():
    """With mlflow simulated-absent at the real import boundary, every helper is a
    no-op and the caller body still runs — nothing raises."""
    tracking._MLFLOW_IMPORT_OK = False
    ran = False
    with tracking.track_run("x") as active:
        assert active is None
        ran = True
        tracking.log_params({"a": 1})
        tracking.log_metrics({"b": 2.0})
        tracking.log_artifact(__file__)
    assert ran


def test_disabled_via_env_is_noop(monkeypatch):
    """MLFLOW_ENABLED=0 turns tracking off even when mlflow is installed."""
    monkeypatch.setenv("MLFLOW_ENABLED", "0")
    ran = False
    with tracking.track_run("x") as active:
        assert active is None
        ran = True
        tracking.log_params({"a": 1})
        tracking.log_metrics({"b": 2.0})
    assert ran
