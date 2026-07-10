"""Fail-open MLflow experiment-tracking helpers for the ChartQA modeling pipeline.

MLflow is an OPTIONAL dependency. Mirroring the backend's guard/chart-gate pattern,
every helper here degrades to a **no-op** (emitting a single warning) when MLflow is
not installed or tracking is turned off — so an eval or training run is *never*
crashed by a tracking failure. The results JSON is still written; only the extra
MLflow record is skipped.

Configuration (env, no in-code absolute paths):
  * ``MLFLOW_ENABLED``        default on; set to 0/false/no/off to disable tracking.
  * ``MLFLOW_TRACKING_URI``   default: MLflow's own ``./mlruns`` file store (we do NOT
                              set a URI unless this is present, so the CWD-relative
                              file store is used — browse it with ``mlflow ui``).
  * ``MLFLOW_EXPERIMENT``     default: the per-caller name passed to ``track_run``
                              (``chartqa-eval`` / ``chartqa-train``); this env var,
                              when set, overrides that name.

Usage::

    with tracking.track_run("blip2-4bit-relaxed-test", tags={"hardware": "cpu"},
                            experiment="chartqa-eval"):
        tracking.log_params({"model": "blip2", "quantization": "4bit"})
        ...
        tracking.log_metrics({"relaxed_accuracy": 0.124})
        tracking.log_artifact("outputs/errors/errors_blip2")
"""

import os
import sys
from contextlib import contextmanager

# --------------------------------------------------------------------------- #
# Guarded optional import — this is the REAL boundary the fail-open tests flip.
# Tests set ``tracking._MLFLOW_IMPORT_OK = False`` to simulate mlflow-absent and
# assert the helpers stay no-ops (never monkeypatch the public helpers themselves).
# --------------------------------------------------------------------------- #
try:
    import mlflow

    _MLFLOW_IMPORT_OK = True
except Exception:  # noqa: BLE001 - any import failure degrades to no-op tracking
    mlflow = None
    _MLFLOW_IMPORT_OK = False

_ENABLED_ENV = "MLFLOW_ENABLED"
_URI_ENV = "MLFLOW_TRACKING_URI"
_EXPERIMENT_ENV = "MLFLOW_EXPERIMENT"
_DEFAULT_EXPERIMENT = "chartqa"

# Deduplicate warnings so a disabled/absent run warns once, not once per helper call.
_warned_keys: set[str] = set()


def _warn(key: str, message: str) -> None:
    """Print a one-time warning (keyed) to stderr; never raise."""
    if key in _warned_keys:
        return
    _warned_keys.add(key)
    print(f"[chartqa.tracking] WARNING: {message}", file=sys.stderr)


def _enabled() -> bool:
    """True only when mlflow imported AND MLFLOW_ENABLED is not a falsy value.

    Read fresh on every call so tests can flip the import flag / env at runtime.
    """
    if not _MLFLOW_IMPORT_OK:
        _warn("absent", "mlflow is not installed; experiment tracking is disabled "
                        "(pip install 'mlflow>=2.14,<3' to enable). Continuing.")
        return False
    flag = os.environ.get(_ENABLED_ENV, "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        _warn("disabled", f"{_ENABLED_ENV}={flag!r}; experiment tracking is off. Continuing.")
        return False
    return True


@contextmanager
def track_run(run_name: str | None = None, tags: dict | None = None,
              experiment: str | None = None):
    """Context manager wrapping one MLflow run. No-op (yields None) when disabled.

    Resolves the tracking URI (only if ``MLFLOW_TRACKING_URI`` is set — otherwise
    MLflow's CWD-relative ``./mlruns`` file store is used) and the experiment name
    (``MLFLOW_EXPERIMENT`` env overrides the ``experiment`` arg). A failure to start
    the run is caught, warned, and downgraded to a no-op — the caller's body still
    runs. Caller exceptions propagate normally and close the run in ``finally``.
    """
    if not _enabled():
        yield None
        return

    active = None
    try:
        uri = os.environ.get(_URI_ENV)
        if uri:
            mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(
            os.environ.get(_EXPERIMENT_ENV) or experiment or _DEFAULT_EXPERIMENT
        )
        active = mlflow.start_run(run_name=run_name, tags=tags)
    except Exception as exc:  # noqa: BLE001 - tracking must never crash the caller
        _warn("start", f"could not start an MLflow run ({exc!r}); continuing untracked.")
        active = None

    try:
        yield active
    finally:
        if active is not None:
            try:
                mlflow.end_run()
            except Exception as exc:  # noqa: BLE001
                _warn("end", f"could not end the MLflow run ({exc!r}).")


def log_params(params: dict) -> None:
    """Log run parameters (``None`` values dropped). No-op when disabled; never raises."""
    if not _enabled():
        return
    clean = {k: v for k, v in params.items() if v is not None}
    if not clean:
        return
    try:
        mlflow.log_params(clean)
    except Exception as exc:  # noqa: BLE001
        _warn("params", f"log_params failed ({exc!r}); continuing.")


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log numeric metrics (non-numeric/``None`` values dropped). No-op when disabled."""
    if not _enabled():
        return
    clean = {
        k: float(v)
        for k, v in metrics.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool)
    }
    if not clean:
        return
    try:
        mlflow.log_metrics(clean, step=step)
    except Exception as exc:  # noqa: BLE001
        _warn("metrics", f"log_metrics failed ({exc!r}); continuing.")


def log_artifact(path: str) -> None:
    """Log a file or a whole directory as run artifacts. No-op when disabled/missing."""
    if not _enabled():
        return
    if not path or not os.path.exists(path):
        return
    try:
        if os.path.isdir(path):
            mlflow.log_artifacts(path)
        else:
            mlflow.log_artifact(path)
    except Exception as exc:  # noqa: BLE001
        _warn("artifact", f"log_artifact failed for {path!r} ({exc!r}); continuing.")
