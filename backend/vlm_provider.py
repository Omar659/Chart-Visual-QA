"""Provider-agnostic VLM warm-up / readiness for the remote serving path.

Only relevant when VLM_URL is set (model_adapter.py's remote mode). ``VLM_PROVIDER``
picks the strategy — this is the ONE place that decides "is it running, should I start
it": both ``GET /api/vlm/warm`` (called on page load) and ``POST /api/ask`` (before
calling ``model_adapter.predict``) go through the same ``warm()`` / ``ensure_running()``
here, so the two call sites never diverge into two different ideas of "running".

    VLM_PROVIDER=none      (default) VLM_URL is assumed always-on (a container, a
                            Cloud Run GPU service with min-instances>0, ...). No-op.
    VLM_PROVIDER=cloudrun  Cloud Run's own request-triggered autoscaling already does
                            "start if not running" — the real request survives a cold
                            start via the generous VLM_TIMEOUT. warm() just sends a
                            best-effort GET to nudge the autoscaler a little earlier.
    VLM_PROVIDER=runpod    Dev only. There is no autoscaler: warm() provisions a fresh
                            pod via `scripts/runpod_up.py --no-wait` in a background
                            thread and updates VLM_URL in-process once it's ready.
                            Never reachable in the production image — no runpodctl, no
                            scripts/, no SSH key there, and docker-compose's .env never
                            sets VLM_PROVIDER=runpod.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import requests

from env_config import env_float, env_str

log = logging.getLogger("vlm_provider")

_ROOT = Path(__file__).resolve().parent.parent
_lock = threading.Lock()
_state = {"starting": False, "ready": False}

# Pod create + SSH-ready + dependency install + model load routinely takes 5-15+
# minutes (see docs/RUNPOD_NOTES.md) — deliberately NOT VLM_TIMEOUT, which is sized for
# a single /predict HTTP call elsewhere in this repo. A too-short timeout here would
# SIGKILL runpod_up.py mid-provisioning and leak a billed pod (it can't run its own
# cleanup once killed).
RUNPOD_PROVISION_TIMEOUT_S = 900


def _provider() -> str:
    return env_str("VLM_PROVIDER").strip().lower()


def _health_url(vlm_url: str) -> str:
    return vlm_url.rsplit("/", 1)[0] + "/health" if vlm_url.endswith("/predict") else vlm_url


def is_ready() -> bool:
    """True if the remote VLM answers its health check right now."""
    url = env_str("VLM_URL").strip()
    if not url:
        return True  # in-process mode — nothing to warm
    try:
        return requests.get(_health_url(url), timeout=3).ok
    except requests.RequestException:
        return False


def warm() -> None:
    """Fire-and-forget nudge — called on page load (GET /api/vlm/warm)."""
    provider = _provider()
    if provider == "cloudrun":
        threading.Thread(target=_warm_cloudrun, daemon=True).start()
    elif provider == "runpod":
        threading.Thread(target=lambda: _ensure_runpod(env_float("VLM_TIMEOUT")), daemon=True).start()


def ensure_running(timeout_s: float | None = None) -> bool:
    """Block (up to timeout_s) until the VLM is reachable. True if ready.

    Called before every real /api/ask so a cold RunPod dev pod gets started (or
    finished starting, if warm() already kicked it off on page load) instead of the
    request just failing against a dead VLM_URL.
    """
    provider = _provider()
    if provider == "runpod":
        return _ensure_runpod(timeout_s or env_float("VLM_TIMEOUT"))
    # none / cloudrun: nothing to trigger here — either VLM_URL is always-on, or Cloud
    # Run's autoscaler starts an instance on the real request itself.
    return True


def _warm_cloudrun() -> None:
    url = env_str("VLM_URL").strip()
    try:
        requests.get(_health_url(url), timeout=5)
    except requests.RequestException as exc:
        log.info("cloudrun warm ping failed (cold start still proceeds on the real request): %s", exc)


def _ensure_runpod(timeout_s: float) -> bool:
    """Provision (or wait for) the RunPod dev pod. ``timeout_s`` bounds how long THIS
    CALLER waits for readiness — it is NOT the provisioning subprocess's own timeout
    (that's RUNPOD_PROVISION_TIMEOUT_S, sized for the real multi-minute pipeline, not
    reused from VLM_TIMEOUT which is a single-HTTP-call budget elsewhere in this repo).

    The lock is held only to atomically claim the "I'm the one provisioning" role, then
    released before the actual (multi-minute) subprocess call — never hold a lock across
    slow I/O. That's what lets a second concurrent caller actually observe
    ``starting=True`` and poll instead of kicking off (and potentially leaking) a
    second billed pod.
    """
    with _lock:
        if _state["ready"] and is_ready():
            return True
        already_starting = _state["starting"]
        if not already_starting:
            _state["starting"] = True

    if already_starting:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if is_ready():
                return True
            time.sleep(2)
        return False

    log.info("starting RunPod dev pod (scripts/runpod_up.py --no-wait)...")
    url = None
    try:
        proc = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "runpod_up.py"), "--no-wait"],
            capture_output=True, text=True, timeout=RUNPOD_PROVISION_TIMEOUT_S,
        )
        if proc.returncode != 0:
            log.error("runpod_up.py failed:\n%s\n%s", proc.stdout, proc.stderr)
        else:
            url = next(
                (line.split("=", 1)[1] for line in proc.stdout.splitlines()
                 if line.startswith("VLM_URL=")),
                None,
            )
    except subprocess.TimeoutExpired:
        # The child is SIGKILLed here and never gets to run its own teardown-on-error
        # path — but runpod_up.py writes the pod id to scripts/.runpod_state.json
        # immediately after creating it (before the slow install/health steps), so a
        # best-effort teardown call still has what it needs to avoid leaking a billed pod.
        log.error("runpod_up.py exceeded %ss; killed. Attempting best-effort teardown "
                   "so the pod doesn't keep billing.", RUNPOD_PROVISION_TIMEOUT_S)
        subprocess.run([sys.executable, str(_ROOT / "scripts" / "runpod_down.py")],
                        capture_output=True, timeout=60)
    finally:
        with _lock:
            _state["starting"] = False
            _state["ready"] = bool(url)
    if url:
        os.environ["VLM_URL"] = url
        log.info("RunPod pod ready: VLM_URL=%s", url)
    return bool(url)
