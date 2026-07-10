"""Tear down the dev RunPod pod created by runpod_up.py — idempotent (safe to run
with nothing active) and the single source of truth for teardown, so runpod_up.py's
signal/error/idle paths and app.py's own shutdown all call the same code instead of
three slightly-different copies of "kill the pod".

Usage:
    python scripts/runpod_down.py
"""

from __future__ import annotations

import subprocess
import sys

from _runpod_common import clear_state, load_state, log


def teardown() -> None:
    state = load_state()
    if not state:
        log("no active pod recorded (nothing to do).")
        return

    tunnel_pid = state.get("tunnel_pid")
    if tunnel_pid:
        log(f"killing SSH tunnel (pid {tunnel_pid})...")
        subprocess.run(["kill", str(tunnel_pid)], capture_output=True)

    pod_id = state.get("pod_id")
    if pod_id:
        log(f"deleting pod {pod_id}...")
        proc = subprocess.run(["runpodctl", "pod", "delete", pod_id],
                               capture_output=True, text=True)
        if proc.returncode != 0:
            log(f"WARNING: `runpodctl pod delete {pod_id}` failed "
                f"(it may already be gone): {proc.stderr.strip()}")
        else:
            log(f"pod {pod_id} deleted — billing stopped.")

    clear_state()


if __name__ == "__main__":
    teardown()
    sys.exit(0)
