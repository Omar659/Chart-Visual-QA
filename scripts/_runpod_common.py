"""Shared helpers for runpod_up.py / runpod_down.py — one state file, one source of
truth for pod bookkeeping, so the two scripts never drift from each other.

Not a public module (leading underscore): it's plumbing for the two CLI scripts in
this folder, not something other code should import.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

STATE_FILE = Path(__file__).resolve().parent / ".runpod_state.json"

# Known-good pair (see docs/RUNPOD_NOTES.md): whatever the pod's base image ships,
# we always pin torch/torchvision to this exact matched cu128 build. Mismatched
# pairs are the #1 failure mode we hit (ABI errors, cuDNN init failures, or a
# driver-incompatible CUDA 13 wheel pulled in as a transitive dependency).
TORCH_VERSION = "2.8.0+cu128"
TORCHVISION_VERSION = "0.23.0+cu128"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"


def log(msg: str) -> None:
    print(f"[runpod] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[runpod] ERROR: {msg}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_state() -> None:
    STATE_FILE.unlink(missing_ok=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, always capturing text output; raise with a readable message."""
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
    if proc.returncode != 0:
        die(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return proc


def runpodctl_json(args: list[str]) -> dict | list:
    """Run `runpodctl <args> -o json` and parse the result."""
    proc = run(["runpodctl", *args, "-o", "json"])
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        die(f"could not parse runpodctl JSON output for `runpodctl {' '.join(args)}`:\n{proc.stdout}")


def check_runpodctl_ready() -> None:
    """Fail fast with a clear message if runpodctl isn't installed/authenticated."""
    from shutil import which

    if which("runpodctl") is None:
        die(
            "runpodctl not found on PATH. Install it: "
            "https://github.com/runpod/runpodctl#installation"
        )
    proc = subprocess.run(["runpodctl", "user", "-o", "json"], capture_output=True, text=True)
    if proc.returncode != 0 or "api key not configured" in (proc.stdout + proc.stderr).lower():
        die(
            "runpodctl has no API key configured. Get one at "
            "https://www.runpod.io/console/user/settings then run:\n"
            "  runpodctl doctor\n"
            "(or `export RUNPOD_API_KEY=...`). We deliberately don't set this for you — "
            "it's your account credential, not something a script should silently persist."
        )


def parse_ssh_command(ssh_command: str) -> dict:
    """Extract host/port/key/user from the `ssh ...` string `runpodctl ssh info` returns."""
    host_m = re.search(r"\b(?:\w+@)?([\w.\-]+\.[\w.\-]+)\b", ssh_command)
    user_m = re.search(r"\b(\w+)@", ssh_command)
    port_m = re.search(r"-p\s+(\d+)", ssh_command)
    key_m = re.search(r"-i\s+(\S+)", ssh_command)
    if not (host_m and port_m):
        die(f"could not parse host/port from `runpodctl ssh info` output: {ssh_command!r}")
    return {
        "host": host_m.group(1),
        "port": int(port_m.group(1)),
        "user": user_m.group(1) if user_m else "root",
        "key": key_m.group(1) if key_m else None,
    }


def ssh_target_args(state: dict) -> list[str]:
    args = ["-p", str(state["ssh_port"])]
    if state.get("ssh_key"):
        args += ["-i", state["ssh_key"]]
    args += ["-o", "StrictHostKeyChecking=accept-new", "-o", "ConnectTimeout=15"]
    return args


def ssh_run(state: dict, remote_cmd: str, timeout: int = 60) -> subprocess.CompletedProcess:
    target = f"{state['ssh_user']}@{state['ssh_host']}"
    cmd = ["ssh", *ssh_target_args(state), target, remote_cmd]
    log("ssh $ " + remote_cmd[:120] + ("..." if len(remote_cmd) > 120 else ""))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
