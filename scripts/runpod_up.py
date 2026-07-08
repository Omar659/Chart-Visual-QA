"""Provision a RunPod GPU pod running vlm_service (Qwen3-VL-8B + LoRA) for dev/testing.

Minimal, single-purpose: create a fresh pod, upload the code + adapter, install deps
with the exact fixes we needed the hard way (see docs/RUNPOD_NOTES.md), launch the
service, tunnel it to localhost, and print VLM_URL. Always provisions a FRESH pod
(never reuses/updates an existing one) — that sidesteps stale-process and
half-upgraded-dependency bugs entirely, at the cost of a few minutes per run.

Usage:
    python scripts/runpod_up.py                  # create, wait, watch idle, teardown on exit
    python scripts/runpod_up.py --no-wait         # create + print VLM_URL, return immediately
                                                    # (caller — e.g. app.py --dev --runpod —
                                                    # owns the pod's lifecycle from here)

Prerequisites: `runpodctl doctor` has been run once (API key configured), and the
SSH public key matching --ssh-key is already added to your RunPod account
(`runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub`).
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

from _runpod_common import (
    TORCH_INDEX_URL,
    TORCH_VERSION,
    TORCHVISION_VERSION,
    check_runpodctl_ready,
    die,
    log,
    parse_ssh_command,
    run,
    runpodctl_json,
    save_state,
    ssh_run,
    ssh_target_args,
)

ROOT = Path(__file__).resolve().parent.parent
IDLE_TIMEOUT_S = 30 * 60
POLL_INTERVAL_S = 60


def create_pod(args: argparse.Namespace) -> str:
    log(f"creating pod (gpu={args.gpu_id!r}, image={args.image!r}, disk={args.disk_gb}GB)...")
    result = runpodctl_json([
        "pod", "create",
        "--name", args.name,
        "--gpu-id", args.gpu_id,
        "--image", args.image,
        "--container-disk-in-gb", str(args.disk_gb),
        "--ssh",
    ])
    pod_id = result.get("id") if isinstance(result, dict) else None
    if not pod_id:
        die(f"pod create did not return an id: {result}")
    log(f"pod created: {pod_id}")
    return pod_id


def wait_for_running(pod_id: str, timeout_s: int = 300) -> None:
    log("waiting for pod to reach RUNNING...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        info = runpodctl_json(["pod", "get", pod_id])
        status = (info.get("desiredStatus") or info.get("status") or "").upper()
        if status == "RUNNING":
            log("pod is RUNNING.")
            return
        time.sleep(5)
    die(f"pod {pod_id} did not reach RUNNING within {timeout_s}s.")


def get_ssh_target(pod_id: str, ssh_key: str, timeout_s: int = 300) -> dict:
    """`runpodctl ssh info` can 404 with "pod not ready" for a bit even after the pod's
    own status already reads RUNNING — the SSH daemon inside isn't necessarily up yet.
    Retry until it returns an actual command instead of treating the first miss as fatal."""
    log("fetching SSH connection info...")
    deadline = time.time() + timeout_s
    info = {}
    while time.time() < deadline:
        info = runpodctl_json(["ssh", "info", pod_id])
        if isinstance(info, dict) and info.get("command"):
            break
        time.sleep(5)
    else:
        die(f"`runpodctl ssh info {pod_id}` never returned a command within "
            f"{timeout_s}s: {info}")
    ssh_command = info["command"]
    parsed = parse_ssh_command(ssh_command)
    if not parsed.get("key"):
        parsed["key"] = ssh_key
    return {
        "pod_id": pod_id,
        "ssh_host": parsed["host"],
        "ssh_port": parsed["port"],
        "ssh_user": parsed["user"],
        "ssh_key": parsed["key"],
    }


def wait_for_ssh(state: dict, timeout_s: int = 180) -> None:
    log("waiting for SSH to accept connections...")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        proc = ssh_run(state, "echo ok", timeout=15)
        if proc.returncode == 0 and "ok" in proc.stdout:
            log("SSH is up.")
            return
        time.sleep(5)
    die(f"SSH did not become reachable within {timeout_s}s.")


def upload_code(state: dict, adapter_path: Path) -> None:
    log("uploading code + adapter (chartqa, vlm_service, LoRA weights)...")
    target = f"{state['ssh_user']}@{state['ssh_host']}"
    ssh_args = ssh_target_args(state)
    scp_port_args = ["-P", str(state["ssh_port"])] + (
        ["-i", state["ssh_key"]] if state.get("ssh_key") else []
    ) + ["-o", "StrictHostKeyChecking=accept-new"]

    ssh_run(state, "mkdir -p /workspace/modeling/checkpoints /workspace/vlm_service", timeout=30)

    run(["scp", *scp_port_args, "-r",
         str(ROOT / "modeling" / "chartqa"),
         str(ROOT / "modeling" / "pyproject.toml"),
         str(ROOT / "modeling" / "requirements.txt"),
         f"{target}:/workspace/modeling/"])
    run(["scp", *scp_port_args, "-r", str(ROOT / "vlm_service"), f"{target}:/workspace/"])
    if adapter_path.is_dir():
        run(["scp", *scp_port_args, "-r", str(adapter_path),
             f"{target}:/workspace/modeling/checkpoints/"])
    else:
        log(f"WARNING: adapter path {adapter_path} not found — deploying base model only.")


# Every fix below was discovered the hard way (docs/RUNPOD_NOTES.md has the full story):
#   - HF_HOME/PIP_CACHE_DIR must be set explicitly in every SSH command: they're only in
#     the pod's PID-1 env, NOT inherited by SSH login shells, so an unset HF_HOME silently
#     downloads the 17GB base model onto the small (small unless --container-disk-in-gb is
#     bumped) container root disk instead of the huge /workspace network mount.
#   - `pip install` on these images needs --break-system-packages (PEP 668), and
#     --ignore-installed to skip Debian-managed packages pip can't safely uninstall
#     (blinker, packaging) — they have no pip RECORD file.
#   - transformers/peft/etc. pull in an unpinned torchvision/torch that can silently
#     upgrade to a CUDA 13 build the pod's driver doesn't support (breaks
#     torch.cuda.is_available()) or a torchvision ABI-mismatched with torch (breaks
#     `torchvision::nms`). Force-reinstall the known-good matched pair WITH its normal
#     dependencies (not --no-deps — that skips the nvidia-cudnn companion package and
#     causes CUDNN_STATUS_NOT_INITIALIZED at the first real conv forward pass).
INSTALL_CMD = (
    "cd /workspace && "
    "export PIP_CACHE_DIR=/workspace/.cache/pip && "
    "pip install --break-system-packages --ignore-installed -e 'modeling[quant]' && "
    f"pip install --break-system-packages --force-reinstall "
    f"torch=={TORCH_VERSION} torchvision=={TORCHVISION_VERSION} "
    f"--index-url {TORCH_INDEX_URL} && "
    "pip install --break-system-packages --no-deps -r vlm_service/requirements.txt"
)


def install_deps(state: dict) -> None:
    log("installing dependencies on the pod (a few minutes)...")
    proc = ssh_run(state, INSTALL_CMD, timeout=900)
    if proc.returncode != 0:
        die(f"dependency install failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
    log("dependencies installed.")


def launch_service(state: dict, args: argparse.Namespace) -> None:
    log(f"launching vlm_service on remote port {args.remote_port}...")
    launch_cmd = (
        "cd /workspace && "
        "export HF_HOME=/workspace/.cache/huggingface && "
        f"export QWEN_MODEL_ID={args.model_id} && "
        f"export QWEN_ADAPTER_PATH=/workspace/modeling/checkpoints/{args.adapter_dirname} && "
        f"export QWEN_QUANTIZATION={args.quantization} && "
        f"export QWEN_MAX_NEW_TOKENS={args.max_new_tokens} && "
        "export QWEN_ANSWER_SUFFIX=' Please answer directly.' && "
        f"export VLM_PORT={args.remote_port} && "
        "setsid python3 vlm_service/server.py > /workspace/vlm.log 2>&1 < /dev/null & "
        "disown; sleep 1; true"
    )
    try:
        ssh_run(state, launch_cmd, timeout=20)
    except subprocess.TimeoutExpired:
        pass  # expected sometimes; health polling below is the real verification


def wait_for_health(state: dict, remote_port: int, timeout_s: int = 600) -> None:
    log("waiting for the model to load (first run downloads ~17.5GB, can take a while)...")
    deadline = time.time() + timeout_s
    check_cmd = f"curl -sf http://127.0.0.1:{remote_port}/health"
    while time.time() < deadline:
        proc = ssh_run(state, check_cmd, timeout=15)
        if proc.returncode == 0 and '"status":"ok"' in proc.stdout:
            log("model ready and serving.")
            return
        time.sleep(10)
    tail = ssh_run(state, "tail -60 /workspace/vlm.log", timeout=15)
    die(f"service did not become healthy within {timeout_s}s. Last log lines:\n{tail.stdout}")


def open_tunnel(state: dict, local_port: int, remote_port: int) -> int:
    """Open the tunnel WITHOUT ssh's own `-f` self-daemonize flag — that would require
    finding the resulting process afterwards (via `pgrep`, which isn't available on
    Windows/Git Bash, the platform this project actually develops on). Launching via
    Popen instead means Python already holds the real PID directly, portably, with no
    external process-lookup tool needed. Not waiting on the Popen object lets it keep
    running as an independent child after this script exits (true on both POSIX and
    Windows) — exactly the "survives after --no-wait returns" behavior we need.
    """
    log(f"opening SSH tunnel localhost:{local_port} -> pod:{remote_port}...")
    target = f"{state['ssh_user']}@{state['ssh_host']}"
    cmd = ["ssh", "-N", "-L", f"{local_port}:127.0.0.1:{remote_port}",
           *ssh_target_args(state), target]
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(10):
        if proc.poll() is not None:
            die(f"SSH tunnel process exited early (code {proc.returncode}).")
        r = subprocess.run(["curl", "-sf", f"http://127.0.0.1:{local_port}/health"],
                            capture_output=True, text=True)
        if r.returncode == 0:
            log("tunnel verified.")
            return proc.pid
        time.sleep(1)
    proc.kill()
    die("tunnel opened but the local health check never succeeded.")


def idle_watchdog(state: dict, remote_port: int) -> None:
    """Block until Ctrl-C, an error, or IDLE_TIMEOUT_S with no requests — then tear down."""
    from runpod_down import teardown

    stop = {"flag": False}

    def _handler(signum, _frame):
        log(f"signal {signum} received; tearing down...")
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

    log(f"watching for idle (>{IDLE_TIMEOUT_S // 60} min with no requests) — Ctrl-C to stop now.")
    try:
        while not stop["flag"]:
            time.sleep(POLL_INTERVAL_S)
            proc = ssh_run(state, "stat -c %Y /workspace/vlm.log", timeout=15)
            if proc.returncode != 0:
                continue
            idle_for = time.time() - int(proc.stdout.strip())
            log(f"idle for {int(idle_for)}s")
            if idle_for > IDLE_TIMEOUT_S:
                log("idle timeout reached; tearing down.")
                break
    except Exception as exc:  # noqa: BLE001 — any unexpected error also triggers teardown
        log(f"unexpected error ({exc}); tearing down.")
    finally:
        teardown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-id", default="NVIDIA RTX A5000")
    parser.add_argument("--image", default="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404")
    parser.add_argument("--disk-gb", type=int, default=40,
                         help="container (root) disk size — bumped from runpodctl's default "
                              "20GB; 20GB was not enough and caused mid-install ENOSPC errors.")
    parser.add_argument("--name", default="chartqa-dev")
    parser.add_argument("--ssh-key", default=str(Path.home() / ".ssh" / "id_ed25519"))
    parser.add_argument("--local-port", type=int, default=8010)
    parser.add_argument("--remote-port", type=int, default=8010,
                         help="8001 is taken by the pod's own nginx — avoid it.")
    parser.add_argument("--adapter", default="modeling/checkpoints/qwen3vl-lora-final2",
                         help="path (repo-root-relative) to the LoRA adapter to deploy.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--quantization", default="4bit", choices=["none", "8bit", "4bit"])
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--no-wait", action="store_true",
                         help="set up and print VLM_URL, then return immediately — the "
                              "caller (e.g. app.py --dev --runpod) owns teardown.")
    args = parser.parse_args()
    args.adapter_dirname = Path(args.adapter).name
    adapter_path = ROOT / args.adapter

    check_runpodctl_ready()

    pod_id = create_pod(args)
    save_state({"pod_id": pod_id})  # save early so a mid-setup failure can still be torn down
    try:
        wait_for_running(pod_id)
        state = get_ssh_target(pod_id, args.ssh_key)
        wait_for_ssh(state)
        upload_code(state, adapter_path)
        install_deps(state)
        launch_service(state, args)
        wait_for_health(state, args.remote_port)
        tunnel_pid = open_tunnel(state, args.local_port, args.remote_port)
        state.update({
            "local_port": args.local_port,
            "remote_port": args.remote_port,
            "tunnel_pid": tunnel_pid,
        })
        save_state(state)
    except SystemExit:
        log("setup failed — tearing down the pod so it doesn't keep billing.")
        from runpod_down import teardown
        teardown()
        raise

    vlm_url = f"http://127.0.0.1:{args.local_port}/predict"
    print(f"VLM_URL={vlm_url}")
    log(f"ready. VLM_URL={vlm_url}  (pod {pod_id}, billing until you tear it down)")

    if not args.no_wait:
        idle_watchdog(state, args.remote_port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
