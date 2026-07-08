# RunPod dev deploy — insights and what to avoid

Notes from actually deploying `vlm_service` (Qwen3-VL-8B + LoRA) to a RunPod A5000 pod
by hand, before it was automated into `scripts/runpod_up.py` / `scripts/runpod_down.py`.
Every fix below is baked into those scripts already — this doc explains **why**, so the
next debugging session (on this project or another) doesn't rediscover them the hard way.

## 1. `HF_HOME` / `PIP_CACHE_DIR` are not inherited by SSH sessions

RunPod pods set `HF_HOME=/workspace/...` and `PIP_CACHE_DIR=/workspace/...` in the
**pod's PID-1 process environment** (visible via `cat /proc/1/environ`), but a plain
`ssh user@pod "..."` login shell does **not** inherit it — `env` in an SSH session shows
neither variable. The practical effect: an unset `HF_HOME` sends the ~17.5GB Qwen3-VL-8B
download to the default `~/.cache/huggingface`, which sits on the pod's **small container
root disk** (20GB by default), not the huge `/workspace` network mount (hundreds of TB,
shared across the datacenter). Same story for `PIP_CACHE_DIR` — pip's wheel cache landed
on the small disk and alone ate 22GB.

**Fix:** export `HF_HOME` and `PIP_CACHE_DIR` explicitly in *every* SSH command that
touches HF or pip — never rely on the pod's baked-in env.

## 2. The container root disk is small; bump it at pod creation

`runpodctl pod create` defaults `--container-disk-in-gb` to 20. Between the pip package
cache, a misdirected model download, and the packages themselves (`nvidia-*` cuDNN/cuBLAS
wheels alone run 2-3GB), 20GB is not enough headroom and installs fail mid-way with
`OSError: [Errno 28] No space left on device` — sometimes mid-uninstall, which is worse
(pip starts rolling back and can leave the environment half-upgraded).

**Fix:** `scripts/runpod_up.py` requests `--container-disk-in-gb 40` by default. Combined
with fix #1 (caches redirected to `/workspace`), this should not come up again — but if
you see `ENOSPC` on a pod, `df -h /` and `du -sh /root/.cache/* /usr/local/lib/python*/dist-packages/*`
are the first two commands to run.

## 3. `pip install` needs `--break-system-packages`, and `--ignore-installed`

These images ship a PEP 668 "externally managed" system Python. Plain `pip install`
refuses outright. `--break-system-packages` gets past that, but a second problem shows
up immediately after: some Debian-managed packages (`blinker`, `packaging` were the ones
we hit) have no pip `RECORD` file, so pip's normal upgrade path
(`Attempting uninstall: blinker` → `error: uninstall-no-record-file`) fails hard.

**Fix:** `pip install --break-system-packages --ignore-installed ...` — skips trying to
uninstall those packages first and just shadows them in site-packages.

## 4. Never let pip pick torch/torchvision on its own

This was the expensive one. `pip install -e "modeling[quant]"` (installing
`transformers`/`peft`/`accelerate`/`bitsandbytes` etc.) pulled in dependency resolution
that silently **upgraded torch to a fresh PyPI build (2.12.1+cu130)** — a CUDA 13 wheel
the pod's driver (built for CUDA 12.8) can't initialize. Symptom:
`torch.cuda.is_available() == False` with a driver-too-old warning, even though
`nvidia-smi` on the same box shows the GPU fine.

Fixing that by reinstalling `torch==2.8.0+cu128 torchvision==0.23.0+cu128` with
`--no-deps` (to "just fix torch, don't touch anything else") introduced a **second**
failure: torchvision unpinned resolved to `0.26.0+cu128` — a version *newer* than what
`torch==2.8.0` actually pairs with — producing
`RuntimeError: operator torchvision::nms does not exist` (an ABI mismatch between
torch and torchvision, not a CUDA problem). Pinning torchvision explicitly to the
version that actually matches (`0.23.0`, from PyTorch's own release compatibility
matrix) fixed the import.

Then a **third** failure: reinstalling with `--no-deps` skips torch's own declared
dependency on `nvidia-cudnn-cu12` — so `torch.cuda.is_available()` reported `True` (the
driver/toolkit are fine) but the first real GPU op crashed with
`RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED`, because the cuDNN library
version actually on disk (a leftover cu13 one from the earlier bad install) didn't match
what torch 2.8.0 expects.

**Fix:** always force-reinstall the *exact matched pair*
(`torch==2.8.0+cu128` + `torchvision==0.23.0+cu128`, from
`https://download.pytorch.org/whl/cu128`) **with** its normal dependency set (not
`--no-deps`) as the very last install step, after everything else. That pulls the correct
`nvidia-cudnn-cu12` companion and leaves nothing pinned to a stale/mismatched CUDA major
version. Verify with a real op, not just `torch.cuda.is_available()`:
```python
import torch
x = torch.randn(1, 3, 64, 64, device="cuda")
torch.nn.Conv2d(3, 8, 3).cuda()(x)   # exercises cuDNN, not just device visibility
```

## 5. Port 8001 is already taken

The pod's own nginx (RunPod's infra, not ours) listens on `0.0.0.0:8001`. Launching
`vlm_service` on 8001 fails with "Address already in use". We never needed a RunPod
HTTP proxy port at all — an SSH local tunnel (`ssh -L <local>:127.0.0.1:<remote> ...`)
reaches the service without touching the pod's exposed-port configuration in the
dashboard. **Fix:** default `vlm_service` to port **8010** on RunPod pods, tunnel to it.

## 6. `pgrep`/`killall` aren't on the image; kill by explicit PID

`pgrep -f server.py` silently fails with "command not found" — a `kill $(pgrep ...)`
one-liner then does nothing (no error, no effect) and a stale process keeps holding the
port. **Fix:** `scripts/runpod_up.py` always provisions a brand-new pod per run instead
of trying to relaunch on a reused one — there is never a stale process to kill on a
fresh pod. If you do need to kill something on an existing pod, use
`ps aux | grep <name>` and `kill <pid>` explicitly.

## 7. `ssh ... "long-running-thing &" ` can appear to hang

`nohup cmd > log 2>&1 & disown` is the right shape to background a process over a
non-interactive SSH command, but the SSH channel can still appear to block if stdin
isn't redirected (`< /dev/null`) — some shells keep the parent waiting on an open stdin
fd. Add `setsid` for a fully independent session too. Either way, treat the launch
command as fire-and-forget with a short timeout, and verify readiness with a **separate**
SSH connection polling `/health` — don't rely on the launch command's own exit to signal
success.

## 8. RunPod API key ends up in the pod's own environment

`RUNPOD_API_KEY` is present in plaintext in the pod's `env` (`/proc/1/environ`). It's
useful for tooling that needs to self-manage from inside the pod, but don't casually
`export`/print it or paste it into a shared terminal — treat it as a live credential.
`scripts/runpod_up.py` never reads or sets this key; it relies on `runpodctl` already
being configured locally (`runpodctl doctor`) with **your own** account key.

## 9. Prerequisites the scripts assume

- `runpodctl doctor` has been run once locally (configures the API key).
- Your SSH public key is added to your RunPod account:
  `runpodctl ssh add-key --key-file ~/.ssh/id_ed25519.pub` (one-time).
- `ssh`/`scp` on PATH (Windows 10/11 ships OpenSSH client by default).

## 10. Why RunPod for dev but Cloud Run GPU for prod

RunPod pods give real SSH access — exactly what made steps 1-7 above debuggable at all.
Cloud Run GPU has no shell access; iterating on a broken container image there means
rebuilding and redeploying for every fix, which is much slower for exploratory
debugging. RunPod is also cheaper per unit of interactive dev time (pay-per-second, no
image build/push round-trip). But RunPod pods bill continuously while running and need
manual teardown — not a fit for a public-facing production endpoint. Cloud Run GPU's
managed scale-to-zero is the right shape for that. Same `vlm_service/` Docker image runs
in both places (see `docs/REVIEW_AND_ROADMAP.md`) — only the orchestration differs.
