# vlm_service — GPU inference image for Qwen3-VL-8B (+ LoRA), CUDA runtime.
#
# Build context is the REPO ROOT (needs `modeling/` alongside this folder):
#   docker build -f vlm_service/Dockerfile -t <you>/chartqa-vlm:latest .
#
# The base image ships a CUDA-enabled torch build already, so `pip install` below only
# adds what's missing (transformers/peft/accelerate/bitsandbytes) instead of pulling a
# second torch wheel. Runs on any CUDA>=12.4 host (cloud GPU pod/serverless).
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# A C compiler is required at RUNTIME, not just build time: bitsandbytes' triton-based
# kernels (nn/triton_based_modules.py) JIT-compile a small C stub the first time they're
# imported (triton's CUDA driver init), via `cc`. This "runtime" base image has no
# compiler at all, so every worker crashed at boot with "Failed to find C compiler" the
# instant QwenVLChat.__init__ tried to load a 4-bit model — confirmed via real Cloud Run
# logs (2026-07-10): 100% reproducible, not a timeout/cold-start issue. The RunPod dev
# path never hit this because that base image (a full dev-oriented template) already had
# build tools; this "runtime" image never had them tested until the first real deploy.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install the chartqa package (shared QwenVLChat wrapper) + quantization extra.
# Copied selectively (not the whole modeling/ tree) to keep the image small — the
# repo's outputs/, notebooks/, dataset/ and the BLIP-2 checkpoint aren't needed here.
COPY modeling/pyproject.toml modeling/requirements.txt /app/modeling/
COPY modeling/chartqa /app/modeling/chartqa
RUN pip install --no-cache-dir -e "/app/modeling[quant]"

# This service's own (light) serving deps.
COPY vlm_service/requirements.txt /app/vlm_service/requirements.txt
RUN pip install --no-cache-dir -r /app/vlm_service/requirements.txt

# Belt-and-suspenders (see docs/RUNPOD_NOTES.md #4): transformers/peft/bitsandbytes'
# own dependency resolution can silently pull a newer torch/torchvision than this base
# image ships, and an unpinned torchvision can be ABI-mismatched with the pinned torch
# (RuntimeError: operator torchvision::nms does not exist). Re-pin the exact matched
# pair WITH its normal deps (not --no-deps, which would skip the nvidia-cudnn companion
# package and break cuDNN init) as the last word on what's installed.
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Bake the LoRA adapter into the image (~170 MB) — no external download at boot, no
# HF Hub upload needed. Rebuild the image when the adapter changes.
COPY modeling/checkpoints/qwen3vl-lora-final2 /app/modeling/checkpoints/qwen3vl-lora-final2

COPY vlm_service /app/vlm_service

# Run as a non-root user (security hardening, §3.7). HF_HOME points at a dir the user
# owns; gunicorn binds 8001/$PORT (>1024), so no privileged port is needed.
ENV HF_HOME=/opt/hf
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/hf && chown -R appuser:appuser /opt/hf /app
USER appuser

# Bake the base model (~17.5 GB) into the image so a scale-to-zero (min=0) cold start
# does NOT download it from HF every time — the cold start becomes "load from local
# disk" (seconds) instead of "download 17.5 GB then load" (minutes, and prone to
# timing out / HF rate limits). Qwen3-VL-8B-Instruct is public (no HF_TOKEN needed).
# Downloaded as appuser into HF_HOME so it's owned + readable at runtime. This makes the
# image large (~30 GB) and the build/push slow — that's the trade for reliable cold
# starts (see cloudbuild.yaml's raised timeout). Keep the repo id in sync with
# QWEN_MODEL_ID below.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3-VL-8B-Instruct')"

EXPOSE 8001

# Model-serving config has NO in-code defaults (fail fast at boot) — pass at
# `docker run -e` / the host's env panel. QWEN_ADAPTER_PATH points at the baked-in
# adapter by default; override to '' for a base-model-only image.
# HF_HUB_OFFLINE=1: the base model is baked (above), so never phone home at boot — this
# removes the last network dependency from a cold start and makes it deterministic.
ENV QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
    QWEN_ADAPTER_PATH=/app/modeling/checkpoints/qwen3vl-lora-final2 \
    QWEN_QUANTIZATION=4bit \
    QWEN_MAX_NEW_TOKENS=64 \
    QWEN_ANSWER_SUFFIX=" Please answer directly." \
    VLM_PORT=8001 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# One worker = one model copy in VRAM; long timeout for cold generation + first HF download.
# Shell form (not exec-form JSON array) so ${PORT:-...} expands — Cloud Run injects PORT
# and requires the container to listen on exactly that value; everywhere else (RunPod,
# local `docker run`), PORT is unset and VLM_PORT (default 8001) applies instead. Same
# image, same command, both environments — see docs/REVIEW_AND_ROADMAP.md.
CMD gunicorn -w 1 -t 300 -b 0.0.0.0:${PORT:-$VLM_PORT} --chdir vlm_service server:app
