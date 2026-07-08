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

# Run as a non-root user (security hardening, §3.7). The base model downloads at boot
# into HF_HOME, so point it at a dir the user owns. gunicorn binds 8001/$PORT (>1024),
# so no privileged port is needed.
ENV HF_HOME=/opt/hf
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/hf && chown -R appuser:appuser /opt/hf /app
USER appuser

EXPOSE 8001

# Model-serving config has NO in-code defaults (fail fast at boot) — pass at
# `docker run -e` / the host's env panel. QWEN_ADAPTER_PATH points at the baked-in
# adapter by default; override to '' for a base-model-only image.
ENV QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
    QWEN_ADAPTER_PATH=/app/modeling/checkpoints/qwen3vl-lora-final2 \
    QWEN_QUANTIZATION=4bit \
    QWEN_MAX_NEW_TOKENS=64 \
    QWEN_ANSWER_SUFFIX=" Please answer directly." \
    VLM_PORT=8001

# One worker = one model copy in VRAM; long timeout for cold generation + first HF download.
# Shell form (not exec-form JSON array) so ${PORT:-...} expands — Cloud Run injects PORT
# and requires the container to listen on exactly that value; everywhere else (RunPod,
# local `docker run`), PORT is unset and VLM_PORT (default 8001) applies instead. Same
# image, same command, both environments — see docs/REVIEW_AND_ROADMAP.md.
CMD gunicorn -w 1 -t 300 -b 0.0.0.0:${PORT:-$VLM_PORT} --chdir vlm_service server:app
