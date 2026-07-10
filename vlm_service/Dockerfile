# vlm_service — THIN serving image: FROM the prebuilt vlm-base (deps + baked base model),
# adds only the fast-changing bits (shared wrapper source, LoRA adapter, server code,
# config). See docs/VLM_IMAGE_SPLIT.md. This is what redeploys on every code/adapter/
# config change — in a couple of minutes, because the base is pulled as a cached layer
# (no apt, no pip, no 17.5 GB model download).
#
# Build context is the REPO ROOT (needs `modeling/` alongside this folder). VLM_BASE_IMAGE
# is injected by cloudbuild.yaml / scripts/gcloud_deploy_vlm.sh (the Artifact Registry
# path of vlm-base:latest). The default is a local convenience tag for `docker build`.
ARG VLM_BASE_IMAGE=chartqa-vlm-base:latest
FROM ${VLM_BASE_IMAGE}

WORKDIR /app

# Refresh the shared wrapper SOURCE over the base's editable install (Python imports from
# these files directly — no reinstall needed). --chown so it's owned by the runtime user
# (the base already switched to USER appuser; COPY defaults to root otherwise).
COPY --chown=appuser:appuser modeling/chartqa /app/modeling/chartqa

# Bake the LoRA adapter into the image (~170 MB) — no external download at boot. Rebuild
# (this thin image) when the adapter changes.
COPY --chown=appuser:appuser modeling/checkpoints/qwen3vl-lora-final2 /app/modeling/checkpoints/qwen3vl-lora-final2

COPY --chown=appuser:appuser vlm_service /app/vlm_service

EXPOSE 8001

# Model-serving config has NO in-code defaults (fail fast at boot) — pass at
# `docker run -e` / the host's env panel. QWEN_ADAPTER_PATH points at the baked-in
# adapter by default; override to '' for a base-model-only image.
# HF_HUB_OFFLINE=1: the base model is baked into vlm-base, so never phone home at boot —
# this removes the last network dependency from a cold start and makes it deterministic.
ENV QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct \
    QWEN_ADAPTER_PATH=/app/modeling/checkpoints/qwen3vl-lora-final2 \
    QWEN_QUANTIZATION=4bit \
    QWEN_MAX_NEW_TOKENS=64 \
    QWEN_ANSWER_SUFFIX=" Please answer directly." \
    VLM_PORT=8001 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# One worker = one model copy in VRAM; long timeout for cold generation.
# Shell form (not exec-form JSON array) so ${PORT:-...} expands — Cloud Run injects PORT
# and requires the container to listen on exactly that value; everywhere else (RunPod,
# local `docker run`), PORT is unset and VLM_PORT (default 8001) applies instead. Same
# image, same command, both environments — see docs/REVIEW_AND_ROADMAP.md.
CMD gunicorn -w 1 -t 300 -b 0.0.0.0:${PORT:-$VLM_PORT} --chdir vlm_service server:app
