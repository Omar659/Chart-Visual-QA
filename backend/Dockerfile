# Backend: Flask API + CLIP chart detector + Layer-2 encoders + Tesseract OCR.
# Served by gunicorn (production mode) via docker-compose.yml / Cloud Run.
FROM python:3.12-slim

# Tesseract OCR engine — the system binary pytesseract shells out to (so the
# "has data values" gate runs without a host install). On PATH here, so the
# backend service sets TESSERACT_CMD="" (see docker-compose.yml).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of torch — the backend runs CLIP on CPU; the GPU is for the vLLM guard.
# Installed first so transformers/detoxify don't pull the much larger CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer-2 guard models pull in transformers/presidio/detoxify and (via `-r
# requirements.txt`) the app + CLIP deps. Cached as its own layer.
COPY requirements.txt requirements-guard.txt ./
RUN pip install --no-cache-dir -r requirements-guard.txt \
    && python -m spacy download en_core_web_lg

# Run as a non-root user (security hardening, §3.7). Create it BEFORE the model-precache
# steps and point every ML cache at a dir it owns, then switch to it — so the weights are
# cached (and found at runtime) under the runtime user, not root's unreadable ~/.cache.
# HF_HOME covers transformers/CLIP/deberta; TORCH_HOME covers detoxify's torch.hub download.
ENV HF_HOME=/opt/ml-cache/hf \
    TORCH_HOME=/opt/ml-cache/torch \
    XDG_CACHE_HOME=/opt/ml-cache/xdg
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /opt/ml-cache && chown -R appuser:appuser /opt/ml-cache /app
USER appuser

# Pre-cache CLIP weights at build so the first request isn't a ~600 MB download.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

# Pre-cache the Layer-2 guard encoders at build (toxicity = detoxify/toxic-bert,
# prompt-injection = deberta) so the backend doesn't download them at boot — warmup
# becomes instant instead of a multi-minute runtime download. Presidio (PII) uses the
# spaCy model already installed above. Keep these model ids in sync with .env defaults
# (GUARD_TOXICITY_MODEL / GUARD_INJECTION_MODEL); a non-default value re-downloads once.
RUN python -c "from detoxify import Detoxify; Detoxify('original')" \
    && python -c "from transformers import pipeline; \
       pipeline('text-classification', \
                model='protectai/deberta-v3-base-prompt-injection-v2', truncation=True)"

COPY --chown=appuser:appuser . .

# All guard/CLIP weights are baked into the image above — never phone home to HF Hub at
# runtime. Without this, every model init still did live HEAD/GET cache-validation calls
# to huggingface.co (confirmed via Cloud Run logs, 2026-07-10: ~10-70s of network round
# trips, sometimes 429-throttled, on the very first /api/ask after a cold start) even
# though the weights were already on disk — same class of bug already fixed for
# vlm_service/Dockerfile, just never applied here.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 5000
# gunicorn, not Flask's dev server: ratelimit.py/answer_cache.py/redis_client.py were
# already designed around "shared across gunicorn workers" (their own comments say so),
# but the CMD never actually used it — `python app.py` runs Werkzeug's single-threaded
# dev server, so Cloud Run's default --concurrency=80 was a lie the container couldn't
# back up (confirmed: a concurrent /api/ask landed on a cold-starting process and 503'd
# while an already-warm instance sat idle). One worker + threads (not multiple worker
# processes) so the ~1-2GB of guard/CLIP models (warmed via gunicorn.conf.py's
# post_worker_init hook) aren't duplicated in memory per worker. Shell form so
# ${PORT:-...} expands (Cloud Run injects PORT; local/docker-compose falls back to 5000).
CMD gunicorn -c gunicorn.conf.py -b 0.0.0.0:${PORT:-5000} app:app
