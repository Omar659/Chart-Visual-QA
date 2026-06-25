# Deployment Architecture — Containers (proposal)

> Status: **proposal for review.** Covers how Chart-Visual-QA runs in containers:
> the backend (with **Tesseract** + **CLIP** baked in), the **Llama Guard** service
> (**Ollama** in dev → **vLLM** in prod), and how `python app.py` orchestrates them.
> Config stays single-source in `.env` (see [.env.example](../.env.example)); the
> backend reads every key as required (no in-code defaults).

## 1. Principles (recap)

- **Load models once, keep them warm; never in the request path** (CLAUDE.md → Engineering principles).
- **Each guard model is its own service**, kept warm, scaled independently of the app.
- **Cheapest check first, short-circuit; gate the expensive LLM behind the cheap layers.**
- **Fail-open**: a missing/down dependency degrades, never crashes the app.
- **Same code dev↔prod; only `GUARD_LLM_URL` changes** (now true — guard_llm speaks the
  OpenAI `/v1/chat/completions` API that both Ollama and vLLM serve).

## 2. Should they be in different containers? — **Yes.**

The backend and the guard LLM go in **separate containers**. Rationale:

| Reason | Detail |
| --- | --- |
| **Different hardware** | The guard LLM wants a **GPU**; the Flask backend + CLIP run fine on **CPU**. Splitting lets you put only the guard on a GPU node. |
| **Independent lifecycle** | Restarting/redeploying the Flask app must **not** reload a multi-GB LLM. The guard stays warm across backend deploys. |
| **Independent scaling** | Keep **one warm guard**; scale backend replicas separately. Later the (heavy) VLM can **scale-to-zero** while the guard stays up. |
| **Swap Ollama↔vLLM** | The backend only knows `GUARD_LLM_URL`. Replace the guard container (Ollama→vLLM) with **zero backend changes**. |
| **Image hygiene** | The backend image (Tesseract + CLIP + Flask) stays lean; the CUDA/vLLM image (several GB) is separate. |

So: **backend container** + **guard container** (+ frontend, see §5). A future **VLM
inference** container is a third service behind the same `run_inference` seam.

## 3. Components

```
                 ┌──────────────────────────────────────────────┐
   browser ─────▶│  frontend  (Vite dev, or nginx static build)  │
                 └───────────────┬──────────────────────────────┘
                                 │  /api/*  (proxy)
                 ┌───────────────▼──────────────────────────────┐
                 │  backend  (Flask)                             │
                 │   • Layer 1  cheap rules                      │
                 │   • Layer 2  toxicity/injection/PII encoders  │
                 │   • chart gate  CLIP (torch) + Tesseract OCR  │  CPU ok
                 │   • run_inference  (mock → VLM later)         │
                 └───────────────┬──────────────────────────────┘
                                 │  GUARD_LLM_URL  (OpenAI /v1/chat/completions)
                 ┌───────────────▼──────────────────────────────┐
                 │  guard  (Llama Guard 3 1B)                    │
                 │   dev:  Ollama   ·   prod: vLLM               │  GPU
                 └──────────────────────────────────────────────┘
```

| Service | Image base | Key deps | Port | GPU |
| --- | --- | --- | --- | --- |
| **backend** | `python:3.12-slim` | Flask, torch+transformers (CLIP), **tesseract-ocr (apt)**, presidio/detoxify | 5000 | optional |
| **guard** | dev `ollama/ollama` · prod `vllm/vllm-openai` | Llama Guard 3 1B weights | 11434 (Ollama) / 8000 (vLLM) | **yes** |
| **frontend** | `node:20` (dev) or `nginx` (prod build) | Vite / static bundle | 5173 / 80 | no |

## 4. Tesseract — in the backend image (no host install)

Instead of installing the Tesseract engine on each machine (the Windows PATH pain we
hit), the backend Dockerfile installs it with **one apt line**:

```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*
```

In the container Tesseract is on `PATH`, so `TESSERACT_CMD` stays **empty** there.
`TESSERACT_CMD` remains the escape hatch for non-container Windows dev.

## 5. Dev vs prod — the guard, and the frontend

**Guard (Ollama → vLLM).** Same OpenAI API both sides; only `GUARD_LLM_URL` + `GUARD_LLM_MODEL` differ:

| | dev | prod |
| --- | --- | --- |
| Engine | Ollama | vLLM (`vllm/vllm-openai`) |
| `GUARD_LLM_URL` | `http://guard:11434` | `http://guard:8000` |
| `GUARD_LLM_MODEL` | `llama-guard3:1b` | `meta-llama/Llama-Guard-3-1B` |
| GPU | Ollama runtime | CUDA container (continuous batching, quantized) |

> **Recommendation:** keep **Ollama for local dev** (lightest; it already works, manages
> its own GPU) and use the **vLLM container for prod**. vLLM in Docker needs a CUDA GPU
> passed in — on Windows that's **WSL2 + NVIDIA Container Toolkit**. Your RTX 4050 can
> serve Llama Guard 3 1B, so a dev vLLM container is *possible*; it's just heavier than
> Ollama. Both are wired identically — flip `GUARD_LLM_URL` in `.env`.

**Frontend.** For dev, keep the **local Vite dev server** (instant hot-reload). For prod,
build static assets and serve via nginx. (Containerizing the dev frontend is supported as
a compose profile but isn't the default — it slows the edit loop.)

## 6. `python app.py` orchestration — two modes

| Command | Mode | Backend | Chart/CLIP | Guard LLM | Frontend |
| --- | --- | --- | --- | --- | --- |
| `python app.py --dev` | **local dev** (today's flow) | venv | local CLIP | **Ollama** (`:11434`) | local Vite |
| `python app.py` | **production** (default) | **Docker** | CLIP **in container** | **vLLM** container (`:8000`) | local Vite |

**Default (production):**
1. **Check the Docker engine** (`docker info`). If it's down → **clear warning**
   (*"Docker Desktop isn't running — start it, or use `python app.py --dev` for the local
   flow"*) and stop.
2. **`docker compose up --build backend guard`** — backend (Flask + CLIP + Tesseract) and
   the vLLM guard.
3. Start the **local Vite** frontend proxying to the container backend on `:5000`.
4. On Ctrl-C, `docker compose down` + stop the frontend.

**`--dev`:** the existing venv + local-Ollama + Vite flow, unchanged. No Docker required.

## 7. Networking & config

- One compose network; the backend reaches the guard at the **service name**
  (`http://guard:8000`), not `localhost`.
- All config via **`.env`** passed to containers with compose `env_file:` — same single
  source of truth as local runs. `GUARD_LLM_URL` is the only value that differs by env.

## 8. Model caching (avoid cold downloads)

- **CLIP**: pre-download in the backend image build (a `RUN python -c "...from_pretrained..."`)
  so the first request isn't a 600 MB pull. Warm it at boot off the request path.
- **Guard weights**: mount a named volume (`ollama`/`hf-cache`) so the model persists
  across restarts instead of re-downloading.

## 9. Decisions (confirmed)

1. **Two modes** — `python app.py --dev` is the **local** flow (venv, local CLIP, **Ollama**,
   Vite). `python app.py` (default) is **production**: backend + **vLLM** guard in Docker,
   CLIP/Tesseract in the backend container, **frontend stays local Vite** on `:5000`.
2. **Guard** — **Ollama for dev** (`:11434`), **vLLM container for prod** (`:8000`, GPU via
   **WSL2 + NVIDIA Container Toolkit**). Same OpenAI API; only `GUARD_LLM_URL`/`_MODEL` differ.
3. **Frontend prod** — **deferred**; add the nginx static-build container when the VLM lands.

### Consequences baked into the files

- The backend now binds **`HOST`** (`0.0.0.0` in the container, `127.0.0.1` locally) and
  toggles **`FLASK_DEBUG`** — both in `.env`. Warmup runs once in either mode.
- Compose overrides three values for the backend service: `GUARD_LLM_URL=http://guard:8000`,
  `GUARD_LLM_MODEL=meta-llama/Llama-Guard-3-1B`, and `TESSERACT_CMD=` (empty → use the
  apt-installed binary on PATH, not the Windows path from local `.env`).
- **`meta-llama/Llama-Guard-3-1B` is a gated HF model** — set **`HF_TOKEN`** (and accept the
  license on Hugging Face) so the vLLM container can pull the weights.
- Backend runs via `python app.py` (Flask) for now; **gunicorn** is the prod hardening step
  once the mock is replaced.

## 10. Proposed files (next step, after you confirm)

- `backend/Dockerfile` — Flask + CLIP + Tesseract, CLIP pre-cached.
- `backend/.dockerignore` — keep `.venv`, `__pycache__`, tests out of the image.
- `docker-compose.yml` — `backend` + `guard` (Ollama default; vLLM via `--profile prod`).
- `frontend/Dockerfile` (optional) — nginx static build for prod.
- `app.py` — Docker-engine check + `docker compose up`, with `--no-docker` fallback.

## 11. Cloud deploy target (scale-to-zero) — two services

All **input validation on CPU**, only the **main VLM on GPU**:

- **Service A — app + gates (CPU, `min=1`):** frontend build + Flask backend + Layer 1 +
  Layer 2 encoders + Layer 3 Llama Guard + CLIP. Cheap to keep warm; it's the "gatekeeper"
  that rejects junk / non-charts / unsafe input **without waking the GPU**.
- **Service B — VLM (GPU, `min=0`):** the main chart-QA model only. Scales to zero; only
  wakes for a legitimate chart question that passed every cheap gate. Biggest cost lever.

Cold start: CPU service ~5–30s (or ~0 at `min=1`); GPU service ~30s–3min (GPU provision +
multi-GB model load) — the pain of `min=0`. Warm inference: gates <0.5s (Llama Guard on CPU
~1–4s); VLM ~1–5s. Keeping CLIP+guards on CPU means the GPU only cold-starts for real work.

## 12. Future / next steps

- **Quantization + precision analysis (don't assume "smaller = fine").** Before shipping any
  quantized model, **measure the accuracy lost**:
  - *Guard:* Llama Guard → GGUF **Q4/Q5** on CPU (Ollama/llama.cpp). Evaluate full vs
    quantized on a held-out **red-team set** ([ROBUSTNESS.md](ROBUSTNESS.md) §1.5); report the
    **per-category precision/recall delta**.
  - *VLM:* **AWQ/GPTQ/fp8** on GPU. Evaluate on the **ChartQA eval set**; report
    exact-match/accuracy delta.
  - Pick the **smallest quant that stays within an agreed accuracy budget**; weigh the delta
    against the win (latency, VRAM, image size, $). Track quality, not just size.
- **CPU guard via Ollama/llama.cpp, not CUDA-vLLM.** For the scale-to-zero CPU service, serve
  Llama Guard from Ollama/llama.cpp (GGUF) — small image (~MBs, not the ~9 GB CUDA vLLM image),
  fast enough on CPU, and `guard_llm.py` already speaks its OpenAI endpoint. Reserve
  CUDA-vLLM for a GPU LLM only.
- **Multi-stage backend image.** A `builder` stage installs deps (and optionally pre-caches
  weights); the slim runtime stage `COPY --from=builder` only the venv/site-packages + model
  cache, dropping build tooling/caches. Marginal today (no compiled deps, `--no-cache-dir`,
  CPU torch already); adopt once a compiled dep appears or the image grows.
- **Main VLM placement (TBD).** Waiting on Susanne & Omar's model → GPU service, `min=0`,
  gated behind the CPU validators.
