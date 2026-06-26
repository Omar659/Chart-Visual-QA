# Chart-Visual-QA

A **Visual Question Answering (VQA)** system for **charts**: give it a chart image and a
natural-language question, get back a short answer (1–10 words). The project compares a
zero-shot VLM baseline against a LoRA fine-tuned VLM on the **ChartQA** dataset.

This repo ships the **webapp** (React UI + Flask API), built **mock-first**: the backend
returns fake answers behind a stable API contract so the UI and the safety layers can be
built and tested before the model is ready. Swapping in the real model touches a single
function (`backend/model_adapter.py::predict`). See [docs/PLAN.md](docs/PLAN.md) for the
full plan and [docs/ROBUSTNESS.md](docs/ROBUSTNESS.md) for the guard design.

## What's built

- **Webapp** — React (Vite) UI + Flask API, mock-first behind a stable `/api/ask` contract.
- **Layered input guard** — screens the question/image *before* the model
  ([docs/ROBUSTNESS.md](docs/ROBUSTNESS.md) §1):
  - **Layer 1** — cheap rules (empty/junk question, image present). Always on, no extra deps.
  - **Chart gate** — **CLIP zero-shot** detector ("is this a chart?") + an OCR "has data
    values" check (Tesseract), with a pixel-heuristic fallback when CLIP/torch isn't present.
  - **Layer 2** — small local encoder classifiers: toxicity, prompt-injection, PII.
  - **Layer 3** — **Llama Guard 3** as a semantic input boundary filter for unsafe content /
    policy / jailbreak, over an **OpenAI-compatible API** (Ollama on CPU, or vLLM on GPU).
  - Every layer is **fail-open**: if its model/service isn't available, the app still runs —
    it just allows the request and **logs a warning** so the gap is visible.

## Quickstart

Two ways to run — full design in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

### Production (default) — containerized

```bash
git clone <repo> && cd Chart-Visual-QA
python app.py
```

`python app.py` runs the **production stack in Docker**: the **backend** container (Flask +
CLIP + Tesseract + Layer-2 encoders, all baked into the image) and the **guard** container
(Llama Guard 3 on **CPU via Ollama**, model baked in), plus a **local Vite frontend**. On
first run it creates `.env` from `.env.example`, runs `npm install`, builds the images (slow
the first time — it compiles the ML stack), and starts everything. Open **http://localhost:5173**.

**Prerequisites:** **Docker Desktop** (running) and **Node.js 18+**. **No GPU and no host
Ollama** — the guard runs on CPU inside its container. If Docker isn't running, `app.py`
**warns and stops** (use `--dev` instead).

### Local dev — no Docker

```bash
python app.py --dev
```

Runs the backend in a local **Python 3.12 venv** (CLIP + Layer-2 models), guards local, and the
Vite frontend — bootstrapping the venv, models, `npm install` and `.env` on first run.
**Prerequisites:** Python 3.12 + Node.js 18+. For **Layer 3** in dev, point `GUARD_LLM_URL` at
any OpenAI-compatible guard — e.g. start just the containerized guard with `docker compose up
guard` (served on `localhost:11434`). Without one, Layer 3 fails open and warns; set
`GUARD_LLM_ENABLED=0` to disable it intentionally. Light setup: `python app.py --dev --no-guard`.

## Configuration (`.env`)

All knobs live in a root **`.env`** file (created from [`.env.example`](.env.example) on first
run; gitignored). The backend loads it at startup via `python-dotenv`. **Every key is required —
the backend has no in-code defaults**, so `.env.example` is the canonical list. Real shell env
vars take precedence. Highlights:

| Var | Example | Meaning |
| --- | --- | --- |
| `USE_MOCK` | `1` | `0` calls the real model (`model_adapter.predict`) |
| `GUARD_LLM_ENABLED` | `1` | Layer-3 Llama Guard on; falls back + warns if unreachable |
| `GUARD_LLM_URL` | `http://localhost:11434` | OpenAI-compatible guard endpoint (dev) |
| `GUARD_LLM_MODEL` | `llama-guard3:1b` | guard model tag |
| `TESSERACT_CMD` | *(empty)* | path to the Tesseract binary; empty = use `PATH` |
| `PORT` / `HOST` | `5000` / `127.0.0.1` | backend bind |

In **production** (`python app.py`), `docker-compose.yml` **overrides** a few of these for the
backend container — `GUARD_LLM_URL=http://guard:11434` (the guard service), `HOST=0.0.0.0`,
`FLASK_DEBUG=0`, `TESSERACT_CMD=` (apt binary on `PATH`) — so the same `.env` works for both modes.

## Architecture

```
image + question
      │
      ▼
 Input Guard   L1 rules → chart gate (CLIP + OCR) → L2 encoders → L3 Llama Guard   (fail-open)
      │ allowed
      ▼
   Model (VLM)        mock now; real model behind model_adapter.predict
      │
      ▼
 short answer
```

In **production** this is two containers — **backend** (Flask + CLIP + Tesseract + L2) and
**guard** (Llama Guard) — wired by `docker-compose.yml`; the backend reaches the guard over the
Docker network at `GUARD_LLM_URL`. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and the
[deploy validation report](docs/CPU_DEPLOY_REPORT.md).

- **frontend/** — React + Vite UI (question box, image picker, answer display).
- **backend/** — Flask API (`/api/health`, `/api/ask`), the chart gate + layered guard, mock inference.
- **guard/** — Dockerfile for the CPU Llama Guard (Ollama) image, model baked in.
- **app.py** — orchestrator: `python app.py` (prod, Docker) / `python app.py --dev` (local).

## API

Base URL (dev): `http://127.0.0.1:5000`

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| GET | `/api/health` | — | `{ "status": "ok", "mock": true }` |
| POST | `/api/ask` | `multipart/form-data`: `image=<file>`, `question=<string>` | `{ "answer"\|"disclaimer": "...", "mock": true, "is_chart": true, "chart_confidence": 0.99, "latency_ms": 0.0 }` |

In **mock mode** (default), `/api/ask` returns a `disclaimer` instead of a fake `answer` (set
`MOCK_REVEAL=1` to get the canned answer, or `USE_MOCK=0` for the real model). If the guard
**blocks** the question it returns **HTTP 200** with
`{ "blocked": true, "category": "...", "reason": "..." }`. Bad input returns `400` with
`{ "error": "..." }`. Uploads are capped at `MAX_UPLOAD_MB` (10 MB).

```bash
curl http://127.0.0.1:5000/api/health
curl -F "question=What was revenue in 2024?" -F "image=@chart.png" http://127.0.0.1:5000/api/ask
```

## Useful flags

```bash
python app.py                            # PRODUCTION: containers (backend + guard) + local frontend
python app.py --dev                      # LOCAL DEV: venv backend + CLIP + Vite, no Docker
docker compose down                      # stop the production containers

# dev-mode flags (with --dev):
python app.py --dev --no-guard           # light setup: skip the heavy Layer-2 guard models
python app.py --dev --backend-only       # just the Flask API
python app.py --dev --frontend-only      # just the Vite dev server
python app.py --dev --setup-only         # install deps + print guard readiness, then exit
python app.py --dev --no-setup           # skip the dependency check (faster restarts)
```

## Tests

```bash
cd backend
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # once  (macOS/Linux: .venv/bin/python)
.venv/Scripts/python.exe -m pytest
```

Tests run **without** any guard models or Ollama — the guard HTTP/model calls are monkeypatched,
so CI stays fast and green on a clean machine.

## Swapping the mock for the real model

Inference goes through one seam — the model team only touches **`backend/model_adapter.py`**:

1. Implement `predict(image_bytes: bytes, question: str) -> str` (add runtime deps to
   `backend/requirements.txt`).
2. Set `USE_MOCK=0` in `.env` (or env). `backend/inference.py` only imports the adapter when
   `USE_MOCK` is off, so the app boots in mock mode without any ML deps. No frontend/API changes.

## Project layout

```
Chart-Visual-QA/
├── app.py                # orchestrator: prod (docker compose) / --dev (local venv)
├── docker-compose.yml    # prod stack: backend + guard containers
├── .env.example          # config template -> copied to .env (gitignored) on first run
├── backend/
│   ├── Dockerfile        # Flask + CLIP + Tesseract + L2 encoders (models baked in)
│   ├── app.py            # Flask API: /api/health, /api/ask; loads .env, boot warm-up
│   ├── env_config.py     # required-env helpers (no in-code config defaults)
│   ├── inference.py      # run_inference() seam; mock vs real (USE_MOCK)
│   ├── model_adapter.py  # model team's predict() landing spot
│   ├── chart_check.py    # chart gate: CLIP zero-shot + OCR "has data", heuristic fallback
│   ├── guard.py          # Layer 2 orchestrator (toxicity / injection / PII) + warmup()
│   ├── guard_llm.py      # Layer 3: Llama Guard over the OpenAI /v1 API (Ollama / vLLM)
│   ├── requirements.txt          # backend + CLIP/OCR deps (torch, transformers, pytesseract)
│   ├── requirements-guard.txt    # Layer-2 models (detoxify, presidio)
│   └── tests/            # pytest: API contract + guard logic (no models / server needed)
├── guard/
│   └── Dockerfile        # CPU Llama Guard via Ollama, model baked into the image
├── frontend/             # React + Vite app (vite.config.js proxies /api -> backend)
└── docs/                 # ARCHITECTURE.md, CPU_DEPLOY_REPORT.md, PLAN.md, ROBUSTNESS.md, ...
```

## Team

- **Victor & Min** — webapp (frontend + backend + input guard).
- **Susanne & Omar** — model choice, architecture, fine-tuning; integrated via `model_adapter`.
