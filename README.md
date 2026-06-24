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
  - **Layer 1** — cheap rules (empty/junk question, image present) + a "is this a chart?"
    heuristic. Always on, no extra deps.
  - **Layer 2** — small local encoder classifiers: toxicity, prompt-injection, PII.
    *(installed by default)*
  - **Layer 3** — **Llama Guard 3** (via Ollama) as an input boundary filter for unsafe
    content / policy / jailbreak. *(install Ollama; the model is pulled automatically)*
  - Every layer is **fail-open**: if its model/service isn't installed, the app still runs —
    it just allows the request and **logs a warning** (Layer 3) so the gap is visible.

## Quickstart

**Prerequisites**

- **Python 3.12** (the backend venv targets 3.12) and **Node.js 18+** with npm.
- *Optional, for the Layer-3 safety guard:* **[Ollama](https://ollama.com)**.

**Run it (one command)**

```bash
git clone <repo> && cd Chart-Visual-QA
python app.py
```

On first run, `app.py` bootstraps **everything**: creates the Python 3.12 venv, installs the
backend requirements **and the input-guard models** (Layer 2), runs `npm install`, copies
`.env.example` → `.env`, pulls the **Layer-3** Llama Guard model if Ollama is running, prints a
**guard-readiness** report, then starts both servers. Open **http://localhost:5173**.

For the **Layer-3** safety guard, install **[Ollama](https://ollama.com)** before running — then
`app.py` pulls `llama-guard3:1b` automatically. If Ollama isn't reachable, `app.py` **warns** and
Layer 3 fails open (unsafe questions may pass). Set `GUARD_LLM_ENABLED=0` in `.env` to run without
it intentionally.

For a **light setup** (skip the heavy Layer-2 models), use `python app.py --no-guard`. Every
layer fails open and warns when its model/service is missing, so the app always runs.

## Configuration (`.env`)

All knobs live in a root **`.env`** file (created from [`.env.example`](.env.example) on first
run; gitignored). The backend loads it at startup via `python-dotenv`. Real shell env vars take
precedence. Highlights:

| Var | Default | Meaning |
| --- | --- | --- |
| `USE_MOCK` | `1` | `0` calls the real model (`model_adapter.predict`) |
| `GUARD_LLM_ENABLED` | `1` | Layer-3 Llama Guard on; falls back + warns if unreachable |
| `GUARD_LLM_URL` | `http://localhost:11434` | Ollama (dev) / vLLM (prod) endpoint |
| `GUARD_LLM_MODEL` | `llama-guard3:1b` | guard model tag |
| `GUARD_LLM_TIMEOUT` | `20` | seconds; generous enough to survive a cold model load |
| `PORT` | `5000` | backend port |

## Architecture

```
image + question
      │
      ▼
 Input Guard   L1 rules + chart heuristic → L2 encoders → L3 Llama Guard   (fail-open)
      │ allowed
      ▼
   Model (VLM)        mock now; real model behind model_adapter.predict
      │
      ▼
 short answer
```

- **frontend/** — React + Vite UI (question box, image picker, answer display).
- **backend/** — Flask API (`/api/health`, `/api/ask`), the layered guard, mock inference.
- **app.py** — dev orchestrator: bootstraps deps + boots both servers with one command.

## API

Base URL (dev): `http://127.0.0.1:5000`

| Method | Path | Body | Response |
| --- | --- | --- | --- |
| GET | `/api/health` | — | `{ "status": "ok", "mock": true }` |
| POST | `/api/ask` | `multipart/form-data`: `image=<file>`, `question=<string>` | `{ "answer": "...", "mock": true, "is_chart": true, "latency_ms": 0.0 }` |

If the guard blocks the question, `/api/ask` returns **HTTP 200** with
`{ "blocked": true, "category": "unsafe", "reason": "..." }`. Bad input returns `400` with
`{ "error": "..." }`. Uploads are capped at 10 MB.

```bash
curl http://127.0.0.1:5000/api/health
curl -F "question=What was revenue in 2024?" -F "image=@chart.png" http://127.0.0.1:5000/api/ask
```

## Useful flags

```bash
python app.py --no-guard                 # light setup: skip the heavy Layer-2 guard models
python app.py --backend-only             # just the Flask API
python app.py --frontend-only            # just the Vite dev server
python app.py --setup-only               # install deps + print guard readiness, then exit
python app.py --no-setup                 # skip the dependency check (faster restarts)
python app.py --backend-port 5001 --frontend-port 5174
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
├── app.py                # dev orchestrator (bootstrap + boot both servers + guard readiness)
├── .env.example          # config template -> copied to .env (gitignored) on first run
├── backend/
│   ├── app.py            # Flask API: /api/health, /api/ask; loads .env, boot warm-up
│   ├── inference.py      # run_inference() seam; mock vs real (USE_MOCK)
│   ├── model_adapter.py  # model team's predict() landing spot
│   ├── chart_check.py    # Layer 1: "is this a chart?" heuristic
│   ├── guard.py          # Layer 2 orchestrator (toxicity / injection / PII) + warmup()
│   ├── guard_llm.py      # Layer 3: Llama Guard via Ollama/vLLM
│   ├── requirements.txt          # base deps (Flask, requests, python-dotenv, Pillow)
│   ├── requirements-guard.txt    # opt-in Layer-2 models (torch, transformers, presidio)
│   ├── requirements-dev.txt      # pytest
│   └── tests/            # pytest: API contract + guard logic (no models needed)
├── frontend/             # React + Vite app (vite.config.js proxies /api -> backend)
└── docs/                 # PLAN.md, ROBUSTNESS.md, TASK_B_LAYER3.md, NEXT_STEPS.md
```

## Team

- **Victor & Min** — webapp (frontend + backend + input guard).
- **Susanne & Omar** — model choice, architecture, fine-tuning; integrated via `model_adapter`.
