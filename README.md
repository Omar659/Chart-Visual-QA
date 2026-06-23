# Chart-Visual-QA

A **Visual Question Answering (VQA)** system for **charts**: give it a chart image and a
natural-language question, get back a short answer (1–10 words). We compare a zero-shot
VLM baseline against a LoRA fine-tuned VLM on the **ChartQA** dataset.

This repo currently ships the **webapp** (React UI + Flask API), built **mock-first**: the
backend returns fake answers behind a stable API contract so the UI can be built and tested
before the model is ready. Swapping in the real model touches a single function
(`backend/inference.py::run_inference`). See [docs/PLAN.md](docs/PLAN.md) for the full plan.

## Architecture

```
Input (image + question) → Preprocess → Model (VLM) → Postprocess → Output (short answer)
```

- **frontend/** — React + Vite UI (question box, image picker, answer display).
- **backend/** — Flask API (`/api/health`, `/api/ask`), mock inference for now.
- **app.py** — dev orchestrator that boots both servers with one command.

## Prerequisites

- **Python 3.12** (the backend venv targets 3.12 specifically)
- **Node.js 18+** and npm

Verify:

```bash
py -3.12 --version    # Windows (py launcher)
python3.12 --version  # macOS/Linux
node --version
npm --version
```

## Setup

**The easy way:** just run `python app.py` (see below) — on first run it creates the
backend virtualenv, installs the Python requirements, and runs `npm install` automatically.
Use `python app.py --setup-only` to install everything without starting the servers.

The manual steps below are only needed if you prefer to set things up yourself.

**1. Backend (Flask) — create a 3.12 virtualenv and install deps:**

```bash
# Windows (PowerShell or Git Bash)
cd backend
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
cd ..
```

```bash
# macOS / Linux
cd backend
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd ..
```

**2. Frontend (React + Vite) — install npm packages:**

```bash
cd frontend
npm install
cd ..
```

## Running

**One command (recommended)** — the orchestrator installs deps if needed, then boots
backend + frontend together, prefixes their logs, and shuts both down on Ctrl-C:

```bash
python app.py
```

Then open the frontend at **http://localhost:5173**. The backend is at
**http://localhost:5000** and the Vite dev server proxies `/api/*` to it.

Useful flags:

```bash
python app.py --backend-only             # just the Flask API
python app.py --frontend-only            # just the Vite dev server
python app.py --setup-only               # install deps and exit
python app.py --no-setup                 # skip the dependency check (faster restarts)
python app.py --backend-port 5001 --frontend-port 5174
```

**Or run each side manually** (two terminals):

```bash
# terminal 1 — backend
cd backend && .venv/Scripts/python.exe app.py        # Windows
cd backend && .venv/bin/python app.py                # macOS/Linux

# terminal 2 — frontend
cd frontend && npm run dev
```

## API

Base URL (dev): `http://127.0.0.1:5000`

| Method | Path          | Body                                              | Response |
| ------ | ------------- | ------------------------------------------------- | -------- |
| GET    | `/api/health` | —                                                 | `{ "status": "ok", "mock": true }` |
| POST   | `/api/ask`    | `multipart/form-data`: `image=<file>`, `question=<string>` | `{ "answer": "...", "mock": true, "latency_ms": 0.0 }` |

Errors return `400` with `{ "error": "..." }`. Uploads are capped at 10 MB.

Quick check:

```bash
curl http://127.0.0.1:5000/api/health
curl -F "question=What was revenue in 2024?" -F "image=@chart.png" http://127.0.0.1:5000/api/ask
```

## Tests

Backend contract tests (lock the `/api/health` + `/api/ask` request/response shape):

```bash
cd backend
.venv/Scripts/python.exe -m pip install -r requirements-dev.txt   # once
.venv/Scripts/python.exe -m pytest                                # Windows
# macOS/Linux: .venv/bin/python -m pytest
```

## Swapping the mock for the real model

All inference goes through one seam — the model team only touches **`backend/model_adapter.py`**:

1. Implement `predict(image_bytes: bytes, question: str) -> str` in `backend/model_adapter.py`
   (add the model's runtime deps to `backend/requirements.txt`).
2. Run the backend with the `USE_MOCK=0` env var — no code edit needed:

   ```bash
   USE_MOCK=0 python app.py          # macOS/Linux / Git Bash
   $env:USE_MOCK=0; python app.py    # PowerShell
   ```

`backend/inference.py` only imports the adapter when `USE_MOCK` is off, so the app boots
fine in mock mode without any ML dependencies installed. No frontend or API changes are
required — the contract stays the same.

## Project layout

```
Chart-Visual-QA/
├── app.py              # dev orchestrator (boots backend + frontend)
├── backend/
│   ├── app.py            # Flask API: /api/health, /api/ask
│   ├── inference.py      # run_inference() seam; mock vs real (USE_MOCK)
│   ├── model_adapter.py  # model team's predict() landing spot
│   ├── tests/            # pytest contract tests for the API
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── .venv/            # Python 3.12 virtualenv (gitignored)
├── frontend/           # React + Vite app
│   ├── src/
│   ├── vite.config.js  # dev server + /api proxy to the backend
│   └── package.json
└── docs/PLAN.md        # experimentation + implementation plan
```

## Team

- **Victor & Min** — webapp (this repo's frontend + backend).
- **Susanne & Omar** — model choice, architecture, fine-tuning; integrated via `run_inference`.
