# Chart-Visual-QA — Experimentation & Implementation Plan

## 1. Objective

A Visual Question Answering system for **charts**. Input: a chart image + a question.
Output: a short answer. We measure a zero-shot VLM against a fine-tuned VLM and ship a
small web UI on top.

- **Dataset:** ChartQA (`lmms-lab/ChartQA`) — chart reasoning.
- **Metrics:** Exact-match for text answers, numeric tolerance (±5%) for numeric answers,
  plus per-question-type breakdown (single value, comparison, calculation, trend).

## 2. Architecture

```
Input (image + question)
        │
        ▼
   Preprocess            (image resize/normalize; optional OCR; prompt building)
        │
        ▼
     Model               Baseline 1: VLM zero-shot
                         Baseline 2: same VLM + LoRA fine-tune
                         Approach:   VLM + our preprocess (+ optional postprocess)
        │
        ▼
  Postprocess (maybe)    (answer cleanup, numeric parsing, unit normalization)
        │
        ▼
   Output (short answer)
```

## 3. Experimentation track — Susanne & Omar

1. **Load & inspect ChartQA.** Question types, answer formats, choose metrics.
2. **Baseline 1 — VLM zero-shot.** Pick a VLM (BLIP-2 / LLaVA / Qwen2-VL). Prompt
   `"Question: ... Answer:"`, generate, record accuracy.
3. **Baseline 2 — VLM + LoRA fine-tune.** Attach LoRA to the LM side, train 1–2 epochs,
   re-evaluate on the same val set.
4. **Our approach — preprocess (+ maybe postprocess).** Add preprocessing around the
   model; compare against both baselines.
5. **Breakdown + error analysis.** Per-question-type accuracy; 20 failure cases grouped
   by failure mode.
6. **Define the inference contract** the backend will call (see §5) so integration is a
   drop-in swap.

**Deliverables:** notebook with both pipelines, results table, 20-case error analysis, README.

## 4. Implementation track — Victor & Min (webapp, mock-first)

We build the full UI + API now against a **mock** backend, then swap in the real model.

- **Phase A — Scaffold.** `frontend/` (React + Vite), `backend/` (Flask), dev scripts, CORS.
- **Phase B — Backend (mock).** `GET /api/health`; `POST /api/ask` accepts an image +
  question (multipart) and returns a canned `{ "answer": "...", "mock": true }`.
- **Phase C — Frontend.** Question text field, clickable image-picker icon (opens file
  explorer), preview, submit, answer display, loading + error states.
- **Phase D — Wire-up.** Frontend → `/api/ask`; handle errors and empty inputs.
- **Phase E — Integration-ready.** Isolate inference behind one function
  (`run_inference(image, question)`) so Susanne & Omar's model replaces the mock with no
  frontend changes.

### Phase A detail — `app.py` orchestrator (single entry point)

A root-level **`app.py`** is the one command that boots the whole stack for local dev.
It shells out to the frontend and backend dev servers so a contributor runs **one** thing
instead of juggling two terminals.

- **Responsibility:** start the Flask backend and the Vite frontend dev server as
  subprocesses (via `subprocess`/`shell` commands), stream their logs, and shut both down
  cleanly on Ctrl-C.
- **Commands it runs (illustrative):**
  - backend: `flask --app backend/app run --port 5000` (or `python backend/app.py`)
  - frontend: `npm --prefix frontend run dev`
- **Behaviour:**
  - Launch both, prefix/interleave their output so failures are visible.
  - Propagate signals — killing `app.py` terminates both child processes (no orphans).
  - Optional flags: `--backend-only` / `--frontend-only` for focused work; `--port` /
    `--api-url` overrides for the dev ports.
  - Cross-platform note: handle Windows vs POSIX process termination (we develop on
    Windows; CI/Colab is POSIX).
- **Why:** keeps the run story trivial (`python app.py`) and gives us one place to wire in
  env vars (e.g. mock vs real inference) once Susanne & Omar's model lands.

## 5. Integration contract (the seam between the two tracks)

```
POST /api/ask        (multipart/form-data)
  fields: image=<file>, question=<string>
  ->  200 { "answer": <string>, "mock": <bool>, "latency_ms": <number> }
      400 { "error": <string> }
```

The model team implements `run_inference(image, question) -> str`. The backend calls it;
the mock just returns a fixed string. Swapping the mock for the real model touches only
that one function.

## 6. Milestones

- **M1** Webapp scaffold + mock backend running locally.
- **M2** Full UI working end-to-end against the mock.
- **M3** Model team's `run_inference` ready (baselines compared).
- **M4** Integration: real model behind the same contract.
- **M5** Results table + error analysis + README.

## 7. Out of scope for now

Model selection, fine-tuning, and final architecture decisions are owned by Susanne &
Omar; the webapp ships mocked until their `run_inference` lands.
