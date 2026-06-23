# Chart-Visual-QA — Project Context

> This file is the persistent context. Read this instead of re-opening `project.md`.
> `project.md` is the full assignment brief; this file is the condensed working version.

## Goal
Build a **Visual Question Answering (VQA)** system for **charts**: given a chart image
and a natural-language question, return a short answer (1–10 words). Compare a
**zero-shot VLM baseline** vs a **fine-tuned VLM** on the same eval set, with an error
analysis. Dataset: **ChartQA** (`lmms-lab/ChartQA`).

## Team & ownership
- **Victor** (me) — webapp (React UI + Flask backend); pair-programming with **Min**.
- **Min** — webapp pair-programming with Victor.
- **Susanne & Omar** — model choice, architecture, fine-tuning. Integrated at the end.

The webapp is built **mock-first**: the Flask backend returns fake answers behind a
stable API contract. When Susanne & Omar's model is ready, we swap the mock for the
real inference call without touching the frontend.

## System architecture (target)
`Input (image + question) → Preprocess → Model (VLM) → Postprocess (optional) → Output (short answer)`
- **Baseline 1**: VLM alone (zero-shot).
- **Baseline 2**: same VLM, LoRA fine-tuned on ChartQA.
- **Our approach**: add a preprocess step (and maybe postprocess) around the model.

## Repo / git workflow
- Active branch: **`feat/webapp`**. `main` is protected.
- **Do NOT commit or push to `main`** — a PreToolUse hook in `.claude/settings.json`
  blocks it. Always work on a feature branch and open a PR.
- Commit/push only when explicitly asked.

## Conventions
- All code, comments, docs, and commit messages in **English**.
- Python 3.10+ for the backend. React (Vite) for the frontend.
- Planned layout (once approved):
  - `frontend/` — React app (question box + image picker + answer display).
  - `backend/`  — Flask API (`/api/health`, `/api/ask`), mock inference first.
  - `docs/PLAN.md` — experimentation + implementation plan.

## Webapp UX (Victor's scope)
- A text field to type the **question**.
- A clickable **image-picker** icon that opens the OS file explorer to choose a chart image.
- A **submit** action that sends `{image, question}` to the backend and shows the answer.
