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
image + question
      │
      ▼
 Input Guard      (layered: cheap rules → small classifiers → LLM classifier)  — §6
      │ allowed
      ▼
 Preprocess       (image resize/normalize; optional RAG prompt building)
      │
      ▼
 Model            Baseline 1: VLM zero-shot · Baseline 2: VLM + LoRA          — model team
      │           (served as its own inference service — §8)
      ▼
 Postprocess      (numeric parsing, unit normalization, answer cleanup)
      │
      ▼
 short answer
```

Two tracks, one seam (`run_inference`, §9):

| Track | Owner | Scope |
| --- | --- | --- |
| Experimentation (model) | **Susanne & Omar** | VLM choice, LoRA fine-tune, eval, error analysis |
| Implementation (webapp) | **Victor & Min** | React UI + Flask API, mock-first, input guard, serving glue |

## 3. Status checklist

**Webapp (Victor & Min)**
- [x] Phase A — scaffold: `frontend/` (React+Vite), `backend/` (Flask), CORS, `.gitignore`
- [x] Phase A — `app.py` orchestrator: auto-bootstrap deps + run both servers
- [x] Phase B — mock backend: `/api/health`, `/api/ask`, deterministic mock answer
- [x] Phase C — frontend UI: image picker + drag/drop, preview, question, submit, loading/error
- [x] Phase D — wire-up: fetch, abort-on-resubmit, 413, empty-input handling
- [x] Phase E — integration seam: `run_inference`, `USE_MOCK` env, `model_adapter.py`
- [x] Restyle to design system (DESIGN.md / Voltagent language)
- [x] Backend contract tests (pytest, 6 passing)
- [x] Docs: README, PLAN, ROBUSTNESS
- [ ] Integrate **Min's Layer-1 guard** (rebase branch `min` onto `feat/webapp` — §10)
- [ ] **Layer 2** encoder classifiers (toxicity / prompt-injection / PII)
- [ ] **Layer 3** LLM classifier (on-topic + safety), Ollama → vLLM
- [ ] Hardening: answer cache, rate-limit, upload safety (PIL verify), observability, eval harness
- [ ] Deployment: vLLM serving for guard + VLM, env-configured URLs (§8)

**Decisions locked**
- [x] **No LLM agent** — most expensive path, out of cost budget (§7)
- [x] **Serving:** vLLM now → llm-d only at multi-node scale (§8)
- [x] **Guard:** 3 layers, LLM used as a *classifier* only in the last layer (§6)

**Model track (Susanne & Omar — for visibility)**
- [ ] Load & inspect ChartQA · [ ] Baseline 1 zero-shot · [ ] Baseline 2 LoRA
- [ ] Per-type breakdown + 20-case error analysis
- [ ] Implement `model_adapter.predict` behind the seam

## 4. Experimentation track — Susanne & Omar

1. **Load & inspect ChartQA.** Question types, answer formats, choose metrics.
2. **Baseline 1 — VLM zero-shot.** Pick a VLM (BLIP-2 / LLaVA / Qwen2-VL). Prompt
   `"Question: ... Answer:"`, generate, record accuracy.
3. **Baseline 2 — VLM + LoRA fine-tune.** Attach LoRA to the LM side, train 1–2 epochs,
   re-evaluate on the same val set.
4. **Our approach — preprocess (+ maybe postprocess).** Add preprocessing around the
   model; compare against both baselines.
5. **Breakdown + error analysis.** Per-question-type accuracy; 20 failure cases grouped
   by failure mode.
6. **Implement the seam** — `model_adapter.predict(image_bytes, question) -> str` (§9).

**Deliverables:** notebook with both pipelines, results table, 20-case error analysis, README.

> **Dataset note (from a streaming peek at `lmms-lab/ChartQA`, fields
> `type/question/answer/image`):** answers are **1–5 words (mean 1.1)** and **~67% contain a
> digit**. ⇒ confirms the metric choice (exact-match + numeric ±5%), and the real questions
> double as positive examples for the Layer-3 on-topic classifier (§6).

## 5. Implementation track — Victor & Min (webapp)

Build the full UI + API against a **mock** backend, then swap in the real model with no
frontend changes. Phases A–E are **done** (see checklist §3); the remaining webapp work is
the input guard (§6) and hardening/deployment (§7–§8).

- **Phase A — Scaffold.** `frontend/` (React + Vite), `backend/` (Flask), dev scripts, CORS.
- **Phase B — Backend (mock).** `GET /api/health`; `POST /api/ask` (multipart) → canned answer.
- **Phase C — Frontend.** Question field, image picker (file explorer + drag/drop), preview,
  submit, answer display, loading + error states.
- **Phase D — Wire-up.** Frontend → `/api/ask`; abort-on-resubmit, 413, empty-input handling.
- **Phase E — Integration-ready.** All inference behind `run_inference(image, question)` +
  `model_adapter.py`; `USE_MOCK` env toggle.

**`app.py` orchestrator (single entry point).** `python app.py` boots backend + frontend as
subprocesses, interleaves their logs, and shuts both down cleanly on Ctrl-C. On first run it
self-bootstraps deps (Python 3.12 venv + `npm install`). Flags: `--backend-only` /
`--frontend-only` / `--setup-only` / `--no-setup` / port overrides. One place to wire env
vars (mock vs real, guard/VLM service URLs).

## 6. Layered input guard (robustness)

Full design in [ROBUSTNESS.md](ROBUSTNESS.md) §1. Validate the question/image **before** the
model, in three layers. The LLM appears **only in the last layer**, used as a *classifier*
(not an "LLM-judge"). Build order: Layer 1 → Layer 2 → Layer 3.

- **Layer 1 — cheap rules, no ML (µs):** empty / length / charset checks, image present,
  per-IP rate limit, cheap "is this a chart?" heuristic. ⇒ **Min already built a first
  version** (`backend/chart_check.py` + validation rules in `app.py`) on branch `min` — needs
  rebasing onto `feat/webapp` (§10).
- **Layer 2 — small encoder classifiers, local, ms (not LLMs):** toxicity
  (`unitary/toxic-bert` / Detoxify), prompt-injection
  (`protectai/deberta-v3-base-prompt-injection-v2`), PII (Microsoft Presidio).
- **Layer 3 — LLM as a classifier, only on ambiguous/novel cases:** "is this a chart
  question?" + safety catch-all. Open-source: **Llama Guard 3 1B** (safety) +
  **Qwen2.5-1.5B-Instruct** (on-topic); alternatives Granite Guardian 3, ShieldGemma. Called
  over HTTP at an **env-configured URL** (Ollama in dev, vLLM in prod — §8), so the code path
  is identical across environments.

**Contract impact:** additive `blocked` response shape; tiny frontend branch. Fail-closed for
safety categories, fail-open with a hint for "off-topic".

## 7. Performance & inference-cost budget

Constraint: **keep per-request inference cost small.** The model itself (VLM choice,
quantization, decoding, fine-tuning) is **owned by Susanne & Omar** — not webapp scope.

**What the webapp controls (our job):**
1. **Fail fast** — the guard rejects junk / no-image before `run_inference`.
2. **Cache answers** by `sha256(image) + question` (already deterministic) — repeats cost 0.
3. **Downsize the upload** before the seam — don't pass a 10 MB image through.
4. **Cheap guard on average** — Layers 1–2 are µs–ms; Layer 3 runs only on ambiguous cases,
   on a small local model (no per-token API bill).
5. **No agent** (decision below).
6. `MOCK_DELAY_S=0` outside demos (UI artifact, not real latency).

**Handed to the model team (their decision, we just surface the signal):** answers are 1–5
words (mean 1.1), so a small generation cap + greedy decoding and a small quantized model are
likely enough for the budget — input to their cost choices, not webapp work.

**Decision: no LLM agent.** We will not build the multi-step tool-using agent
([ROBUSTNESS.md](ROBUSTNESS.md) §3) — most expensive path, outside the budget.

## 8. Deployment, serving & LLMOps

Today everything runs **local** (Ollama for any LLM, mock for the VLM). For a future deploy:

**vLLM vs llm-d — they don't compete; llm-d runs vLLM underneath.**

| | **vLLM** | **llm-d** |
| --- | --- | --- |
| What | Inference *engine* (PagedAttention, continuous batching, OpenAI-compatible API) | K8s-native *orchestration* over vLLM (prefill/decode disaggregation, KV-cache-aware routing, multi-node autoscale) |
| Scale | 1 node (multi-GPU via tensor-parallel) | Cluster, high QPS |
| Use when | Default for almost everything | Only when one node can't keep up |

→ **Use vLLM.** llm-d is for real multi-node scale later; adopting it now is over-engineering.
Alternatives at the vLLM tier: SGLang (great for the classifier's structured outputs), TGI,
TensorRT-LLM/Triton.

**Trajectory:** `Ollama (local)` → `2× vLLM services on 1 node (prod)` → `llm-d (only at scale)`.
The seam never changes — only the env-configured URLs do.

**How the two models run — two independent services, never co-located:**

| | Guard (Layer 3) | Main VLM |
| --- | --- | --- |
| Size | 1–3B | larger, multimodal |
| Frequency | only the *ambiguous* subset | every *allowed* request |
| Output | tiny (label / JSON) | short (1–5 words) |
| Serving | vLLM, **structured/guided decoding**, kept warm | vLLM, **scale-to-zero** if bursty |

The Flask backend calls both over HTTP (`GUARD_URL`, `VLM_URL`): `guard()` → guard service,
`run_inference()` → VLM service. Answer cache sits in front of the VLM.

**LLMOps practices:**
- **Versioning** — pin VLM + LoRA + guard-policy versions; registry (HF Hub / MLflow / W&B).
- **CI/CD eval gate** — promote a model only if it passes the golden set (exact-match + numeric
  ±5%); roll out via canary / shadow / blue-green. Extends `backend/tests/`.
- **Observability** — per-stage latency (guard vs VLM), tokens, GPU util, cache hit-rate,
  request-id tracing (Prometheus/Grafana + OpenTelemetry; Langfuse/Arize for LLM traces).
- **Quality/drift** — track guard block-rate, off-topic rate, answer distribution, OOD inputs;
  guard false-positive rate is a first-class KPI, with human review of blocks.
- **Cost** — continuous batching, autoscaling (KEDA/HPA), scale-to-zero on the VLM, semantic
  cache, spot instances for batch eval.
- **Data flywheel** — log `(image hash, question, answer, block reason)` → feeds error analysis,
  improves the on-topic classifier, surfaces drift (mind PII — Layer 2 redacts).
- **Config/secrets** — env / secret manager; model artifacts in object storage; service URLs
  injected per environment.

## 9. Integration contract (the seam between the two tracks)

```
POST /api/ask        (multipart/form-data)
  fields: image=<file>, question=<string>
  ->  200 { "answer": <string>, "mock": <bool>, "latency_ms": <number> }
      400 { "error": <string> }
      200 { "blocked": true, "category": <string>, "reason": <string> }   (guard)
```

The model team implements `model_adapter.predict(image, question) -> str`; the backend calls
it through `run_inference`. The mock just returns a fixed string. Swapping mock → real model
touches only that one function.

## 10. Out of scope & housekeeping

- **Model decisions** (selection, fine-tuning, architecture) are owned by Susanne & Omar; the
  webapp ships mocked until `model_adapter.predict` lands.
- **Min's PR — branch hygiene.** Her PR currently targets **`main`** (protected) but must
  target **`feat/webapp`**. Her branch was cut from `1f6f4af` (Phases A–C), so it predates the
  Phase D–E restyle and will conflict on the frontend (`App.jsx/App.css/index.css` rewritten)
  and lightly on `backend/app.py` / `inference.py`. Action: retarget the PR base to
  `feat/webapp`, rebase, keep her backend Layer-1 logic re-applied on top of current `app.py`.

## 11. Milestones

- **M1** Webapp scaffold + mock backend running locally. ✅
- **M2** Full UI working end-to-end against the mock. ✅
- **M3** Input guard (Layers 1–3) integrated.
- **M4** Model team's `model_adapter.predict` ready (baselines compared).
- **M5** Integration: real model behind the same contract + serving (vLLM).
- **M6** Results table + error analysis + README.
