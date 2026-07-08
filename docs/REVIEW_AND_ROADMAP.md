# Chart-Visual-QA — Architecture Review & Phased Roadmap

> Status snapshot after merging `feat/webapp` + `feat/model` into `main`.
> Scope: architecture review, integration audit, pipeline state, a local test plan
> for an **RTX 4050 (6 GB)**, MLOps (MLflow tracking + registry), deploy hardening,
> observability (Prometheus + Grafana), data layer (Postgres + Redis), security & cost
> control for a **public portfolio deploy**, Llama-Guard fine-tuning, a v2.0
> conversational upgrade, and a **master execution checklist** (stack, deploy workflow,
> fine-tuning workflow). Checklists are grouped by phase. `[x]` = done today,
> `[ ]` = proposed work.

---

## Execution progress (updated 2026-07-08)

Work lives on branch **`feat/quantization-flag`** (not pushed). Backend suite: **63 passed /
4 skipped**.

**v1.0 modeling & reproducibility (done):**
- **1.3 quantization flag** ✅ — `--quantization {none,8bit,4bit}` / `QWEN_QUANTIZATION`,
  NF4+double-quant+bf16, LoRA stays attached (no merge) when quantized, fail-loud w/o bnb.
- **1.2 analysis pipeline validated** ✅ — 8/8 result JSONs byte-identical from committed dumps.
- **§1.2b Windows segfault FIXED** ✅ — `datasets` before `torch` in `chartqa_dataset.py`
  (pyarrow DLL crash, exit 139 → 0); eval + analysis run on Windows.
- **2.1 de-dup** ✅ — root `model/`+`data/` deleted, `modeling/` installable, `backend/`
  copy CI-drift-checked.
- **2.2 reproducibility** ✅ — per-model LoRA r/α (qwen 16/32, blip2 32/64), qwen modules
  cut to the committed 7 LM-only, `max_steps` recovered (qwen **200** from `training_args.bin`,
  blip2 **884** from `trainer_state.json`), MODELCARDs w/ HF-Trainer-vs-custom-loop caveats.
- **2.3 MLflow local tracking** ✅ (eval path) — fail-open `chartqa/tracking.py`; `evaluate.py`
  logs params + accuracy + latency/VRAM/load/size. Deferred: `finetune_lora.py` instrumentation,
  MLflow **server + registry** (→ 3.6).

**Phase 1.5 quantization study (in progress):**
- **Stage A** ✅ — BLIP-2 4-bit smoke ran locally end-to-end (real model, real 4-bit, tracked).
- **Stage B2** ✅ (verdict) — **Qwen-8B 4-bit does NOT fit the 6 GB 4050**; CPU/disk offload
  hits a bnb-4bit+accelerate meta-tensor bug (the `QWEN_CPU_OFFLOAD` attempt was reverted).
  **8B is cloud-only `min=0`.**
- **Stage B1** ⏳ (user, on Kaggle T4) — Qwen 4-bit zero-shot + fine-tuned accuracy, 2500
  samples, via `modeling/notebooks/kaggle_quant_eval.ipynb` (+ `KAGGLE_GUIDE.md`). → fills the
  comparison table (Δ vs bf16 84.60/86.08).

**Phase 3 deploy hardening (done this session):**
- **3.1 serving seam** ✅ — `model_adapter.predict` routes to a remote VLM over HTTP when
  `VLM_URL` set (else in-process); new **`vlm_service/`** (Flask wrapper around the
  single-source `QwenVLChat`, `/predict` + `/health`). Tested vs stub. Remaining: GPU
  Dockerfile + compose (needs CUDA host).
- **3.2 frontend container** ✅ **Docker-verified** — `frontend/Dockerfile` (node build →
  nginx), `nginx.conf` (SPA + `/api` proxy via lazy Docker DNS), `frontend` compose service
  (8080:80). Image builds, `nginx -t` ok, serves the SPA (GET / → 200).
- **3.3 backend image** ✅ — `.dockerignore` excludes `.venv/`/`.env*`/tests. Multi-stage optional.
- **3.4 request hardening** ✅ (cache + upload) — `answer_cache.py` (fail-open LRU+TTL,
  short-circuits guard+chart+VLM) + `uploads.py` `sanitize_image` (re-encode strips payloads).
  Rate-limit + evasion still TODO.
- **3.5 observability** ✅ — `metrics.py` (fail-open Prometheus) + `/metrics`; per-stage
  latency (guard/chart_gate/vlm) + counters (requests, blocked, cache, `vlm_invocations`).
  Prometheus/Grafana **containers + dashboard** = the immediate next piece.
- **3.7 security** ✅ (partial) — CORS pinned via `CORS_ORIGINS` + security headers.

**Open items:** env pollution (`datasets`/`mlflow`/`bitsandbytes`/`accelerate`/`peft`/
`prometheus-client` in `backend/.venv`; no dedicated modeling venv); dataset-source decision
(HuggingFaceM4 vs lmms-lab); nothing pushed.

---

## 0. Executive summary

- **The system is code-complete and well-architected for v1.0**, with one real gap: the
  main VLM is *wired* (real `model_adapter.predict`) but **not deployable** yet — the
  backend Docker image is CPU-only and the planned **GPU VLM service (scale-to-zero) does
  not exist**. Everything else (3-layer guard, chart gate, frontend, mock contract,
  eval/error-analysis) is integrated and tested.
- **Best model = Qwen3-VL-8B-Instruct + LoRA**: **84.6% → 86.08%** relaxed accuracy on the
  2500-sample ChartQA test split (zero-shot → fine-tuned). BLIP-2 is the weak baseline
  (8.4% → 12.4%).
- **RTX 4050 (6 GB) verdict**: you can run the **whole gatekeeper stack + webapp in mock
  mode + the eval/analysis tooling** locally. You **cannot** reliably run Qwen3-VL-8B
  inference (needs ~16 GB bf16 / ~6-7 GB at 4-bit *with vision tokens* → OOM-prone on 6 GB)
  and you **cannot** fine-tune it locally (recipe targets a 24 GB 4090). The **1B Llama
  Guard is the one model you *can* fine-tune on the 4050**.
- **Chatbot (v2.0)**: multi-turn memory here is a ~thin layer on Qwen3-VL's native chat
  template + a server-side conversation store. **LangChain is not recommended** for this
  shape of problem (single local VLM, no multi-tool/RAG).

---

## 1. Current-state map & integration audit

| Subsystem                        | Where                                                                | Status                              | Notes                                                                                                    |
| -------------------------------- | -------------------------------------------------------------------- | ----------------------------------- | -------------------------------------------------------------------------------------------------------- |
| Frontend (React 19 + Vite)       | `frontend/`                                                        | ✅ Integrated                       | Single-shot;`FormData → POST /api/ask`; `useState` only; **no chat history**                  |
| Flask API                        | `backend/app.py`                                                   | ✅ Integrated                       | `GET /api/health`, `POST /api/ask`; latency wraps inference only                                     |
| Mock/real seam                   | `backend/inference.py`, `backend/model_adapter.py`               | ✅ Integrated                       | `USE_MOCK` flag; real path = `predict()` → `QwenVLChat`                                           |
| Real VLM (Qwen3-VL-8B + LoRA)    | `backend/qwen_vl_chat.py`                                          | ⚠️ Wired,**not deployable** | Loads base +`merge_and_unload`; **no quant**, **CPU-only image**, **no GPU service** |
| Layer-1 guard (cheap rules)      | `backend/app.py`                                                   | ✅ Integrated                       | Empty/weak-question + image-present checks → 400                                                        |
| Layer-2 guard (encoders)         | `backend/guard.py`                                                 | ✅ Integrated + tested              | detoxify toxicity, deberta prompt-injection, Presidio PII; fail-open                                     |
| Layer-3 guard (Llama Guard 3 1B) | `backend/guard_llm.py`, `guard/Dockerfile`                       | ✅ Integrated + tested              | Ollama, CPU, baked at build; custom`S99` off-topic code; fail-open                                     |
| Chart gate                       | `backend/chart_check.py`                                           | ✅ Integrated + tested              | CLIP zero-shot + OCR "has-data" gate + pixel-heuristic fallback                                          |
| Orchestrator                     | `app.py` (root)                                                    | ✅ Integrated                       | `--dev` (venv) vs prod (`docker compose up backend guard` + host Vite)                               |
| Containers                       | `docker-compose.yml`, `backend/Dockerfile`, `guard/Dockerfile` | ⚠️ Partial                        | backend + guard only;**frontend prod & GPU VLM not containerized**                                 |
| Training pipeline                | `modeling/chartqa/training/finetune_lora.py`                       | ✅ Complete                         | Custom loop, PEFT LoRA, bf16, grad-checkpoint, cosine, best-by-val-loss                                  |
| Eval + error analysis            | `modeling/chartqa/evaluation/`, `modeling/chartqa/analysis/`     | ✅ Complete                         | relaxed (5%) + exact; error dumps; disagreement sets; category table                                     |
| Committed adapters               | `modeling/checkpoints/{qwen3vl-lora-final2, blip2-lora-final}`     | ✅ Present (LFS)                    | Real runs (BLIP-2 trainer_state logs 700+ steps)                                                         |

**Integration verdict:** the mock-first contract held — the frontend never has to change.
The real model dropped cleanly behind `model_adapter.predict`. The only thing standing
between "demo in mock mode" and "real answers in prod" is a **GPU serving path** for the
8B VLM (see Phase 3).

---

## 2. Architecture review — strengths & improvement opportunities

### What's good (keep it)

- **Defense-in-depth, cheapest-first, fail-open.** Layer 1 (rules, ~0 ms) → Layer 2
  (encoders, ~10-50 ms) → Layer 3 (1B LLM, ~0.5-4 s). Every layer degrades to *allow* with
  a startup warning if a dependency is missing. This matches the project's
  production-efficiency principle exactly.
- **Expensive path is gated.** The VLM only runs after rules + guard + chart detection pass
  — the basis for the **scale-to-zero** design (CPU gatekeeper `min=1`, GPU VLM `min=0`).
- **Clean seams.** `USE_MOCK`, `model_adapter.predict`, env-driven config with no in-code
  defaults, warm-at-boot guard models in a background thread.
- **Real eval rigor.** Zero-shot vs fine-tuned, two metrics, per-sample error dumps, and a
  *disagreement* analysis — not just a single accuracy number.

### Improvement opportunities (ranked)

1. **[High] The VLM runs in-process in a CPU-only image.** `backend/Dockerfile` installs
   CPU torch; setting `USE_MOCK=0` there would try to load an 8B VLM on CPU → unusable.
   The scale-to-zero design *names* a GPU VLM service but it isn't built. → **Phase 3.1**.
2. **[Med] `merge_and_unload()` at load blocks quantized serving.** Folding LoRA requires a
   full-precision base in memory, so you can't serve a 4-bit base this way. For
   memory-bound or vLLM serving, keep the adapter attached (vLLM supports LoRA) **or**
   pre-merge once and save merged fp16 weights to disk. → **Phase 3.1**.
3. **[Med] Source duplication / drift.** `qwen_vl_chat.py` exists in **three** places
   (`backend/`, `model/`, `modeling/chartqa/models/`) and `chartqa_dataset.py` in **two**
   (`data/`, `modeling/`). The `backend/` copy is intentionally vendored for Docker
   isolation, but the repo-root `model/` and `data/` look like pre-`modeling/` leftovers.
   → **Phase 2.1**.
4. **[Med] Config↔checkpoint drift in modeling — blocks reproducing the team's Qwen result.**
   Verified on disk: the committed **Qwen** adapter (`qwen3vl-lora-final2/adapter_config.json`)
   is **`r=16, alpha=32`** (dropout 0.05, 7 target modules), but `constants.py` declares
   **`LORA_R=32, LORA_ALPHA=64`** and `max_steps=20` (a "quick-test value"), and the committed
   runs used far more than 20 steps (BLIP-2 `trainer_state` logs 700+). **Re-running
   `run_trainings.sh` as-is would NOT reproduce the 84.60%→86.08% Qwen numbers** — it would
   train a *different* config (`r=32`) for only 20 steps → a toy adapter. The open question
   the team must nail down: **which exact config (`r`, `alpha`, `max_steps`, lr, eff. batch)
   reproduces the committed Qwen checkpoint.** Start from the adapter's own
   `adapter_config.json` (`r=16/alpha=32`) as ground truth. → **Phase 2.2** (reproducibility).
5. **[Med] No answer cache / rate limit / upload re-encode.** Already on the authors'
   `NEXT_STEPS`. Cache by `(image_hash, question)` short-circuits repeats; re-encoding
   uploads via PIL strips malicious payloads. → **Phase 3.4**.
6. **[Med] No observability.** No per-stage latency, no metrics endpoint — hard to see
   which layer spends the time or how often the expensive VLM path fires (= cost).
   → **Phase 3.5** (Prometheus + Grafana).
7. **[Med] No experiment tracking.** Training/eval runs live in ad-hoc JSONs and
   hand-filled tables; the config↔checkpoint drift in #4 is exactly the failure class
   that tracking prevents. → **Phase 2.3** (MLflow).
8. **[Med] No persistence layer.** Nothing durable server-side: no cache store, no
   rate-limit counters that survive a restart, nowhere for conversations (v2.0),
   feedback events, or MLflow's backend store. → **Phase 3.6** (Postgres + Redis).

---

## 3. Pipeline state & results

### Training (`finetune_lora.py`)

Custom PyTorch loop (not HF `Trainer`): PEFT LoRA, **bf16** + autocast, gradient
checkpointing, `use_cache=False`, AdamW, cosine schedule w/ warmup, effective batch
**32** (bs 2 × grad-accum 16), best-checkpoint by validation cross-entropy. **No
quantization** (full-precision base). Built/tested on a **24 GB RTX 4090**.

> ⚠️ `constants.py` ships `max_steps=20` ("quick-test value"). The **committed** adapters
> were trained for real (BLIP-2 `trainer_state.json` → 700+ steps, eval_loss 3.53 → 2.56).
> Bump/parametrize this before any reproduction run (Phase 2.2).

### Evaluation (`evaluate.py` + `metrics.py`)

- **Relaxed match**: 5% numeric tolerance, %/thousand-separator aware, case-insensitive text.
- **Exact match**: strict string equality after whitespace strip.
- Optional `--errors-dir` dumps misclassified images + `errors.json`; `compare_errors`
  builds the zero-shot↔fine-tuned disagreement set; `category_table` does per-category
  accuracy.

### Results (ChartQA test, 2500 samples)

| Model                 | Metric  | Zero-shot     | LoRA fine-tuned         | Δ                 |
| --------------------- | ------- | ------------- | ----------------------- | ------------------ |
| **Qwen3-VL-8B** | Relaxed | 84.60% (2115) | **86.08%** (2152) | **+1.48 pp** |
| **Qwen3-VL-8B** | Exact   | 75.36% (1884) | **77.00%** (1925) | +1.64 pp           |
| BLIP-2 flan-t5-xl     | Relaxed | 8.40% (210)   | 12.40% (310)            | +4.00 pp           |
| BLIP-2 flan-t5-xl     | Exact   | 0.84% (21)    | 5.60% (140)             | +4.76 pp           |

**Reading:** Qwen3-VL is already strong zero-shot; LoRA adds a modest, real gain. BLIP-2
(flan-t5-xl) is architecturally weak at reading chart text/numbers — useful as a
"baseline 1 vs baseline 2" contrast, not a deployment candidate.

---

# PHASE PLAN

> **v1.0 = Phases 1-4.** **v2.0 = Phase 5.**

## Phase 1 — Test what we have, on the RTX 4050 (6 GB)

**Goal:** verify the whole local stack and the modeling tooling without needing a big GPU.

### 1.1 Feasibility on 6 GB VRAM

| Task                                                    | Runs on 4050?                        | How                                                                             |
| ------------------------------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------- |
| Webapp + 3-layer guard + chart gate,**mock mode** | ✅ Yes                               | CPU/RAM bound, not the 6 GB GPU                                                 |
| Llama Guard 3**1B** (Ollama, CPU)                 | ✅ Yes                               | Already CPU-only                                                                |
| Eval/error-analysis on the committed result JSONs       | ✅ Yes                               | No model load at all                                                            |
| BLIP-2 (3.9B) inference/eval                            | ✅ Yes (4-bit ~2.5 GB / 8-bit ~4 GB) | Needs a quant flag added to the wrapper                                         |
| **Qwen3-VL-8B inference**                         | ⚠️**Unreliable**             | bf16 ≈ 16 GB; 4-bit ≈ 6-7 GB*with vision tokens* → OOM-prone on 6 GB       |
| **Qwen3-VL-8B fine-tuning**                       | ❌**No**                       | Recipe targets 24 GB; QLoRA on a*vision* 8B is OOM-prone at 6 GB → use cloud |

### 1.2 Checklist

- [ ] Smoke-test the stack in mock mode: `python app.py --dev` → open `http://localhost:5173`,
  upload a `dataset/*.png`, confirm answer + chart-detection pill + latency.
- [ ] Exercise the guard: send a toxic / prompt-injection / PII / off-topic question and
  confirm the `blocked` path (run `cd backend && pytest` for the full guard/chart suite).
- [ ] Bring up the **real** Llama Guard container: `docker compose up guard` and re-test
  Layer-3 blocking against the live `:11434` endpoint.
- [ ] Validate the analysis pipeline with no GPU **and no VLM**: run
  `modeling/scripts/run_results_from_errors.sh` against the committed `outputs/errors/*`
  and diff against `outputs/results/*`.
  > **Why no GPU / no model:** this step is pure post-processing of results the VLM
  > *already* produced. `results_from_errors.py` loads **no model** — it reads the
  > committed `errors.json` (the list of failed indices from a past eval), rebuilds the
  > successes as every index not in that list, and pulls only the **question text** from
  > `ChartQADataset` to label each record. The only download is the ChartQA dataset
  > (text/metadata, CPU). Same for `category_table.py` / `compare_errors.py`. So "run the
  > analysis without a GPU" = re-derive the results/tables from saved eval dumps, **not**
  > re-run inference.
  >
- [ ] **(chosen — do this)** Sanity-check eval *logic* on a tiny slice with BLIP-2 4-bit:
  `python -m chartqa.evaluation.evaluate --model blip2 --metric relaxed --limit 20`
  (after adding the 4-bit flag in 1.3). This exercises the real eval path (dataset →
  model → metric → error dump) cheaply, on a model that fits the 4050.
- [ ] **(chosen — local smoke test before any cloud)** For a real Qwen3-VL answer, run
  **CPU inference** as a one-off smoke test (slow, minutes/answer). The point is *not*
  accuracy or latency — it's to confirm the **whole real path runs end-to-end locally
  before we spend on cloud**: run one of the weak/quantized configs (BLIP-2 4-bit, or
  Qwen at 4-bit / CPU) purely to see everything wires up and returns an answer, even if
  precision is low. Only after that green light do we push the accuracy runs to a cloud
  GPU (Colab/Kaggle T4 16 GB). Do **not** expect interactive latency on the 4050.

### 1.2b Known issues found during execution (2026-07-04)

- [X] **[Med · Windows-only] Analysis pipeline segfaults on the documented `python -m`
  invocation. FIXED 2026-07-07 (verified on the real invocations).**
  `modeling/chartqa/data/chartqa_dataset.py` imported `torch.utils.data` **before**
  `datasets`. On Windows with torch 2.6.0+cu124 loaded before pyarrow 24.0.0,
  `import pyarrow.dataset` crashes the process (access violation, **exit 139**). Both real
  entrypoints route through this module and it is the **first** in each chain to touch
  either library: the analysis entrypoint `results_from_errors.py:6` and the eval
  entrypoint `evaluate.py:19` both do `from chartqa.data.chartqa_dataset import ChartQADataset` — and in `evaluate.py` that line (19) runs **before** the torch-importing
  `chartqa.models.qwen_vl_chat` (line 20); `constants` and `evaluation.metrics` import
  neither torch nor datasets. So ordering the imports *inside* `chartqa_dataset.py` fixes
  both chains.
  **Fix applied:** swapped the two imports in `chartqa/data/chartqa_dataset.py` so
  `from datasets import load_dataset` runs **before** `from torch.utils.data import Dataset`,
  with a comment explaining the Windows torch-CUDA vs pyarrow DLL-load ordering (mirrors the
  lazy `datasets` import Phase 2.1 added to `chartqa/models/qwen_vl_chat.py`). No pyarrow
  pin needed. **Note — the Phase 2.1 change alone did NOT fix this**: it hardened a
  *different* module (`chartqa/models/qwen_vl_chat.py`) not on the analysis chain.
  **Measured (`backend/.venv`, dataset cached, no download):**
  - raw `import torch, datasets` → **139** (crash) vs `import datasets, torch` → **0** (mechanism confirmed).
  - `import chartqa.data.chartqa_dataset` → **139 before → 0 after**.
  - `import chartqa.evaluation.evaluate` (the quant-experiment entrypoint) → **139 before → 0 after**.
  - real `python -m chartqa.analysis.results_from_errors --errors-dir outputs/errors/errors_blip2_zero_shot_relaxed --out <scratchpad>/verify_1_2b.json`
    → **exit 0**; regenerated file is **byte-for-byte identical** to the committed
    `outputs/results/results_blip2_zero_shot_relaxed.json` (same sha256, same 342253 bytes)
    — the fix only unblocked the import, behavior unchanged.
  - backend suite still green: **48 passed, 2 skipped**.
    **Residual risk:** Windows-specific DLL-load ordering; the fix relies on
    `chartqa_dataset.py` staying the first module to import torch/datasets in each entrypoint
    chain. If a future edit adds a torch import earlier (e.g. to `constants.py` or
    `evaluation.metrics`, or reorders `evaluate.py` to import the models module before the
    dataset), a package-level `import datasets` guard in `chartqa/__init__.py` would be the
    more durable fix. Linux/WSL unaffected.
- [X] **[Low · resolved] Dataset name discrepancy.** Docs said `lmms-lab/ChartQA`; code
  uses `HuggingFaceM4/ChartQA` (`constants.py` `DATASET_NAME` — the mirror that produced
  the committed results). Clarifying notes added to `CLAUDE.md` and `docs/PLAN.md`;
  `project.md` (assignment brief) left as-is. Open question for Victor: keep the
  HuggingFaceM4 mirror, or switch the code to the assignment's `lmms-lab` source.

- **Env hygiene note:** the analysis validation installed `datasets==5.0.0` + ~16
  transitive packages into `backend/.venv` (not in `backend/requirements.txt`). Formalize
  or clean this when the modeling env is set up (Phase 2.1 packaging).

### 1.3 Small enabler (needed for local BLIP-2 / quantized runs)

- [ ] Add an opt-in `load_in_4bit` / `BitsAndBytesConfig` path to the model wrappers
  (`modeling/chartqa/models/*.py`, and `backend/qwen_vl_chat.py`) behind an env flag —
  the current loaders are full-precision only. (Shared with Phase 1.5.)

---

> ✅ **Reviewed end-to-end (Victor, 2026-07-04).** Decisions locked in this pass:
> MLflow experiment tracking (Phase 2.3) · production-minded de-dup (2.1) · Prometheus +
> Grafana observability (3.5) · Postgres + Redis data layer (3.6) · security, auth & cost
> control for the public portfolio deploy (3.7) · stock-vs-fine-tuned Llama Guard
> precision/recall comparison (Phase 4) · LangChain section dropped from Phase 5 ·
> master execution checklist (stack / deploy / fine-tuning workflows) at the end.

## Phase 1.5 — Quantization & cost study: Qwen3-VL-8B → production (v1.0)

**Goal:** find the **cheapest precision that runs the *same* model** (Qwen3-VL-8B +
`qwen3vl-lora-final2`) in production within budget — ideally on the **6 GB RTX 4050**,
otherwise on a **cheap cloud GPU at `min=0`**. The method is **incremental, lowest-cost-
first**: prove the harness on free/tiny runs *before* spending on full runs, so GPU money
buys results, not setup-debugging time.

> No 4090 access anymore → the Qwen-8B configs that don't fit 6 GB run on **free/cheap
> cloud**. The design below keeps the whole *accuracy* study inside **free-tier T4 16 GB**.

### Guiding principles

- **Accuracy is hardware-independent** → measure each config on the *cheapest GPU that
  holds it*. A free **T4 (16 GB)** holds both 4-bit (~6 GB) and 8-bit (~10 GB) of the 8B.
- **bf16 accuracy is already measured** (committed results: 84.60% / 86.08% relaxed) — do
  **not** pay to re-run it. It is the reference row.
- **Latency + peak VRAM are hardware-dependent** → measure on the **target** hardware
  (the 4050 for configs that fit; otherwise report the cloud GPU and mark "does not fit 6 GB").
- **Always validate with `--limit` first** (seconds of GPU) before a full 2500-sample run
  (tens of minutes).
- **Persist the HF cache + dataset** across sessions (Kaggle Dataset / Colab Drive / RunPod
  volume) so you never re-download ~16 GB.

### Cost ladder (climb only when forced)

| Tier        | GPU                                                                   | Cost           | Use for                                                             |
| ----------- | --------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------- |
| **0** | Kaggle T4 16 GB (~30h/wk) or Colab free T4                            | **Free** | The**entire accuracy study** (4-bit + 8-bit, full 2500)       |
| **1** | L4 24 GB / A5000 24 GB / community RTX 4090 (RunPod/Vast, per-minute) | Cheap, spot    | bf16 latency reference, or if T4 OOMs                               |
| **2** | A100 / H100                                                           | Expensive      | **Avoid early** — only on a *proven* OOM / throughput wall |
| local       | **RTX 4050 6 GB**                                               | Owned          | VRAM + latency of configs that**fit** (4-bit candidate)       |

### Stage A — Validate the pipeline at minimal cost (local → free cloud, tiny sample)

Prove the *mechanics* (model load, LoRA/merged-weights, processor, metric computation,
error dumps) where it's cheapest — debug setup on a handful of samples, not on a paid run.

- [ ] **A0 — Local dry-run, no big model.** Run the eval harness end-to-end with a tiny
  slice to exercise dataset load + metrics + error dumps:
  `python -m chartqa.evaluation.evaluate --model blip2 --metric relaxed --limit 8`
  (BLIP-2 fits locally; the goal is to shake out the *harness*, not accuracy).
- [ ] **A1 — Decide the LoRA-on-quant strategy** (merge does **not** combine with a
  quantized base):
  - **Path 1 (cheapest, default for the study):** load base with `load_in_4bit`, then attach
    the adapter with `PeftModel.from_pretrained(base_4bit, adapter)` — **no merge**, no 16 GB
    GPU needed.
  - **Path 2 (production packaging):** merge LoRA into bf16 **once** (CPU-merge is free if you
    have ~16 GB RAM; or a Tier-1 GPU), save merged fp16 weights, then quantize-load those.
- [ ] **A2 — Free-cloud smoke test on the *real* model.** On Kaggle/Colab T4, run **4-bit**
  on ~50-100 samples to confirm the full real path (real Qwen + real 4-bit + real metrics)
  works before the full run: `--quantization 4bit --limit 100`.

### Stage B — 4-bit first (the production candidate)

4-bit NF4 is the **only** config with a chance of fitting the 4050, so it goes first.

- [ ] **B1 — Accuracy (free cloud).** Full **2500-sample** eval at **4-bit NF4** on a T4.
  Because accuracy is hardware-independent, this number **equals** what the 4050 would
  produce. Record relaxed + exact; diff vs the bf16 reference.
- [X] **B2 — VRAM + latency (local 4050). DONE (2026-07-08): does NOT run usably on 6 GB.**
  Qwen3-VL-8B at 4-bit NF4 does **not fit** the 6 GB 4050: `device_map="auto"` refuses
  (bnb 4-bit blocks CPU/disk offload) unless `llm_int8_enable_fp32_cpu_offload=True`. With
  offload enabled the weights *load* (7 min) but dispatch spills layers to **disk**, which
  hits a **bitsandbytes-4bit + accelerate meta-tensor bug** (`Tensor.item() cannot be called on meta tensors`) at `attach_execution_device_hook`. `PYTORCH_CUDA_ALLOC_CONF= expandable_segments` is unsupported on Windows. The `QWEN_CPU_OFFLOAD`/`--cpu-offload`
  experiment was implemented, hit this wall, and was **reverted** (kept the plain
  `--quantization` flag). BLIP-2 4-bit *does* run locally (proves the harness end-to-end).
- [X] **B-gate (decision): 8B is CLOUD-ONLY `min=0`.** Per the rule below (OOM on 4050),
  Qwen3-VL-8B does not get a local-deployable config; its accuracy study (B1) and the VLM
  service (Phase 3.1) run on cloud GPUs. The `max_memory` (GPU+CPU, no disk) lever is
  untried and could dodge the meta-tensor bug, but even a successful load would likely OOM
  on the vision prefill — not worth the fight for a 6 GB card.
  - **Fits 6 GB with margin (<~5.5 GB peak) AND Δrelaxed ≤ ~1-2 pp** → ✅ local-deployable
    config found. *(Not achieved — see B2.)*
  - **OOM on 4050** → 8B stays **cloud-only `min=0`** (Phase 3.1); record the verdict and the
    cloud latency/VRAM instead. *(This is the outcome.)*

### Stage C — Larger configs, only after the harness is proven

- [ ] **C1 — 8-bit, full 2500 (free cloud).** ~10 GB fits a T4. Record accuracy + latency/VRAM
  on that GPU (won't fit 6 GB → not a 4050 candidate, but completes the comparison).
- [ ] **C2 — bf16 reference.** Accuracy already done. *Optionally* measure bf16 latency/VRAM
  once on a Tier-1 GPU (L4/A5000) for the latency column.
- [ ] **C3 — (extension) AWQ or GPTQ 4-bit** *only if* NF4 accuracy drop is unacceptable —
  calibration-based, better retention, more work for a VLM (vision tower handling).

### Stage D — Cost-control rules (apply throughout)

- [ ] Free tier (Tier 0) by default; the whole accuracy study fits there.
- [ ] On-demand Tier-1 GPUs **per-minute, shut down when idle**; never leave a box running.
- [ ] `--limit` validation **before** every full run.
- [ ] Persist HF cache + dataset; pin `bitsandbytes` / `transformers` / `accelerate` versions
  for reproducible quant behavior.
- [ ] No A100/H100 unless a measured OOM/throughput wall forces it.

### Harness deliverable (extends `evaluate.py`)

- [ ] Add `--quantization {none,8bit,4bit}` (builds the matching `BitsAndBytesConfig`).
- [ ] Record per config, alongside accuracy: **p50/p95 latency** (with
  `torch.cuda.synchronize()`, discard a warmup sample), **peak VRAM**
  (`torch.cuda.reset_peak_memory_stats()` → `torch.cuda.max_memory_allocated()`),
  **load time**, and **on-disk size**.
- [ ] Write one results JSON per `(model, quant, metric)` and a roll-up comparison table.
- [ ] Log every run to **MLflow** (Phase 2.3) — params, metrics, hardware tag — so the
  comparison table below becomes a query, not a hand-filled artifact.

### Comparison table (fill as runs complete)

| Config     | Where run        | Relaxed      | Exact        | Δrelaxed vs bf16 | p50 / p95 latency | Peak VRAM    | Fits 6 GB?                                                    |
| ---------- | ---------------- | ------------ | ------------ | ----------------- | ----------------- | ------------ | ------------------------------------------------------------- |
| bf16 (ref) | committed        | 86.08%       | 77.00%       | —                | —                | ~16 GB       | ❌                                                            |
| 8-bit      | T4 (free)        | _tbd_      | _tbd_      | _tbd_           | _tbd_           | _tbd_      | ❌                                                            |
| 4-bit NF4  | 4050 (attempted) | _tbd_ (T4) | _tbd_ (T4) | _tbd_           | —                | did not load | ❌**no** — offload→disk hits bnb meta-tensor bug (B2) |

> **Outcome feeds Phase 3.1:** the winning precision + whether it's *local-4050* or
> *cloud-`min=0`* determines how the VLM service is built and sized.

---

## Phase 2 — Architecture cleanup & reproducibility (v1.0)

### 2.1 De-duplicate sources (production architecture)

**Principle:** in a production system, every deployable artifact consumes shared code in
exactly one of two ways — as a **versioned dependency** (installable package) or as a
**vendored copy verified by CI**. Unverified copies are drift bombs: with three
`qwen_vl_chat.py` files, the one you *deploy* can silently stop being the one you
*trained and evaluated* with.

Target layout — one owner per artifact:

| Artifact                           | Owner (source of truth)                 | How others consume it                                         |
| ---------------------------------- | --------------------------------------- | ------------------------------------------------------------- |
| Training/eval/analysis code        | `modeling/chartqa/` (installable pkg) | `pip install -e ./modeling` in dev/CI                       |
| Serving wrapper`qwen_vl_chat.py` | `modeling/chartqa/models/`            | `backend/` keeps a **vendored copy + CI drift check** |
| Dataset code`chartqa_dataset.py` | `modeling/chartqa/data/`              | nobody else — the root copy is deleted                       |

- [ ] Make `modeling/` an installable package (`pyproject.toml`, `pip install -e`) so
  training/eval imports stop depending on path tricks — the standard shape for a
  reusable ML package.
- [ ] **Delete** the repo-root `model/qwen_vl_chat.py` and `data/chartqa_dataset.py`
  (pre-`modeling/` leftovers). Grep for imports first; point any stragglers at
  `modeling/chartqa`.
- [ ] Keep `backend/qwen_vl_chat.py` **vendored** — the backend image must stay
  independently buildable without the modeling tree and its training-only deps — but
  make the copy *verified*: a pytest (run in CI, Phase "deploy workflow") that fails
  when the vendored file diverges from `modeling/chartqa/models/qwen_vl_chat.py`
  (hash compare), with a header comment in both files naming the counterpart.
- [ ] Document the rule in `CLAUDE.md`: shared code lives in `modeling/chartqa`;
  vendoring into `backend/` requires the drift check.

### 2.2 Fix modeling reproducibility

> **Primary goal (per Victor): find the config that reproduces the *same* Qwen result the
> team got (84.60% → 86.08% relaxed).** The committed adapter is the source of truth —
> reconcile the code to *it*, not the other way around.

- [ ] Reconcile `constants.py` LoRA values with the committed **Qwen** adapter — pin
  `LORA_R=16, LORA_ALPHA=32` (from `qwen3vl-lora-final2/adapter_config.json`), not the
  declared `r=32/α=64`. Confirm the 7 `target_modules` match too. Document any BLIP-2 vs
  Qwen difference if the two adapters used different `r`.
- [ ] Recover and pin the **real `max_steps`** (and lr, warmup, eff. batch) the team used —
  the current `max_steps=20` is a quick-test toy. Cross-check against `trainer_state.json`
  in each committed checkpoint. Make it a CLI arg with a loud, correct default so
  `run_trainings.sh` reproduces the committed checkpoints instead of a 20-step toy.
- [ ] **Verify by re-eval, not by re-train first:** run `evaluate.py` against the committed
  `qwen3vl-lora-final2` adapter and confirm you get the committed 86.08% relaxed — this
  proves the eval harness + adapter are the ones that produced the reported number, before
  spending GPU on a reproduction training run.
  > **Deferred to cloud / Phase 1.5 (2026-07-07).** This GPU re-eval needs Qwen3-VL-8B
  > inference, which does not fit the local 6 GB RTX 4050 (OOM-prone even at 4-bit with
  > vision tokens — see §1.1). It rides along with the Phase 1.5 quantization study on a
  > free-tier T4. The rest of 2.2 (constants reconciled to the committed adapters,
  > per-model r/α, 7 LM-only qwen modules, `max_steps` parametrized, MODELCARDs) is done
  > and needs no GPU.
  >
- [X] Commit a short `MODELCARD`/run-log per adapter (steps, lr, eff. batch, final eval_loss,
  hardware) next to each checkpoint. **Done:** `checkpoints/qwen3vl-lora-final2/MODELCARD.md`
  and `checkpoints/blip2-lora-final/MODELCARD.md` — verifiable facts only; BLIP-2 records
  the recovered 884-step schedule + losses, Qwen records that its schedule is not
  recoverable from committed artifacts.

### 2.3 Experiment tracking — MLflow (MLOps foundation)

**Why MLflow (vs W&B / nothing):** self-hosted, free, no per-seat pricing — matches the
project's local-first / no-paid-API principle, and owning the infra is the better
portfolio signal. It also directly fixes the failure class in 2.2: with tracked runs,
"which config produced this checkpoint?" is a lookup, not archaeology.

> **Status (local tracking landed, 2026-07-07).** The **local-runnable** tracking layer is
> done and tested: `modeling/chartqa/tracking.py` (fail-open helper — no-op with a warning
> when `mlflow` is absent or `MLFLOW_ENABLED=0`; local `./mlruns` file store, env-driven URI/
> experiment), `evaluate.py` fully instrumented (params + accuracy + guarded load-time /
> p50-p95 latency / peak-VRAM / adapter-size, hardware tag), `mlflow>=2.14,<3` added to
> `modeling/requirements.txt`, and `modeling/tests/test_tracking.py` (3 tests: real temp
> file store + fail-open-when-absent + disabled-via-env; all green). **View runs:**
> `cd modeling && mlflow ui`. The items still open below are production/registry pieces.

- [ ] Add an `mlflow` service to `docker-compose.yml` (official image): backend store =
  **Postgres** (shared instance from Phase 3.6, dedicated database), artifact store =
  named volume (MinIO/S3 only if artifacts outgrow the disk). UI **not exposed
  publicly** (see 3.7 — SSH tunnel/Tailscale only). **DEFERRED to Phase 3.6** — local
  file store is what v1-local uses.
- [ ] **Instrument `finetune_lora.py`** (`log_params` LoRA r/α/dropout/target_modules, lr,
  warmup, eff. batch, max_steps, scheduler, seed; per-eval-step `log_metrics` train/val
  loss; `log_artifacts` adapter + `trainer_state.json` + MODELCARD). **DEFERRED** — wraps
  ~100 lines of the custom training loop and can't be runtime-verified without a
  multi-hour GPU training run, so it wasn't shipped untested; the `tracking.py` helper is
  ready to drop in. (Training isn't run at the local checkpoint; eval tracking is the path
  the quant study exercises.)
- [X] **Instrument `evaluate.py`**: one run per `(model, adapter, quantization, metric)` —
  relaxed/exact accuracy, p50/p95 latency, peak VRAM, load time, on-disk size, hardware
  tag (cpu / 4050 / T4). Guarded for CPU (VRAM/latency skip cleanly). **The Phase 1.5
  comparison table then fills itself.** DONE + tested.
- [ ] Use the **Model Registry** for adapters: register `qwen3vl-lora` and (Phase 4)
  `guard-lora` versions; a version only gets the `production` alias after passing the
  eval gate (see the master checklist). Deploys reference a registry version, not a
  loose directory.
- [ ] Cloud runs (Kaggle/Colab): do **not** expose the tracking server to the internet —
  log to a local `mlruns/` in the session, download, and import into the tracked store
  (`mlflow-export-import`); or re-log the final metrics JSON. Cheap and safe.
- [ ] Backfill the two committed adapters (`qwen3vl-lora-final2`, `blip2-lora-final`)
  as registered versions with their known metrics, so the registry reflects reality
  from day one.

---

## Phase 3 — Deploy hardening & containerizing what's missing (v1.0)

### 3.1 Build the GPU VLM inference service (the missing piece)

> **Status (2026-07-08): the serving seam is built + tested; only the GPU container/infra
> remains.** `vlm_service/` is a real Flask service wrapping the (single-source-of-truth)
> `QwenVLChat`, warm-loaded at boot, exposing `POST /predict {image, question} → {answer}`
> and `GET /health`. `backend/model_adapter.predict` now routes to it when `VLM_URL` is set
> (else in-process), tested end-to-end against a stub server (`test_model_adapter_remote.py`,
> real HTTP round-trip; no GPU needed). This also enables the "local app + remote cloud GPU"
> demo path (README shows the Kaggle/Colab + tunnel recipe).

- [X] Stand up Qwen3-VL-8B as a **separate HTTP service** (not in-process), so the backend
  calls it like it calls the guard. Built as a small **Flask wrapper** (`vlm_service/`)
  reusing the tested `QwenVLChat` — the **vLLM** OpenAI-compatible route (serves LoRA
  directly, higher throughput) stays a documented swap-in for later if throughput demands.
- [X] Pre-merge alternative noted; the wrapper keeps the adapter attached (works with a
  quantized base). **GPU service, `min=0`**, behind the CPU gatekeeper — by design.
- [X] Add a `VLM_URL` env (mirroring `GUARD_LLM_URL`) + `VLM_TIMEOUT` (generous for cold
  start) and switch `model_adapter.predict` to an HTTP call when set; keep the in-process
  path for dev on a big GPU. **DONE + tested.**
- [ ] **Remaining:** a **GPU Dockerfile** for `vlm_service/` + a `docker-compose` service
  with `deploy.resources.reservations.devices: [{capabilities: [gpu]}]` and `min=0`
  autoscaling. Needs a CUDA host to build/run — deferred to the deploy environment.

### 3.2 Containerize the frontend (prod)

- [X] **DONE** — `frontend/Dockerfile` (multi-stage: `node:20-alpine` builds the Vite app →
  `nginx:1.27-alpine` serves the static `dist/`), `frontend/nginx.conf` (static + SPA
  fallback + `/api` proxy to `backend:5000` + `client_max_body_size 10m` + security
  headers), `frontend/.dockerignore`, and a `frontend` service in `docker-compose.yml`
  (`8080:80`, `depends_on: backend`). Same-origin in prod (no CORS). `docker compose config`
  validates; run `docker build ./frontend` on a Docker host to produce the image.

### 3.3 Slim the backend image

- [~] **Partly done.** `backend/.dockerignore` already excludes the big/dev items —
  critically `.venv/` (can be several GB) so `COPY . .` doesn't bake it — plus
  `__pycache__`, `tests/`, `*.md`, and now `.env*` (secrets never in the image). *Remaining
  (optional):* a true multi-stage builder→runtime split; the baked CLIP + Layer-2 encoder
  weights (~1 GB+) dominate size, so the gain is modest — do it with a `docker build` on a
  host to measure before/after.

### 3.4 Request hardening

- [X] **Answer cache** keyed by `(sha256(image_bytes), normalized_question)` — short-circuits
  repeats before the guard/VLM. **DONE** (`backend/answer_cache.py`): in-memory LRU + TTL,
  fail-open (any error → miss, never breaks a request), env-config (`ANSWER_CACHE_*`), real
  answers only (never the mock disclaimer), wired into `/api/ask` before the guard, cached
  value = `{answer, is_chart, chart_confidence}`. Unit-tested. **Redis backend = the seam's
  prod swap (3.6).**
- [ ] **Rate limiting** (per-IP token bucket) via Flask-Limiter with **Redis storage**
  (3.6) so limits survive restarts and hold across gunicorn workers.
- [X] **Upload safety** — **DONE** (`backend/uploads.py` `sanitize_image`): decode with PIL
  and re-encode from pixels to strip embedded/trailing payloads (polyglots), reject
  non-images with a 400. Wired at the top of `/api/ask` (also yields the canonical bytes
  the cache keys on). `MAX_UPLOAD_MB` was already enforced via `MAX_CONTENT_LENGTH`.
  Unit-tested (valid re-encode, trailing-payload strip, JPEG→PNG, non-image reject).
- [ ] **Evasion hardening**: NFKC normalize / de-homoglyph / de-leet, and decode-and-rescreen
  base64/hex before the guard (already designed in `ROBUSTNESS.md`).

### 3.5 Observability — Prometheus + Grafana

**Verdict: yes, this is the right stack here.** Self-hosted, free, the de-facto market
standard for container metrics, and only two extra CPU-light containers — strong
portfolio signal without operational weight. (Skip distributed tracing — Jaeger/Tempo is
overkill for one API service; per-stage latency histograms answer the same question.)

- [ ] Expose `/metrics` from Flask via `prometheus_client`:
  - `http_requests_total{route, status}` (counter)
  - `stage_latency_seconds{stage=layer1|guard_l2|guard_l3|chart_gate|vlm}` (histogram —
    definitively answers "which layer spends the time")
  - `blocked_total{reason=toxicity|injection|pii|off_topic|not_chart|weak_question}`
  - `cache_hits_total` / `cache_misses_total`
  - `vlm_invocations_total`, `vlm_tokens_generated_total` — **the cost proxies** that
    drive the budget alerts in 3.7
  - guard **fail-open events** (a dependency silently missing in prod is a silent
    security downgrade — it must page, not hide)
- [x] **DONE** — `prometheus` + `grafana` compose services scrape the backend; the datasource
  and a dashboard (request rate, per-stage p95 latency, VLM invocations = cost, cache hit
  ratio) are **provisioned** as JSON/YAML under `observability/`, checked into the repo.
  `docker compose config` + the YAML/JSON validate; run `docker compose up prometheus grafana`
  on a Docker host to view (Grafana `localhost:3000`, Prometheus `localhost:9090`). Screenshot
  for the README. *(Runtime render not smoke-tested — Docker engine was down at write time.)*
- [ ] Grafana alert rules: VLM invocations/hour over budget, GPU-active minutes/day over
  budget, p95 end-to-end latency, 5xx rate, any guard fail-open event.
- [~] Per-stage `*_ms` — `latency_ms` (inference) is in the API response; structured JSON
  logs still TODO.
- [x] `/metrics`, Grafana, Prometheus bound to **localhost** in compose (off the public
  ingress; reach via SSH tunnel / private net — see 3.7). MLflow UI same when added (3.6).

### 3.6 Data layer — Postgres + Redis

**Current state: no database at all** — the app is fully stateless (fine for v1.0
single-shot, and part of why it's cheap). Three phases now need state, and the
market-standard split is:

| Store                          | Used for                                                                                        | Why this one                                                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Redis** (7-alpine)     | Answer cache w/ TTL (3.4) · rate-limit counters (3.4/3.7) · v2.0 conversation hot-store (5.1) | In-memory speed, native TTL/eviction, atomic counters                                                                                                  |
| **Postgres** (16-alpine) | MLflow backend store (2.3) · v2.0 conversation persistence · feedback events (5.1)            | Durable + relational, the production default — SQLite would*work* at this scale, but Postgres is what production uses and a container makes it free |

- [ ] Add `redis` + `postgres` compose services with named volumes and healthchecks;
  wire `depends_on: condition: service_healthy` for consumers.
- [ ] Config via env (`REDIS_URL`, `DATABASE_URL`), consistent with the existing
  no-in-code-defaults convention; both **fail-open**: no Redis → in-memory cache +
  per-process rate limit; no Postgres → the features that need it disable cleanly.
- [ ] Don't create app tables before something needs them — Postgres arrives with MLflow
  (2.3) and earns its keep immediately; the app schema (conversations, messages,
  feedback) lands in Phase 5 with **Alembic** migrations.

### 3.7 Security, auth & cost control (public portfolio deploy)

**Auth verdict: no user accounts.** For a portfolio demo a login wall kills the demo.
What is actually needed: (a) **abuse/cost control** on the public path, (b) **real auth
on admin surfaces** (Grafana, MLflow, Prometheus, `/metrics`). Revisit only if v2.0
should persist conversations per visitor — and then a signed anonymous session cookie
suffices, still no login.

**Cost-runaway risk: yes, real — and it's the GPU.** The CPU stack (VPS + guard) is
fixed-cost; the scenarios that actually hurt are: (1) a bot/script hammering `/api/ask`
and keeping the GPU service warm 24/7, (2) the scale-to-zero service never reaching zero
because a health check or uptime pinger hits it, (3) a forgotten per-hour dev box.
Defenses, cheapest first:

- [ ] **Provider-level hard cap — the real safety net.** Prefer a prepaid/credit GPU
  provider (RunPod credits, Modal spend limit): worst case is a paused demo, never a
  surprise bill. Set billing alerts regardless.
- [ ] **Cloudflare free tier in front** of the public hostname: TLS, DDoS/bot filtering,
  static-asset caching. (Alternative: Caddy + Let's Encrypt on the VPS if fewer parties
  is preferred.)
- [ ] **Rate limits** (3.4, Redis-backed): per-IP limit on `/api/ask` **plus a global
  concurrency cap** on VLM calls (semaphore) so a burst queues instead of fanning out to
  the GPU.
- [ ] **Daily VLM budget + circuit breaker**: count `vlm_invocations_total` per day;
  past budget, short-circuit with HTTP 429 "demo quota reached — try tomorrow" while
  cache/guard/mock paths keep working. The demo degrades; the bill doesn't.
- [ ] **Scale-to-zero hygiene**: idle timeout ≤ 2-5 min on the GPU service; ensure
  *nothing* (uptime monitor, compose healthcheck, warm-up cron) probes the GPU service
  directly — probe the CPU gatekeeper instead; Grafana alert (3.5) on GPU-active
  minutes/day.
- [ ] **Admin surfaces off the public ingress**: Grafana/MLflow/Prometheus/`/metrics`
  bound to localhost or a private network, reached via SSH tunnel or Tailscale; Grafana
  gets a real admin password.

- [~] **App hardening** — *partly done*: **CORS pinned** via `CORS_ORIGINS` env (default
  `*` for dev; set to the frontend origin(s) in prod) ✅; **security headers** on every
  response (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`) ✅; `MAX_CONTENT_LENGTH` enforced ✅; **upload re-encode** (3.4) ✅; VLM
  HTTP call has an explicit generous timeout (`VLM_TIMEOUT`) ✅. *Remaining:* containers
  run non-root; base images pinned by digest; verify guard timeout values.

- [ ] **Supply chain & secrets**: `pip-audit` (or Dependabot) in CI; secrets only via
  env / provider secret store — never in images or the repo; `.env` stays gitignored.
- [ ] **Log hygiene**: don't log raw questions at INFO (Presidio already flags PII — log
  a hash + guard verdict instead); never log image bytes.

---

## Phase 4 — Fine-tune Llama Guard (v1.0)

**Why:** today you run stock `llama-guard3:1b` with a *prompt-injected* custom `S99`
off-topic taxonomy. Fine-tuning lets you (a) harden the chart-domain on-topic/off-topic
decision, (b) reduce false-positives that block legitimate chart questions, (c) bake the
custom taxonomy into weights instead of the prompt. **Bonus: at 1B this is the only model
you can train on the RTX 4050** (bf16 ≈ 2 GB; LoRA/QLoRA trivially fits 6 GB).

### Checklist

- [ ] **Build a labeled set** of `question → {safe | unsafe + category}` covering the
  taxonomy you use (S1-S14 + custom S99 off-topic), with a strong block of *legit chart
  questions* (negatives) so you don't teach it to over-block.
- [ ] Seed data from: the guard's own outputs on real/synthetic questions, jailbreak/PII
  corpora, and chart-question paraphrases (on-topic) vs random non-chart asks (off-topic).
- [ ] **LoRA fine-tune `meta-llama/Llama-Guard-3-1B`** reusing the existing PEFT harness
  pattern from `modeling/` (same `target_modules` family as Qwen LM layers).
- [ ] **Evaluate zero-shot first (the baseline row):** run the **stock**
  `llama-guard3:1b` (with the current prompt-injected S99 taxonomy) on the held-out
  split; record per-category **precision / recall / F1**, the confusion matrix, and the
  **false-positive rate on legit chart questions** (the primary guardrail metric).
- [ ] **Evaluate the fine-tuned guard on the same split** and publish the side-by-side —
  the guard's version of the Qwen zero-shot-vs-LoRA table (both runs logged to MLflow,
  Phase 2.3):

  | Guard config                     | Precision | Recall  | F1      | FPR on legit chart Qs |
  | -------------------------------- | --------- | ------- | ------- | --------------------- |
  | Stock (zero-shot + S99 prompt)   | _tbd_   | _tbd_ | _tbd_ | _tbd_               |
  | LoRA fine-tuned (taxonomy baked) | _tbd_   | _tbd_ | _tbd_ | _tbd_               |

  **Ship gate:** fine-tuned recall ≥ stock **and** FPR strictly lower — otherwise keep
  stock in prod (the fine-tune is portfolio-valuable either way; shipping it is not
  automatic).
- [ ] **Package**: convert to GGUF + quantize (llama.cpp), build a custom Ollama `Modelfile`,
  and swap the baked tag in `guard/Dockerfile`. The OpenAI-compatible endpoint stays the
  same → `backend/guard_llm.py` is unchanged.
- [ ] A/B the fine-tuned guard vs stock on the eval set before flipping `GUARD_LLM_MODEL`.

---

## Phase 5 — Conversational v2.0: multi-turn chat with memory

**Goal:** instead of "one image + one question → one answer", let the user keep asking
follow-ups about the *same* chart, with history.

> **Framework decision (settled — section removed):** no LangChain/LangGraph. Memory =
> appending turns to Qwen3-VL's native chat template with a sliding window + a
> server-side store (~30 lines). Revisit only if real retrieval/tools (multi-chart
> search, SQL over extracted data, RAG) ever arrive.

### 5.1 Backend checklist

- [ ] Add a conversation store: in-memory `dict[conversation_id → {image_bytes, messages}]`
  for `--dev`; **Redis** (3.6) for prod/multi-worker, with a **TTL/eviction policy**
  (e.g. 30 min idle) so abandoned sessions don't accumulate.
- [ ] New/extended endpoint: `POST /api/ask` accepts an optional `conversation_id`; the
  image is uploaded **once** on turn 1 and cached server-side by `conversation_id`.
- [ ] Extend `QwenVLChat.build_messages` to accept prior turns (it already supports
  system/user/assistant roles) and feed the image only on the first user turn.
- [ ] **Sliding window** on history (keep last N turns; optionally summarize older turns) to
  stay within the token budget — charts rarely need more than the last few turns.
- [ ] **Run the guard on every user turn**, not just the first.
- [ ] Reuse the **answer cache** (Phase 3.4) keyed by `(image_hash, conversation_state, question)`.
- [ ] **Streaming (SSE)** for answers — market-standard chat UX (was "stretch"; it isn't
  anymore). Flask serves it with a generator response; keep a buffered fallback for
  non-streaming clients.
- [ ] **Persist transcripts to Postgres** (3.6): `conversations` / `messages` tables via
  Alembic. Redis stays the hot store; Postgres is the durable copy.
- [ ] **Feedback endpoint** (`POST /api/feedback`: 👍/👎 + optional note per answer) →
  Postgres. This is the market-standard **data flywheel**: reviewed feedback becomes
  the next eval/fine-tuning set (see the master checklist).

### 5.2 Frontend checklist

- [ ] Replace the single `answer` state with a **messages array**; render a chat transcript.
- [ ] Pin the selected image for the session; show it as the conversation header.
- [ ] Send `conversation_id` on follow-ups; only attach the image on turn 1.
- [ ] Add "new chart / reset conversation" to clear the session.
- [ ] Render streamed tokens as they arrive (SSE) + a **stop-generation** control
  (wire the existing AbortController to the stream).
- [ ] **👍/👎 on each assistant message**, wired to `POST /api/feedback`.
- [ ] Render answers as **markdown** (longer chart answers include lists/tables).
- [ ] Keep the existing abort-on-resubmit and accessibility (`aria-live`) behavior.

### 5.3 Stretch

- [ ] Shareable read-only conversation links (served from the Postgres transcript).
- [ ] Optional: extract the chart's underlying data table once, cache it, and let follow-ups
  query *that* (cheaper + more accurate than re-reading pixels every turn).

---

# MASTER EXECUTION CHECKLIST — stack, deploy workflow, fine-tuning workflow

> The phases above are the *what*; this section is the cross-cutting *how*: the exact
> stack, the CI/CD deploy workflow, the fine-tuning (MLOps) loop, and the definition of
> done for the v1.0 portfolio launch.

## Stack (final)

| Layer                          | Choice                                                                                               | Phase                         |
| ------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------- |
| Frontend                       | React 19 + Vite → static build served by an**nginx** container                                | 3.2                           |
| API                            | Flask behind**gunicorn** (multi-worker) in the backend container                               | 3.3                           |
| Guard L1                       | in-process rules (~0 ms)                                                                             | done                          |
| Guard L2                       | detoxify + DeBERTa injection + Presidio (in-process, warm at boot)                                   | done                          |
| Guard L3                       | Llama Guard 3**1B** on Ollama (CPU container, model baked at build)                            | done → fine-tuned in Phase 4 |
| Chart gate                     | CLIP zero-shot + OCR + pixel fallback (in-process)                                                   | done                          |
| VLM serving                    | Qwen3-VL-8B + LoRA on**vLLM**, separate GPU service, **scale-to-zero (`min=0`)**       | 3.1                           |
| Cache / rate-limit / hot-store | **Redis 7**                                                                                    | 3.4/3.6                       |
| Durable DB                     | **Postgres 16** (MLflow store · v2.0 transcripts · feedback)                                 | 2.3/3.6                       |
| Experiment tracking            | **MLflow** server + Model Registry (Postgres-backed)                                           | 2.3                           |
| Metrics / dashboards / alerts  | **Prometheus + Grafana** (dashboard JSON provisioned in-repo)                                  | 3.5                           |
| Ingress / TLS / bot filter     | **Cloudflare free tier** (or Caddy + Let's Encrypt)                                            | 3.7                           |
| CI/CD                          | **GitHub Actions** → images to **GHCR** → `docker compose pull` on the host          | below                         |
| Hosts                          | CPU stack on a small**VPS** (Hetzner-class); GPU on **RunPod/Modal serverless, prepaid** | 3.1/3.7                       |

## Deploy workflow (CI/CD)

- [ ] **CI on every PR**: ruff + backend `pytest` (including the 2.1 vendored-file drift
  check) + `npm run build` — branch protection already forces the PR flow.
- [ ] **Build on merge to `main`**: GitHub Actions builds `backend`, `guard`, `frontend`
  images; tags `sha-<short>` + `latest`; pushes to **GHCR**.
- [ ] **Deploy job (manual approval)**: SSH to the VPS → `docker compose pull && docker compose up -d` (compose references GHCR tags, not local builds).
- [ ] **Post-deploy smoke test in the pipeline**: `GET /api/health`, one mock
  `POST /api/ask`, one guard-block probe — fail loudly if any breaks.
- [ ] **Rollback** = redeploy the previous image tag (one command; document it in the
  README next to the deploy instructions).
- [ ] **GPU service deploys separately**: a RunPod/Modal template pinned to a **registry
  version** of the packaged weights (MLflow registry, 2.3); the CPU stack knows only
  `VLM_URL`.
- [ ] `.env.example` stays authoritative for config shape; real secrets only in the VPS
  env / provider secret store (3.7).

## Fine-tuning workflow (the MLOps loop)

> **Market-standard note:** at this scale, a *scripted, human-triggered loop with
> tracked runs* is the honest standard — Airflow/Kubeflow-style continuous retraining
> would be resume-driven overkill. What **is** standard and worth demonstrating:
> experiment tracking, a model registry, a blocking eval gate before deploy, and a
> feedback flywheel.

- [ ] **Data**: ChartQA splits pinned (VLM); labeled guard set built in Phase 4 (guard).
  Every dataset version recorded as an MLflow artifact/tag with its run.
- [ ] **Train**: `finetune_lora.py` on the cheapest GPU that fits (guard 1B → local
  4050; Qwen 8B → cloud per the Phase 1.5 cost ladder); all params/metrics → MLflow.
- [ ] **Eval gate (blocking)**: VLM — relaxed accuracy ≥ zero-shot baseline on the full
  2500 split; guard — recall ≥ stock **and** FPR on legit chart questions < stock
  (Phase 4 table). No pass → no registry promotion.
- [ ] **Register & package**: promote in the MLflow registry; package for serving
  (VLM: merged fp16 or 4-bit per the Phase 1.5 verdict; guard: GGUF → Ollama
  `Modelfile` → baked image).
- [ ] **Deploy** via the workflow above; **monitor** in Grafana — online accuracy has no
  ground truth, so watch the proxies: block rates, per-stage latency, VLM budget.
- [ ] **Flywheel (v2.0)**: review 👍/👎 feedback periodically; hard negatives become the
  next eval/fine-tuning set → loop back to **Data**.

## Definition of done — v1.0 portfolio launch

1. [ ] Phase 1 smoke tests green locally (mock mode, guard blocks, analysis pipeline).
2. [ ] Phase 1.5 verdict recorded: 4-bit accuracy Δ vs bf16 + fits-4050 yes/no → serving
    plan chosen (local vs cloud `min=0`).
3. [ ] Phase 2.1 dedupe merged: root `model/` + `data/` deleted, drift check in CI.
4. [ ] Phase 2.2 reproducibility: committed adapter re-evaled to 86.08%, constants
    reconciled, MODELCARDs committed.
5. [ ] Phase 2.3 MLflow up; both committed adapters backfilled into the registry.
6. [ ] Phase 3.1 GPU VLM service live behind `VLM_URL`; scale-to-zero verified (reaches
    zero within the idle timeout; cold-start time measured and surfaced in the UI).
7. [ ] Phase 3.2/3.3 frontend container live; backend image slimmed (size documented).
8. [ ] Phase 3.4 cache + rate limit + upload re-encode live.
9. [ ] Phase 3.5 dashboard live; each alert rule test-fired once.
1. [ ] Phase 3.6 Redis + Postgres in compose with healthchecks and volumes.
1. [ ] Phase 3.7 cost controls verified end-to-end: budget breaker trips in a test,
     provider spend cap set, admin UIs unreachable from the public internet.
1. [ ] Phase 4 guard comparison published (stock vs fine-tuned P/R/F1/FPR) and the ship
     gate decision recorded.
1. [ ] CI/CD green end-to-end, including the post-deploy smoke test and one rollback
     drill.
1. [ ] README updated: architecture diagram, results tables, Grafana screenshot, live
     demo link — **the actual portfolio deliverable**.

---

## Appendix — key files

- API + seam: `backend/app.py`, `backend/inference.py`, `backend/model_adapter.py`
- VLM wrapper (prod): `backend/qwen_vl_chat.py`
- Guard: `backend/guard.py`, `backend/guard_llm.py`, `guard/Dockerfile`
- Chart gate: `backend/chart_check.py`
- Orchestrator + compose: `app.py`, `docker-compose.yml`, `backend/Dockerfile`
- Config: `.env.example`
- Modeling: `modeling/chartqa/{training,evaluation,analysis,models,data}/`, `modeling/chartqa/constants.py`
- Results: `modeling/outputs/results/*.json`
- Frontend: `frontend/src/{App.jsx,api.js}`, `frontend/vite.config.js`
