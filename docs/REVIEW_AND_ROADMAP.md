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

## Execution progress (updated 2026-07-10)

Work lives on branch **`feat/quantization-flag`** (not pushed). Backend suite: **96 passed /
4 skipped**.

**v1.0 is live in production on Cloud Run — this is the current real state, not a plan:**
- **All 4 services deployed and confirmed scale-to-zero** (`chartqa-vlm`, `chartqa-guard`,
  `chartqa-backend`, `chartqa-frontend` — `minScale` unset/0 on all four, verified via
  `gcloud run services describe`, real data not assumption; `maxScale` 1/2/3/3 respectively).
- **Required Google login shipped** — `/api/ask` and `/api/vlm/warm` verify a Google ID
  token server-side (`backend/auth.py`, stateless, no session table); anonymous access is
  gone. Global rate-limit/budget now backed by **Upstash Redis** (`REDIS_URL` set in
  prod), closing the per-instance-multiplying gap that existed when Redis was unconfigured.
- **Guard Layer 3 (Llama Guard 3 1B) deployed and actually screening**, not fail-open —
  `scripts/gcloud_deploy_guard.sh`, private + service-to-service auth (same backend SA
  used for the VLM). Two compounding cold-start bugs found and fixed: (1) Ollama's GPU
  auto-discovery wasted ~90s probing CUDA/Vulkan on every cold start of a CPU-only
  container → fixed with `OLLAMA_LLM_LIBRARY=cpu`; (2) `GUARD_LLM_TIMEOUT=20s` was
  shorter than real cold model-load time, so a client disconnect made Ollama abort and
  restart the load — a self-defeating race that meant guard could never finish a cold
  load in time → fixed by raising the timeout to 60s.
- **VLM latency fixed** — `QWEN_QUANTIZATION` switched `4bit` → `none` on the 24 GB L4
  production GPU. Root cause: bitsandbytes dequantizes weights on every forward pass
  (a training/small-GPU optimization, not a serving one) and blocks LoRA
  `merge_and_unload()`, stacking a second per-layer cost. 4-bit was only ever needed for
  the 6 GB dev 4050; full bf16 fits the 24 GB L4 comfortably and — because Cloud Run GPU
  bills by instance-active time, not VRAM — is the same or lower cost. Documented as a
  standing finding (project memory + Phase 1.5 §Stage C3 cross-ref): if quantization is
  ever needed again, prefer **AWQ/GPTQ via vLLM/TensorRT-LLM** (purpose-built serving
  kernels) over bitsandbytes.
- **SSE streaming with real per-stage progress** — `POST /api/ask/stream`, guard and
  chart-gate now run genuinely concurrently via `ThreadPoolExecutor` + `as_completed()`
  (overlapping I/O waits — Llama Guard's HTTP round-trip, Tesseract's OCR subprocess —
  not true CPU parallelism on Cloud Run's single-vCPU allocation). Frontend renders a
  live stage list ("Verifying image" / "Verifying question" / "Processing model") with
  a spinner-to-checkmark transition and per-stage elapsed time. Caught and fixed a real
  concurrency bug in the process: sequential `.result()` calls corrupted the
  second-checked future's elapsed-time measurement; fixed with a `_timed()` wrapper
  combined with `as_completed()` for order-independent, correctly-timed events.
- **Image split for fast redeploys** — `vlm-base` (rarely-changing deps + baked model,
  built via a separate rare script) with a thin `chartqa-vlm` service image `FROM` it.
  Live-timed: **10m8s** thin redeploy vs. ~20–29min monolithic (Cloud Build has no
  persistent cross-build layer cache, so this split is the only lever available).
- **Chart hard-block** — `CHART_BLOCK_THRESHOLD` (0.4, raised from an initial 0.3 after a
  real off-topic photo scored 0.37 and still reached the VLM) short-circuits confidently
  non-chart uploads before the GPU call.
- **`/metrics` (Prometheus) live in prod** — per-stage latency histograms + counters,
  used to root-cause every fix above from real numbers, not guesses.

**v1.0 modeling & reproducibility (done, prior sessions):**
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

**Phase 3.6 data layer + 3.7 cost/abuse controls (done this session):**
- **3.6** ✅ **Docker-verified** — `redis` + `postgres` + a Postgres-backed `mlflow`
  server, all localhost-bound, healthchecked, with volumes. One shared fail-open
  `backend/redis_client.py` behind the answer cache, rate limiter, and VLM budget.
  Verified live: redis PONG, `pg_isready`, MLflow `/health` 200 + Postgres-backed API.
  (Also closes the 2.3 "mlflow service backed by Postgres" item.)
- **3.7** ✅ (partial) — **per-IP rate limit** (`ratelimit.py`, Redis+in-memory, 429 +
  metric), **daily VLM budget breaker** (`budget.py`, refuses before the GPU, 429 +
  metric), **guard fail-open metric** (closes the 3.5 gap), **non-root containers**
  (frontend Docker-verified uid 101; backend/vlm_service same pattern, images not rebuilt
  this session — flaky network). 17 new backend tests (suite **80 passed / 4 skipped**).
  Remaining: global VLM concurrency cap, Grafana alerts, edge/Cloudflare, prod spend cap.

**3.8 dev/prod GPU serving automation (done this session):**
- **RunPod dev pod deployed manually first** — real Qwen3-VL-8B + LoRA (4-bit) served
  from a RunPod A5000, answered correctly through the full local stack (Docker
  backend+frontend+guard → SSH tunnel → pod). Confirmed the whole pipeline works
  end-to-end with the real model, not just the mock.
- **`scripts/runpod_up.py` / `runpod_down.py`** ✅ — that manual process automated:
  fresh pod → upload code+adapter → install deps (every fix from the manual run baked
  in, see `docs/RUNPOD_NOTES.md`) → launch → SSH tunnel → prints `VLM_URL`. Idempotent
  teardown; idle watchdog (30 min) + signal/error traps when run standalone.
  **Live-tested** (not just written): first real run caught a genuine bug (`ssh info`
  can 404 "pod not ready" even after the pod's own status reads RUNNING) — fixed with a
  retry loop, verified again.
- **`docs/RUNPOD_NOTES.md`** ✅ — every gotcha hit going from zero to a working pod,
  written up so the fixes don't need rediscovering (torch/torchvision version pinning,
  `HF_HOME`/`PIP_CACHE_DIR` not inherited by SSH sessions, PEP 668, port conflicts, ...).
- **`vlm_service/` made Cloud-Run-GPU-ready** ✅ — `server.py` and the `Dockerfile`'s
  `CMD` now honor `$PORT` when set (Cloud Run) and fall back to `VLM_PORT` otherwise
  (RunPod/local) — **same image, no fork**. `Dockerfile` also force-pins the matched
  torch/torchvision pair (same lesson as the RunPod script) as a build-time safety net.
- **`backend/vlm_provider.py`** ✅ — the one place that decides "is the VLM running,
  should I start it," used by both `GET /api/vlm/warm` (frontend calls it on page load)
  and `POST /api/ask` (before calling `model_adapter.predict`) — so the two call sites
  can't diverge. `VLM_PROVIDER=none|cloudrun|runpod` selects the strategy; `none` (the
  existing default) is a no-op, zero behavior change for docker-compose.
- **`scripts/gcloud_deploy_vlm.sh` / `gcloud_deploy_app.sh`** ✅ — two separate,
  CLI-automated Cloud Run deploys (Cloud Build → Artifact Registry → `gcloud run
  deploy`): the GPU model service (scale-to-zero, `nvidia-l4`) and the CPU app
  (backend+frontend) independently, so an app change never rebuilds the multi-GB CUDA
  image. `frontend/nginx.conf.template` is now env-substituted at container start
  (`BACKEND_URL`/`DNS_RESOLVER`/`PORT`) so the same frontend image works unmodified in
  docker-compose and Cloud Run.
- **Not yet live-tested**: the two `gcloud_deploy_*.sh` scripts are real, correctly-flagged
  code (verified against `gcloud`'s actual CLI help, not guessed), but deploying to Cloud
  Run needs a GCP project with billing + Cloud Run GPU quota — that's the next real-money
  step, still to be run for the first time.
- **Reviewer pass found + fixed 3 real bugs** before this was called done: (1)
  `vlm_provider._ensure_runpod` reused `VLM_TIMEOUT` (a single-HTTP-call budget) as the
  timeout for the whole multi-minute pod-provisioning subprocess — on expiry the child
  was SIGKILLed mid-provisioning with no chance to run its own teardown, leaking a
  billed pod; fixed with a dedicated `RUNPOD_PROVISION_TIMEOUT_S` (900s) + a best-effort
  `runpod_down.py` call in the timeout handler. (2) The same function's "someone else is
  already provisioning, poll instead of starting a second pod" branch was dead code — the
  starting-flag was set/cleared inside the same lock the slow subprocess call ran under,
  so a second caller could only block on the lock, never observe the flag; fixed by
  releasing the lock before the subprocess call. (3) `runpod_up.py`'s tunnel PID lookup
  shelled out to `pgrep`, which doesn't exist on Windows/Git Bash — this project's own
  dev platform; fixed by launching the tunnel via `subprocess.Popen` directly (no
  external process-lookup tool needed at all). Also fixed: the frontend nginx template
  forwarded the wrong `Host` header to the backend (`$host` — the frontend's own
  hostname — instead of `$proxy_host`), which works by accident against
  docker-compose's `backend:5000` but breaks Cloud Run's Host-based routing to the
  backend service; added `proxy_ssl_server_name on` for correct SNI too. Re-verified
  after fixes: `py_compile` clean, backend pytest 63/4 unchanged, frontend rebuilt +
  `nginx -t` passes with the corrected template, `/api/health` and `/api/vlm/warm` both
  live-verified again.

**Open items:** env pollution (`datasets`/`mlflow`/`bitsandbytes`/`accelerate`/`peft`/
`prometheus-client` in `backend/.venv`; no dedicated modeling venv); dataset-source decision
(HuggingFaceM4 vs lmms-lab); Grafana alert rules not test-fired; provider spend cap on the
prod GCP project; Cloudflare/edge; base-image digests; nothing pushed to `main` yet.

**Decision (2026-07-10): Phase 4 (fine-tune Llama Guard) is deprioritized for now** — the
stock 1B Llama Guard is live and screening correctly; revisit only if a real false-
positive/negative pattern shows up in `/metrics`. **Phase 5 (Conversational v2.0) is the
next target** — see that section below; its framework decision (no LangChain, native
chat-template + sliding window + Redis/Postgres) was already settled before this session
and doesn't need to be revisited.

---

## 0. Executive summary

> **Update (2026-07-10):** the whole stack is live on Cloud Run — GPU VLM (de-quantized,
> bf16), guard L3, backend, frontend, all scale-to-zero, gated behind required Google
> login. Left as originally written for history; see the "Execution progress" section at
> the top of this doc for current status.

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
| Real VLM (Qwen3-VL-8B + LoRA)    | `backend/qwen_vl_chat.py`, `vlm_service/`                          | ✅ Wired + deployable | Quant support (4/8-bit); GPU service exists (`vlm_service/`), deploys via RunPod (dev) or Cloud Run GPU (prod, Phase 3.8) |
| Layer-1 guard (cheap rules)      | `backend/app.py`                                                   | ✅ Integrated                       | Empty/weak-question + image-present checks → 400                                                        |
| Layer-2 guard (encoders)         | `backend/guard.py`                                                 | ✅ Integrated + tested              | detoxify toxicity, deberta prompt-injection, Presidio PII; fail-open                                     |
| Layer-3 guard (Llama Guard 3 1B) | `backend/guard_llm.py`, `guard/Dockerfile`                       | ✅ Integrated + tested              | Ollama, CPU, baked at build; custom`S99` off-topic code; fail-open                                     |
| Chart gate                       | `backend/chart_check.py`                                           | ✅ Integrated + tested              | CLIP zero-shot + OCR "has-data" gate + pixel-heuristic fallback                                          |
| Orchestrator                     | `app.py` (root)                                                    | ✅ Integrated                       | `--dev` (venv) vs prod (`docker compose up backend guard` + host Vite)                               |
| Containers                       | `docker-compose.yml`, `backend/`, `guard/`, `frontend/`, `vlm_service/` | ✅ Complete           | backend + guard + frontend (nginx) + observability in compose; GPU VLM containerized separately (RunPod/Cloud Run, Phase 3.8) |
| Training pipeline                | `modeling/chartqa/training/finetune_lora.py`                       | ✅ Complete                         | Custom loop, PEFT LoRA, bf16, grad-checkpoint, cosine, best-by-val-loss                                  |
| Eval + error analysis            | `modeling/chartqa/evaluation/`, `modeling/chartqa/analysis/`     | ✅ Complete                         | relaxed (5%) + exact; error dumps; disagreement sets; category table                                     |
| Committed adapters               | `modeling/checkpoints/{qwen3vl-lora-final2, blip2-lora-final}`     | ✅ Present (LFS)                    | Real runs (BLIP-2 trainer_state logs 700+ steps)                                                         |

**Integration verdict:** the mock-first contract held — the frontend never has to change.
The real model dropped cleanly behind `model_adapter.predict`. The **GPU serving path**
now exists and has answered real questions end-to-end (RunPod dev pod, Phase 3.8); what's
left is running it on **Cloud Run** for the first time (needs a real GCP project/billing).

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

1. **[High] RESOLVED (Phase 3.1/3.8).** The VLM ran in-process in a CPU-only image; the
   scale-to-zero GPU service now exists (`vlm_service/`) and deploys via RunPod (dev) or
   Cloud Run GPU (prod).
2. **[Med] RESOLVED (Phase 1.5/3.1).** `merge_and_unload()` at load would have blocked
   quantized serving — **decided against**: the adapter stays attached to a quantized
   base (Path 1, no merge), which is what `QwenVLChat`/`vlm_service` actually do.
3. **[Med] RESOLVED (Phase 2.1).** Source duplication across three `qwen_vl_chat.py`
   copies — the repo-root `model/`/`data/` leftovers are deleted; `backend/`'s copy stays
   an intentionally vendored, CI-drift-checked copy of `modeling/chartqa/models/`.
4. **[Med] RESOLVED (Phase 2.2).** Config↔checkpoint drift — `constants.py` now matches
   the committed Qwen adapter (`r=16/alpha=32`, 7 LM-only modules, `max_steps=200`
   recovered from `training_args.bin`); MODELCARDs document the provenance caveats.
5. **[Med] Answer cache + upload re-encode DONE (Phase 3.4); rate limit still open.**
   Cache by `(image_hash, question)` short-circuits repeats (`answer_cache.py`);
   re-encoding uploads via PIL strips malicious payloads (`uploads.py`). Rate limiting
   is still not implemented — needs Redis (3.6) to survive restarts/hold across workers.
6. **[Med] RESOLVED (Phase 3.5).** Per-stage latency + a `/metrics` endpoint now exist
   (`metrics.py`, fail-open Prometheus) — which layer spends the time and how often the
   expensive VLM path fires are both answerable. Alert rules still open.
7. **[Med] RESOLVED (Phase 2.3, eval path).** Training/eval runs used to live in ad-hoc
   JSONs; `evaluate.py` now logs every run to MLflow, directly fixing the "which config
   produced this checkpoint" failure class. `finetune_lora.py` instrumentation + the
   Model Registry are still open (see 2.3).
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

- [X] Smoke-test the stack in mock mode: `python app.py --dev` → open `http://localhost:5173`,
  upload a `dataset/*.png`, confirm answer + chart-detection pill + latency. **Done** — and
  repeated later in **real** mode too (`docs/app-demo.png`, chart detected 99%, real answer).
- [X] Exercise the guard: send a toxic / prompt-injection / PII / off-topic question and
  confirm the `blocked` path (run `cd backend && pytest` for the full guard/chart suite).
  **Done** — backend suite green throughout (63 passed / 4 skipped as of Phase 3.8).
- [X] Bring up the **real** Llama Guard container: `docker compose up guard` and re-test
  Layer-3 blocking against the live `:11434` endpoint. **Done** — the `guard` container
  runs as part of the compose stack brought up repeatedly this session.
- [X] Validate the analysis pipeline with no GPU **and no VLM**: run
  `modeling/scripts/run_results_from_errors.sh` against the committed `outputs/errors/*`
  and diff against `outputs/results/*`. **Done** — see §1.2b (byte-identical to committed).
  > **Why no GPU / no model:** this step is pure post-processing of results the VLM
  > *already* produced. `results_from_errors.py` loads **no model** — it reads the
  > committed `errors.json` (the list of failed indices from a past eval), rebuilds the
  > successes as every index not in that list, and pulls only the **question text** from
  > `ChartQADataset` to label each record. The only download is the ChartQA dataset
  > (text/metadata, CPU). Same for `category_table.py` / `compare_errors.py`. So "run the
  > analysis without a GPU" = re-derive the results/tables from saved eval dumps, **not**
  > re-run inference.
  >
- [X] **(chosen — do this)** Sanity-check eval *logic* on a tiny slice with BLIP-2 4-bit:
  `python -m chartqa.evaluation.evaluate --model blip2 --metric relaxed --limit 20`
  (after adding the 4-bit flag in 1.3). This exercises the real eval path (dataset →
  model → metric → error dump) cheaply, on a model that fits the 4050. **Done** — Stage A.
- [X] **(chosen — local smoke test before any cloud)** For a real Qwen3-VL answer, confirm
  the whole real path runs end-to-end before spending on the full cloud accuracy study.
  **Superseded, not skipped**: instead of a slow local CPU smoke test, this was proven a
  stronger way — a real RunPod GPU dev pod (Phase 3.8) served real Qwen3-VL-8B + LoRA
  (4-bit) end-to-end through the full local Docker stack and answered correctly
  (`docs/app-demo.png`). Same goal (confirm the real path wires up before paying for the
  full cloud accuracy run), better evidence (GPU-accurate, not a CPU approximation).

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

- [X] Add an opt-in `load_in_4bit` / `BitsAndBytesConfig` path to the model wrappers
  (`modeling/chartqa/models/*.py`, and `backend/qwen_vl_chat.py`) behind an env flag —
  the current loaders are full-precision only. (Shared with Phase 1.5.) **Done** —
  `--quantization {none,8bit,4bit}` / `QWEN_QUANTIZATION`, NF4+double-quant+bf16, LoRA
  stays attached (no merge) when quantized, fails loudly without `bitsandbytes`.

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

- [X] **A0 — Local dry-run, no big model.** Run the eval harness end-to-end with a tiny
  slice to exercise dataset load + metrics + error dumps:
  `python -m chartqa.evaluation.evaluate --model blip2 --metric relaxed --limit 8`
  (BLIP-2 fits locally; the goal is to shake out the *harness*, not accuracy). **Done.**
- [X] **A1 — Decide the LoRA-on-quant strategy** (merge does **not** combine with a
  quantized base). **Decided + implemented: Path 1** — load base with `load_in_4bit`,
  then attach the adapter with `PeftModel.from_pretrained(base_4bit, adapter)` —
  **no merge**, no 16 GB GPU needed. This is what `QwenVLChat`/`vlm_service` actually do.
  Path 2 (pre-merge to fp16, then quantize-load) stays a documented alternative, not used.
- [ ] **A2 — Free-cloud smoke test on the *real* model.** On Kaggle/Colab T4, run **4-bit**
  on ~50-100 samples to confirm the full real path (real Qwen + real 4-bit + real metrics)
  works before the full run: `--quantization 4bit --limit 100`. Still pending — rides
  along with Stage B1 (user, on Kaggle).

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

- [X] Add `--quantization {none,8bit,4bit}` (builds the matching `BitsAndBytesConfig`). **Done.**
- [X] Record per config, alongside accuracy: **p50/p95 latency** (with
  `torch.cuda.synchronize()`, discard a warmup sample), **peak VRAM**
  (`torch.cuda.reset_peak_memory_stats()` → `torch.cuda.max_memory_allocated()`),
  **load time**, and **on-disk size**. **Done** — all logged by `evaluate.py` (2.3).
- [X] Write one results JSON per `(model, quant, metric)` and a roll-up comparison table.
  **Done** — `outputs/results/*.json` per config; the Kaggle notebook's last cell builds
  the roll-up via `mlflow.search_runs`. The **filled-in numbers** for 4-bit/8-bit are
  still pending Stage B1/C1 (tracked separately below), but the harness capability exists.
- [X] Log every run to **MLflow** (Phase 2.3) — params, metrics, hardware tag — so the
  comparison table below becomes a query, not a hand-filled artifact. **Done.**

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

- [X] Make `modeling/` an installable package (`pyproject.toml`, `pip install -e`) so
  training/eval imports stop depending on path tricks — the standard shape for a
  reusable ML package. **Done.**
- [X] **Delete** the repo-root `model/qwen_vl_chat.py` and `data/chartqa_dataset.py`
  (pre-`modeling/` leftovers). Grep for imports first; point any stragglers at
  `modeling/chartqa`. **Done.**
- [X] Keep `backend/qwen_vl_chat.py` **vendored** — the backend image must stay
  independently buildable without the modeling tree and its training-only deps — but
  make the copy *verified*: a pytest (run in CI, Phase "deploy workflow") that fails
  when the vendored file diverges from `modeling/chartqa/models/qwen_vl_chat.py`
  (hash compare), with a header comment in both files naming the counterpart. **Done**
  — `backend/tests/test_vendor_sync.py`.
- [X] Document the rule in `CLAUDE.md`: shared code lives in `modeling/chartqa`;
  vendoring into `backend/` requires the drift check. **Done.**

### 2.2 Fix modeling reproducibility

> **Primary goal (per Victor): find the config that reproduces the *same* Qwen result the
> team got (84.60% → 86.08% relaxed).** The committed adapter is the source of truth —
> reconcile the code to *it*, not the other way around.

- [X] Reconcile `constants.py` LoRA values with the committed **Qwen** adapter — pin
  `LORA_R=16, LORA_ALPHA=32` (from `qwen3vl-lora-final2/adapter_config.json`), not the
  declared `r=32/α=64`. Confirm the 7 `target_modules` match too. Document any BLIP-2 vs
  Qwen difference if the two adapters used different `r`. **Done** — per-model
  `LORA_R`/`LORA_ALPHA` dicts (qwen 16/32, blip2 32/64), qwen cut to the 7 LM-only modules.
- [X] Recover and pin the **real `max_steps`** (and lr, warmup, eff. batch) the team used —
  the current `max_steps=20` is a quick-test toy. Cross-check against `trainer_state.json`
  in each committed checkpoint. Make it a CLI arg with a loud, correct default so
  `run_trainings.sh` reproduces the committed checkpoints instead of a 20-step toy. **Done**
  — qwen **200** (from `training_args.bin`), blip2 **884** (from `trainer_state.json`).
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
- [X] Cloud runs (Kaggle/Colab): do **not** expose the tracking server to the internet —
  log to a local `mlruns/` in the session, download, and import into the tracked store
  (`mlflow-export-import`); or re-log the final metrics JSON. Cheap and safe. **Done** —
  exactly what `modeling/notebooks/kaggle_quant_eval.ipynb` does (local `mlruns/`, zipped
  and downloaded in the last cell).
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
- [X] **GPU Dockerfile built** — `vlm_service/Dockerfile` (see 3.8). A `docker-compose`
  GPU service (`deploy.resources.reservations.devices`) was the original plan here, but
  we don't have a local CUDA host to run it against, so **Cloud Run GPU** became the
  actual production target instead (3.8) — no compose GPU service needed.

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

- [~] Expose `/metrics` from Flask via `prometheus_client`. **Mostly done**:
  - [X] `http_requests_total{route, status}` (counter)
  - [X] `stage_latency_seconds{stage}` (histogram — guard/chart_gate/vlm; answers "which
    layer spends the time")
  - [X] `blocked_total{reason}`
  - [X] `answer_cache_hits_total` / `answer_cache_misses_total`
  - [X] `vlm_invocations_total` — the cost proxy that drives the budget alerts in 3.7
  - [ ] `vlm_tokens_generated_total` — **not done** (only invocation count, not tokens)
  - [ ] guard **fail-open events** as a metric — **not done** (currently only a log
    warning; a dependency silently missing in prod should page, not just log)
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

> **Status (2026-07-08): DONE + Docker-verified.** `redis`, `postgres`, and an
> `mlflow` server (Postgres-backed) are compose services, all localhost-bound. Redis
> earns its keep immediately (shared answer cache + rate-limit + VLM-budget counters);
> Postgres earns its keep immediately as the MLflow backend store. **Verified live**:
> `redis-cli ping` → PONG, `pg_isready` → accepting, MLflow `/health` → 200 and its
> Postgres-backed API returns the Default experiment.

The market-standard split (both now implemented):

| Store                          | Used for                                                                                        | Why this one                                                                                                                                           |
| ------------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Redis** (7-alpine)     | Answer cache w/ TTL (3.4) ✅ · per-IP rate-limit counters (3.7) ✅ · daily VLM budget (3.7) ✅ · v2.0 conversation hot-store (5.1) | In-memory speed, native TTL/eviction, atomic counters                                                                                                  |
| **Postgres** (16-alpine) | MLflow backend store (2.3) ✅ · v2.0 conversation persistence · feedback events (5.1)            | Durable + relational, the production default — SQLite would*work* at this scale, but Postgres is what production uses and a container makes it free |

- [X] Add `redis` + `postgres` compose services with named volumes and healthchecks;
  wire `depends_on: condition: service_healthy` for consumers. **Done** — `redis`
  (healthcheck `redis-cli ping`, no persistence — every key is a TTL'd cache/counter),
  `postgres` (healthcheck `pg_isready`, `pgdata` volume), and `mlflow` (depends on
  `postgres` **service_healthy**, `mlartifacts` volume). All bound to `127.0.0.1`.
- [X] Config via env (`REDIS_URL`), consistent with the no-in-code-defaults convention;
  **fail-open**: empty/unreachable Redis → in-memory cache + per-process rate limit +
  per-process budget, never an error. **Done** — one shared `backend/redis_client.py`
  (resolve-once, ping-checked, fail-open) behind `answer_cache.py`, `ratelimit.py`,
  `budget.py`. compose sets `REDIS_URL=redis://redis:6379/0` for the backend.
  (`DATABASE_URL` for app tables isn't needed yet — see the next item; MLflow's own
  Postgres URI is passed to the mlflow server, not the backend.)
- [X] Don't create app tables before something needs them — Postgres arrives with MLflow
  and earns its keep immediately; the app schema (conversations, messages, feedback)
  lands in Phase 5 with **Alembic** migrations. **Honored** — no app tables created;
  Postgres today is purely the MLflow store. This also closes the **2.3** "add an mlflow
  service backed by Postgres" item (`mlflow/Dockerfile` adds `psycopg2` to the official
  image; point the modeling code at it with `MLFLOW_TRACKING_URI=http://localhost:5001`).

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

- [X] **Provider-level hard cap — the real safety net.** *Dev*: the RunPod account used
  for Phase 3.8 has a `spendLimit` configured (`runpodctl user` reports it). *Prod*
  (2026-07-09): a **GCP Billing Budget** on `chartv-qa` — R$50/month, alerts at
  50/90/100% (`gcloud billing budgets create ...`, see `.claude/skills/gcloud-deploy/
  SKILL.md`). This is a **notification**, not an automatic spend-stop — no Pub/Sub
  action wired (that's a separate, higher-risk task, not built unprompted). Also fixed a
  real naming confusion: `VLM_DAILY_BUDGET` is a **count** of VLM invocations/day, NOT a
  dollar amount — it was misread as "$200/day" during setup; default lowered to 20 and
  every comment/docstring now says so explicitly (`budget.py`, `.env.example`,
  `gcloud_deploy_app.sh`). Also new: both `gcloud_deploy_*.sh` scripts now call
  `cleanup_old_images` (`scripts/_gcloud_common.sh`) after each deploy, deleting the
  OLD untagged image digest left behind by re-pushing under the same `:latest` tag —
  matters most for `vlm-service`, whose baked-model image is ~30GB and would otherwise
  accumulate pure storage cost with zero usage.
- [ ] **Cloudflare free tier in front** of the public hostname: TLS, DDoS/bot filtering,
  static-asset caching. (Alternative: Caddy + Let's Encrypt on the VPS if fewer parties
  is preferred.)
- [~] **Rate limits** (3.4/3.7, Redis-backed): per-IP limit on `/api/ask`. **DONE**
  (`backend/ratelimit.py`): fixed-window per-IP (Redis `INCR`+`EXPIRE`, atomic + shared
  across gunicorn workers; in-memory fallback), fail-open, wired as the first thing
  `/api/ask` does → **429** on over-limit + a `rate_limited_total` metric; unit-tested +
  an API 429 test. *Remaining:* a **global concurrency cap** (semaphore) on VLM calls so
  a burst queues instead of fanning out to the GPU — not done.
- [X] **Daily VLM budget + circuit breaker**: count real VLM invocations per UTC day;
  past budget, short-circuit with **HTTP 429** "demo quota reached — try tomorrow"
  *before touching the GPU* while cache/guard/mock keep working. **Done**
  (`backend/budget.py`, Redis-backed day counter + in-memory fallback, fail-open,
  `VLM_DAILY_BUDGET` env, `vlm_over_budget_total` metric); unit-tested + an API 429 test.
- [ ] **Scale-to-zero hygiene**: idle timeout ≤ 2-5 min on the GPU service; ensure
  *nothing* (uptime monitor, compose healthcheck, warm-up cron) probes the GPU service
  directly — probe the CPU gatekeeper instead; Grafana alert (3.5) on GPU-active
  minutes/day. *(RunPod side has the idle watchdog, §3.8; Cloud Run idle is native. The
  "nothing probes the GPU directly" rule holds — the frontend warm ping hits the backend's
  `/api/vlm/warm`, not the VLM. Grafana alert still open.)*
- [~] **Admin surfaces off the public ingress**: Grafana/MLflow/Prometheus/`/metrics`.
  **Done for the network binding** — Grafana, Prometheus, and now MLflow are all bound to
  `127.0.0.1` in compose (reach via SSH tunnel / private net). Grafana has an admin
  password env. `/metrics` is still served by the backend on its public port (scrape it
  from inside the private net; a proxy allowlist is the remaining hardening).

- [X] **App hardening** — **CORS pinned** via `CORS_ORIGINS` ✅; **security headers** on
  every response ✅; `MAX_CONTENT_LENGTH` enforced ✅; **upload re-encode** (3.4) ✅; VLM
  HTTP call has an explicit generous timeout (`VLM_TIMEOUT`) ✅; **containers run
  non-root** ✅ — **frontend Docker-verified** (`nginxinc/nginx-unprivileged`, confirmed
  running as **uid 101** + serving 200); backend (a `uid 10001` user with HF_HOME/
  TORCH_HOME/XDG_CACHE_HOME redirected to a chowned dir so the build-time model precache
  still works under the runtime user) and vlm_service (`uid 10001`, `HF_HOME` redirected)
  use the identical, now-proven pattern but their images were **not rebuilt this session**
  — the backend rebuild kept failing on a transient network truncation of the 400 MB
  spaCy `en_core_web_lg` wheel (a *pre-existing* build step, unrelated to the non-root
  change), so build-verify those two on the next clean network. *Remaining:* base images
  pinned by **digest** (currently tags); a guard-timeout audit.

- [ ] **Supply chain & secrets**: `pip-audit` (or Dependabot) in CI; secrets only via
  env / provider secret store — never in images or the repo; `.env` stays gitignored.
- [~] **Log hygiene**: never log raw questions or image bytes. **Currently satisfied by
  omission** — the backend doesn't log question text or image bytes anywhere (werkzeug
  logs only the request line). A *positive* structured audit log (question hash + guard
  verdict) is the remaining nice-to-have, not a leak to fix.

### 3.8 Dev/prod GPU serving: RunPod (dev) + Cloud Run GPU (prod), one image

**Why two providers.** RunPod pods give real SSH access — the only reason the torch/
cuDNN version fights below were debuggable at all — and are cheaper per unit of
interactive dev time (pay-per-second, no image build/push round-trip). But they bill
continuously while running and need manual teardown: not a fit for a public endpoint.
Cloud Run GPU's managed scale-to-zero autoscaling is the right shape for that, at the
cost of no shell access (every fix needs a rebuild+redeploy). **Same `vlm_service/`
Docker image runs unmodified in both** — only the orchestration around it differs.

**RunPod — dev loop:**
- [X] `scripts/runpod_up.py` — provisions a **fresh** pod every run (sidesteps
  stale-process/half-upgraded-dependency bugs entirely), uploads `chartqa/` +
  `vlm_service/` + the LoRA adapter, installs deps with every fix below baked in,
  launches the service, opens an SSH tunnel, prints `VLM_URL`. `--no-wait` for
  programmatic callers (used by `app.py --dev --runpod` and `backend/vlm_provider.py`);
  default (standalone) mode blocks with an idle watchdog (30 min) and tears itself
  down on Ctrl-C / any error.
- [X] `scripts/runpod_down.py` — idempotent teardown (kill tunnel, `runpodctl pod
  delete`, clear state); the single source of truth `runpod_up.py`'s own signal/error/
  idle paths, `app.py`'s shutdown, and manual use all call into.
- [X] `python app.py --dev --runpod` — provisions the pod before the backend starts,
  passes it `VLM_URL`/`VLM_PROVIDER=runpod`/`USE_MOCK=0`, tears it down in the same
  `finally` block that already handles Ctrl-C / a crashed child.
- [X] **Live-tested end to end**, not just written: the first real run of
  `runpod_up.py` surfaced a genuine bug (`runpodctl ssh info` can return "pod not
  ready" even after the pod's own status already reads RUNNING) — fixed with a retry
  loop and re-verified. `runpod_down.py` was also exercised for real (pod created,
  then correctly deleted after a mid-setup failure — confirming teardown doesn't leak
  a billed pod even on the failure path).
- [X] `docs/RUNPOD_NOTES.md` — every gotcha from going manual-SSH-debugging → working
  automated script, written up so they don't get rediscovered: `HF_HOME`/
  `PIP_CACHE_DIR` aren't inherited by SSH sessions (silently fills the small container
  disk); PEP 668 needs `--break-system-packages` + `--ignore-installed`; unpinned
  `pip install` can silently upgrade torch to a CUDA-13 build the pod's driver can't
  init, or a torchvision ABI-mismatched with torch (`torchvision::nms` missing), or
  skip torch's `nvidia-cudnn-cu12` companion package (`CUDNN_STATUS_NOT_INITIALIZED`)
  — the fix is pinning the exact matched pair **with** its normal deps, last, always;
  RunPod's own nginx already owns port 8001; `pgrep` isn't on the image.

**Cloud Run GPU — production:**
- [X] `vlm_service/server.py` + `Dockerfile` listen on `$PORT` when set (Cloud Run
  injects it) and fall back to `VLM_PORT` otherwise (RunPod/local `docker run`) — one
  `CMD`, both environments. The `Dockerfile` also force-reinstalls the exact matched
  torch/torchvision pair as a build-time safety net (same class of bug as the RunPod
  script, hit in a different environment).
- [X] **Base model baked into the image** (2026-07-09) so a `min=0` cold start loads from
  local disk (~1 min) instead of downloading 17.5 GB from HF every time — image ~30 GB,
  `HF_HUB_OFFLINE=1` at runtime, `cloudbuild.yaml` timeout raised to 3600s + 100 GB disk.
- [X] `scripts/gcloud_deploy_vlm.sh` — `gcloud builds submit` (Cloud Build, so the
  multi-GB CUDA image never needs a local Docker build) → Artifact Registry → `gcloud
  run deploy --gpu=1 --gpu-type=nvidia-l4 --min-instances=0 --max-instances=1
  --no-gpu-zonal-redundancy` (a fresh project has no zonal-redundancy quota; without the
  flag `gcloud` prompts interactively and hangs a scripted run). Prints the service URL
  for `VLM_URL`.
- [X] **Live-deployed** (2026-07-09/10) on `chartv-qa`/`us-central1`. First real deploy
  surfaced and fixed real bugs, not hypothetical ones: (1) no root `.gcloudignore` →
  `gcloud builds submit` fell back to `.gitignore`'s `checkpoints/` rule and silently
  dropped the force-added LoRA adapter from the build context (`COPY failed`) — fixed
  with a root `.gcloudignore`; (2) `bitsandbytes`/`triton` need a real C compiler for
  their CUDA-driver JIT step even on a "runtime" (non-devel) CUDA base image — every
  worker crashed on boot with `RuntimeError: Failed to find C compiler`, diagnosed via
  `gcloud run services logs read chartqa-vlm` (not guessed — the traceback was
  100%-reproducible, ruling out the initial cold-start-timeout hypothesis), fixed by
  installing `build-essential` in `vlm_service/Dockerfile`. See
  `.claude/skills/gcloud-deploy/SKILL.md`'s gotchas section for the full list.
- [X] **Auth: the GPU service is PRIVATE**. `gcloud_deploy_vlm.sh` deploys it
  `--no-allow-unauthenticated`; `backend/gcp_auth.py`'s `fetch_id_token()` (extracted
  from `model_adapter.py` so `guard_llm.py` can share it, see 3.7 below) attaches a
  Google-signed **ID token** (from the Cloud Run metadata server, no new dependency,
  audience = the VLM's URL) when `VLM_AUTH=gcp_id_token`; `gcloud_deploy_app.sh` creates
  a backend service account, grants it `run.invoker` on `chartqa-vlm`, and deploys the
  backend with `--service-account` + `VLM_AUTH=gcp_id_token`. So the expensive endpoint
  can't be hit by the public internet — only by this backend (whose own rate-limit +
  daily budget cap cost on the public path). `VLM_AUTH=none` (default) keeps the RunPod
  tunnel / docker-compose / dev paths unauthenticated.

**Guard Layer 3 deploy — private, CPU, same auth pattern as the GPU:**
- [X] `scripts/gcloud_deploy_guard.sh` (new) — mirrors `gcloud_deploy_vlm.sh`'s shape but
  simpler: `guard/Dockerfile` bakes the 1B model (Ollama), it's CPU-only, no custom
  `cloudbuild.yaml` needed (image is small — a plain `gcloud builds submit --tag`
  suffices). Deployed `--no-allow-unauthenticated`, `--min-instances=0`.
- [X] **Fixed a real Cloud-Run-incompatibility in `guard/Dockerfile`** while wiring this
  up: the base `ollama/ollama:latest` image's `CMD` hardcodes Ollama's own default bind
  address (`:11434`), but Cloud Run requires the container to listen on whatever `$PORT`
  it injects (usually 8080) or the revision never becomes ready. Overrode `CMD` with
  `OLLAMA_HOST=0.0.0.0:${PORT:-11434} exec ollama serve` — falls back to 11434 when
  `$PORT` is unset (docker-compose), binds to Cloud Run's port otherwise.
- [X] Auth: same pattern as the GPU — `GUARD_LLM_AUTH=gcp_id_token` (new env, mirrors
  `VLM_AUTH`), `backend/guard_llm.py` attaches an ID token via `gcp_auth.auth_header()`;
  `gcloud_deploy_app.sh --guard-url` grants the same backend service account
  `run.invoker` on `chartqa-guard` too (one identity, two grants).

**App deploy (backend + frontend, no GPU) — separate from the model on purpose:**
- [X] `scripts/gcloud_deploy_app.sh` — builds+deploys `backend/` and `frontend/` to
  Cloud Run independently of the model service, so an app-only change never rebuilds
  the CUDA image. `--vlm-url` wires a deployed `vlm_service` in (`USE_MOCK=0`,
  `VLM_PROVIDER=cloudrun`); omitted, it deploys self-contained in `USE_MOCK=1`.
  `--guard-url` wires up Guard Layer 3 (`GUARD_LLM_ENABLED=1`); omitted, it stays off
  (fails open, Layers 1/2 still run). `--redis-url` (e.g. an Upstash `rediss://` URL)
  makes rate-limit/VLM-budget counters global across backend instances instead of
  per-instance in-memory — chosen over GCP Memorystore because Memorystore needs a VPC
  connector and costs $35-50+/month even at the smallest tier, incompatible with the
  R$50/month ceiling this deploy targets. `--google-client-id` turns on required Google
  login (see below); omitted, deploys with no login wall.
- [X] `frontend/nginx.conf.template` — `PORT`/`BACKEND_URL`/`DNS_RESOLVER` are
  env-substituted at container start (nginx:alpine's built-in `envsubst`-on-templates
  entrypoint step), so the identical frontend image proxies to `http://backend:5000`
  in docker-compose and to the backend's actual Cloud Run HTTPS URL in production —
  no frontend code fork.

**Required Google login (2026-07-10) — stateless ID-token verification, no new DB table:**
- [X] `backend/auth.py` — `verify_google_token()` wraps `google-auth`'s
  `verify_oauth2_token` (audience = `GOOGLE_CLIENT_ID`); returns decoded claims or
  `None` on any failure (expired, bad signature, wrong audience). `AUTH_ENABLED` gates
  it (default `0`, matches local dev / the `GUARD_ENABLED`-style opt-in convention).
- [X] `backend/app.py` — `/api/ask` and `/api/vlm/warm` check `Authorization: Bearer
  <token>` first (cheapest-check-first, before the rate limiter) when `AUTH_ENABLED=1`;
  missing/invalid → `401`. `/api/health` and `/metrics` stay open for monitoring.
  `ratelimit.allow()`'s key switches from client IP to the authenticated user's Google
  `sub` when auth is on (a real stable identity beats IP once one exists).
- [X] Frontend: Google Identity Services script tag (`index.html`), a sign-in gate in
  `App.jsx` (renders instead of the question form until a token exists, `sessionStorage`
  so a refresh doesn't sign the user out, `warmVlm()` moved to fire on sign-in instead
  of page load — no reason to wake the billed GPU for an unauthenticated visitor). Token
  sent as a Bearer header on every `askQuestion`/`warmVlm` call (`api.js`); a `401`
  response clears the stored token and re-shows the sign-in gate.
- [X] `VITE_GOOGLE_CLIENT_ID` is baked in at **build** time (Vite's `import.meta.env` is
  compile-time, unlike the nginx template's runtime `envsubst`) — new
  `frontend/cloudbuild.yaml` + `frontend/Dockerfile` `ARG`/`ENV`, wired through
  `gcloud_deploy_app.sh --google-client-id`.
- [ ] **Manual, one-time, not scriptable**: create the OAuth 2.0 Web Client ID in Google
  Cloud Console (Authorized JavaScript origins = the frontend's Cloud Run URL) and the
  Upstash Redis database — see `.claude/skills/gcloud-deploy/SKILL.md` for the exact
  steps.
- [ ] **Not yet live-tested end-to-end** — the code is unit-tested (`backend/tests/
  test_auth.py` mocks the real `google.oauth2.id_token.verify_oauth2_token` boundary,
  never the project's own wrapper; `test_api.py` covers the 401 paths; 90 passed / 4
  skipped) and `npm run build` is verified, but the actual Cloud Run redeploy with
  `--guard-url`/`--redis-url`/`--google-client-id` + a real browser smoke test (sign in,
  ask a question, confirm a second session shares the Redis-backed rate limit) hasn't
  run yet.

**Warm-start UX (shared code, not duplicated per provider):**
- [X] `backend/vlm_provider.py` is the one place that decides "is the VLM running,
  should I start it" — both `GET /api/vlm/warm` (the frontend calls this on page load,
  fire-and-forget) and `POST /api/ask` (before calling `model_adapter.predict`) go
  through the same `warm()` / `ensure_running()`, so the two call sites can never
  diverge into two different ideas of "running". `VLM_PROVIDER=none` (existing
  default) is a no-op — zero behavior change for the current docker-compose stack.
  `cloudrun`: Cloud Run's own request-triggered autoscaler already does "start if not
  running" — `warm()` just sends an early nudge to shave cold-start latency off the
  first real request. `runpod`: no autoscaler exists, so `warm()`/`ensure_running()`
  actually provisions a pod (`runpod_up.py --no-wait`) in a background thread and
  patches `VLM_URL` in-process once ready — dev-only, unreachable from the production
  image (no `runpodctl`/`scripts/`/SSH key there, and prod's `.env` never sets
  `VLM_PROVIDER=runpod`).

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
  Postgres (`conversation_id, question, image_hash, model_answer, vote, note`). This is
  the market-standard **data flywheel**, but it is a *sourcing* mechanism, not a
  training set by itself — a 👎 only proves the answer was wrong, it doesn't carry the
  correct one. The actual loop (decision 2026-07-10, matches the "scripted, human-
  triggered, not continuous" MLOps note below): periodically review 👎 rows → write the
  correct answer for each (human-in-the-loop, unavoidable for VQA) → append curated
  `(image, question, answer)` triples to a versioned `feedback_vN.jsonl` (MLflow
  artifact) → re-run `finetune_lora.py` from the current adapter once there's a
  meaningful batch → same **blocking eval gate** as any other run (≥ baseline on the
  full 2500-sample split) before registry promotion. No auto-retraining on a schedule
  or per-vote — that repeats the exact "continuous retraining is resume-driven
  overkill at this scale" mistake the master checklist already warns against.

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

### 5.4 Scope boundary — where this project ends (decision, 2026-07-10)

**This project's ceiling is Phase 5 (5.1 + 5.2, feedback flywheel included).** That's a
complete, coherent portfolio story: zero-shot vs. fine-tuned VLM comparison → production
deploy with a real 3-layer guard → multi-turn chat with memory and a feedback loop. Each
piece reuses infrastructure already built (SSE, Redis, Postgres, auth) — no new stack.

**A RAG + agents project is a new, separate repo, not an extension of this one** — the
product story, data shape, and failure modes are different enough (retrieval quality,
tool-calling/multi-step planning, grounding/citations) that bolting it on here would
blur both portfolio pieces rather than strengthen either. Two concrete forks worth
tracking if picked up later:
- **Tool-calling on structured data** (e.g. a car-rental-style booking assistant) is
  the natural extension of 5.3's "extract the chart's table, query that" stretch goal —
  same shape of problem, could plausibly stay in *this* repo if it ever happens.
- **RAG-over-documents** (e.g. Q&A over a Drive/document corpus) is a genuinely
  different product and belongs in a **new repo** — though it can reuse this project's
  Cloud Run/Docker/auth scaffolding directly rather than starting from zero.

**Where user feedback belongs**: *in this project*, not deferred to the next one — 5.1's
`POST /api/feedback` is already scoped and cheap (one Postgres table + a button), and a
working feedback flywheel (reviewed 👎 answers → next eval/fine-tune set) is itself a
strong, differentiated portfolio signal that most single-shot VQA demos don't have.

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
| VLM serving                    | Qwen3-VL-8B + LoRA behind a **Flask wrapper** (`vlm_service/`), scale-to-zero (`min=0`) — **vLLM decided against for now** (documented swap-in later if throughput demands) | 3.1/3.8 |
| Cache / rate-limit / hot-store | **Redis 7**                                                                                    | 3.4/3.6                       |
| Durable DB                     | **Postgres 16** (MLflow store · v2.0 transcripts · feedback)                                 | 2.3/3.6                       |
| Experiment tracking            | **MLflow** server + Model Registry (Postgres-backed)                                           | 2.3                           |
| Metrics / dashboards / alerts  | **Prometheus + Grafana** (dashboard JSON provisioned in-repo)                                  | 3.5                           |
| Ingress / TLS / bot filter     | **Cloudflare free tier** (or Caddy + Let's Encrypt) — in front of Cloud Run                    | 3.7                           |
| CI/CD                          | **GitHub Actions** for CI (lint/test/build) on every PR; deploy via `scripts/gcloud_deploy_*.sh` — **the VPS/GHCR/`docker compose pull` plan below was decided against** in favor of Cloud Run | below |
| Hosts                          | **Cloud Run** for both the CPU app (backend+frontend) and the GPU model (scale-to-zero) — **RunPod/Modal serverless decided against for prod**; RunPod is the **dev-only** GPU loop (3.8) | 3.1/3.8 |

## Deploy workflow (CI/CD)

> **Superseded (2026-07-08):** the original plan below was GitHub Actions → GHCR →
> `docker compose pull` on a VPS, with the GPU model on a RunPod/Modal serverless
> template. That's decided against — **Cloud Run** (both the CPU app and the GPU model)
> is the actual target, deployed via `scripts/gcloud_deploy_app.sh` /
> `gcloud_deploy_vlm.sh` (Cloud Build → Artifact Registry → `gcloud run deploy`), not a
> VPS pull. RunPod is kept as the **dev-only** GPU loop (Phase 3.8), not a prod host.

- [ ] **CI on every PR**: ruff + backend `pytest` (including the 2.1 vendored-file drift
  check) + `npm run build` — branch protection already forces the PR flow. Not set up yet
  (no `.github/workflows/`).
- [X] **Build + deploy, GPU model**: `scripts/gcloud_deploy_vlm.sh` — Cloud Build →
  Artifact Registry → `gcloud run deploy --gpu=1 --gpu-type=nvidia-l4 --min-instances=0`.
  Script written and verified against `gcloud`'s real CLI; not yet run against a live GCP
  project (needs billing + Cloud Run GPU quota).
- [X] **Build + deploy, CPU app**: `scripts/gcloud_deploy_app.sh` — backend + frontend to
  Cloud Run, independently of the model deploy (`--vlm-url` wires a deployed model in).
  Same not-yet-live-tested caveat as above.
- [ ] **Post-deploy smoke test in the pipeline**: `GET /api/health`, one mock
  `POST /api/ask`, one guard-block probe — fail loudly if any breaks. Not automated yet
  (the deploy scripts print the service URL but don't smoke-test it themselves).
- [ ] **Rollback** = `gcloud run services update-traffic --to-revisions` to the previous
  Cloud Run revision (Cloud Run keeps revision history natively — simpler than the
  original VPS plan's "redeploy the previous image tag"). Not scripted yet.
- [ ] `.env.example` stays authoritative for config shape; real secrets for Cloud Run go
  via `--set-env-vars` / Secret Manager (3.7), never baked into an image.

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
  > **Production finding (2026-07-10, live on the L4):** the 4-bit choice in Phase 1.5
  > was driven by fitting the **6 GB dev 4050** — it does not carry over to the
  > production **L4 (24 GB)**, where an 8B model fits in bf16 with room to spare. Ran
  > both live: 4-bit (bitsandbytes NF4, unmerged LoRA — quantized bases can't
  > `merge_and_unload()`) averaged **~82s/answer**; switching to bf16 (`none`, LoRA
  > merged) dropped that dramatically, confirmed via `/metrics`. Root cause:
  > bitsandbytes dequantizes weights on every forward pass (recurring per-token cost
  > during generation, not a one-time load cost), and it's built for *training-time*
  > memory savings, not optimized serving throughput. **Rule of thumb**: don't quantize
  > unless there's a real memory constraint (small GPU, or need for a bigger batch than
  > fits unquantized) — full precision is free when the GPU has headroom, and is
  > strictly better on accuracy too. If a future model/GPU combo *does* need
  > quantization for serving, prefer **AWQ or GPTQ via a serving-optimized runtime
  > (vLLM, TensorRT-LLM)** over bitsandbytes — purpose-built inference kernels avoid
  > most of the per-forward dequant tax (see Stage C3 below, already flagged as the
  > extension path if NF4 accuracy is ever a problem).
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
6. [X] Phase 3.1 GPU VLM service live behind `VLM_URL` on Cloud Run; scale-to-zero
    **verified with real `gcloud` data** (`minScale` unset/0 on all 4 services). Cold-start
    time not yet surfaced in the UI (nice-to-have, not blocking).
7. [X] Phase 3.2/3.3 frontend container live; backend image slimmed (non-root, `.dockerignore`
    hardened).
8. [X] Phase 3.4 cache + **rate limit** + upload re-encode live, **and now global**
    (Upstash Redis in prod, closes the per-instance-multiplying gap). Global VLM
    concurrency cap still open.
9. [~] Phase 3.5 dashboard live (local compose only — no Grafana in prod, `/metrics`
    read directly via curl); each alert rule test-fired once. Alert rules still open.
10. [X] Phase 3.6 Redis + Postgres (+ MLflow server) in compose with healthchecks and
    volumes. **Done + Docker-verified** (redis PONG, pg_isready, MLflow /health 200).
    Prod uses Upstash Redis; Postgres/MLflow remain local-only (no prod durable DB yet —
    needed for Phase 5 transcripts/feedback).
11. [X] Phase 3.7 cost controls: budget breaker trips in a test ✅, rate limit ✅ **and
    global** ✅, non-root containers ✅, admin UIs localhost-bound ✅, **required Google
    login** ✅ (bigger abuse lever than rate-limiting alone). **Remaining**: provider
    spend cap for the *prod* GCP project; Cloudflare/edge; base-image digests.
12. [~] Phase 4 — **deprioritized by decision (2026-07-10)**; stock Llama Guard 3 1B is
     live and screening correctly in prod. Revisit only if `/metrics` shows a real
     false-positive/negative pattern worth fine-tuning against.
13. [ ] Phase 5 — Conversational v2.0 not started; see that section for the settled
     framework decision and checklist. **Proposed next phase.**
14. [ ] CI/CD green end-to-end, including the post-deploy smoke test and one rollback
     drill. Deploys today are manual via `scripts/gcloud_deploy_*.sh`, not yet in CI.
15. [ ] README updated: architecture diagram, results tables, live demo link (no Grafana
     screenshot — not deployed to prod) — **the actual portfolio deliverable**.

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
- Cloud Run deploy: `scripts/gcloud_deploy_{vlm,app}.sh`, `scripts/_gcloud_common.sh`,
  `.claude/skills/gcloud-deploy/SKILL.md` (the `/gcloud-deploy` operational playbook)
- Frontend: `frontend/src/{App.jsx,api.js}`, `frontend/vite.config.js`
