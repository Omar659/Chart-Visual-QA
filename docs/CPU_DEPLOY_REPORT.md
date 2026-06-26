# CPU Containerized Stack — Validation Report

**Date:** 2026-06-25
**Scope:** Run the full input-validation stack **containerized, CPU-only** — no GPU, no
vLLM, and **no dependency on a host/external Ollama**. (vLLM/GPU is deferred: the local
NVIDIA driver predates the `cuda>=13` the vLLM image requires — see "Known limits".)

## 1. What runs where

Two containers (see [docker-compose.yml](../docker-compose.yml), [backend/Dockerfile](../backend/Dockerfile), [guard/Dockerfile](../guard/Dockerfile)):

| Container | Contents | Port | Hardware |
| --- | --- | --- | --- |
| **backend** | Flask API · **CLIP** chart detector (torch) · **Tesseract** OCR (apt) · **Layer 2** encoders (toxicity / injection / PII) · Layer 1 rules | 5000 | CPU |
| **guard** | **Ollama** serving **Llama Guard 3 1B** over the OpenAI API, **model baked into the image** | 11434 | CPU |

The backend reaches the guard over the Docker network at `http://guard:11434` — the same
OpenAI `/v1/chat/completions` endpoint used in dev, so dev↔prod is a URL change only.

## 2. Confirmed: containerized Ollama, NOT the host Ollama

| Check | Result |
| --- | --- |
| Backend `GUARD_LLM_URL` | `http://guard:11434` (Docker **service name**, not `localhost`) |
| `guard` DNS inside backend | resolves to `172.22.0.2` (container's internal IP) |
| Guard container logs | `POST /v1/chat/completions` requests from `172.22.0.3` (the **backend container**) at the smoke-test timestamps |

Traffic is **container → container** on the compose network; the host Ollama is never
involved. (Even if a host Ollama is running on `:11434`, the backend container uses the
internal `guard` service, bypassing the host entirely.)

## 3. Guard model baked in (fast cold start)

`guard/Dockerfile` pre-pulls `llama-guard3:1b` **at build time** into the image, and the
compose service mounts **no volume** over `/root/.ollama` (a volume would shadow the baked
model and force a re-pull). Result: the guard is ready **instantly** on start — confirmed
by the model being available on the first readiness poll, with **no runtime download**.
Not a gated model → **no `HF_TOKEN`** needed.

## 4. End-to-end validation (against the containerized backend, `localhost:5000`)

| Test input | Expected | Result |
| --- | --- | --- |
| Weak question (`"hi"`) | Layer 1 reject | `HTTP 400` ✓ |
| Chart image + normal question | allowed, detected | `allowed · is_chart=True (0.996)` ✓ (CLIP in container) |
| Toxic question | Layer 2 block | `BLOCKED [toxic]` ✓ |
| Prompt injection | Layer 2 block | `BLOCKED [prompt_injection]` ✓ |
| Question with an email | Layer 2 block | `BLOCKED [pii]` ✓ |
| Unsafe request ("build a bomb") | Layer 3 block | `BLOCKED [unsafe]` ✓ (Llama Guard, containerized Ollama) |
| Non-chart (noise) image | low confidence | `allowed · is_chart=False (0.241)` ✓ |

All seven behaved correctly. Layer-3 inference latency on CPU: ~0.4–6 s/request (first
call warmer-bound, then ~0.5–1.2 s) — fine for a gate.

## 5. Known limits / rough edges

- **Slow backend cold start.** The **Layer-2 encoders download at boot** (toxic-bert ~418 MB,
  the injection deberta, etc.) because, unlike CLIP, they were **not baked into the backend
  image**. On a constrained WSL2/Docker setup this download saturated resources and the
  first requests timed out until warmup finished. **Fix (next step): bake the L2 encoders
  into the backend image** like CLIP — removes the runtime download.
- **vLLM/GPU deferred.** The vLLM guard image needs `cuda>=13`; the host driver is older, so
  the GPU path errors at container init. The CPU Ollama guard above is the working path now;
  vLLM stays as a commented GPU upgrade in `docker-compose.yml` for when the driver is updated.
- **Resource pressure.** Big concurrent pulls/downloads can hang Docker Desktop on this box;
  give WSL2 more RAM (Docker Desktop → Settings → Resources) and keep disk reclaimed
  (`docker builder prune`, remove unused images).

## 6. Next steps

1. **Bake Layer-2 encoders into the backend image** (fixes §5 cold start). ← in progress
2. **Quantization + precision analysis** before shrinking models — measure accuracy lost
   (see [ARCHITECTURE.md](ARCHITECTURE.md) §12).
3. **Cloud scale-to-zero**: CPU service (all validators) `min=1`, GPU service (the main VLM)
   `min=0` (see [ARCHITECTURE.md](ARCHITECTURE.md) §11).
4. **vLLM/GPU guard** once the NVIDIA driver is updated.
