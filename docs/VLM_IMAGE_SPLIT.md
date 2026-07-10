# Plan — Split the VLM image into `vlm-base` + `vlm-service`

## Problem

Every `chartqa-vlm` redeploy rebuilds the **entire** image from scratch — including the
`apt-get`, the multi-GB pip installs (torch/transformers/peft/bitsandbytes), and the
**~17.5 GB base-model download** from Hugging Face. A code- or config-only change (e.g.
the recent `QWEN_ADAPTER_PATH` fix) therefore costs a full ~20-29 min build.

**Root cause:** Cloud Build runs on an ephemeral VM with **no persistent Docker layer
cache** between builds. So even though `vlm_service/Dockerfile` is already ordered
cheap→expensive, that ordering only helps a *local* rebuild — on Cloud Build every layer
is cold every time, and the model download reruns on every deploy.

## Solution — two images, split at the "changes rarely / changes often" line

Split the single Dockerfile into a **base image** (everything expensive + immutable) and
a **thin service image** that is `FROM` the base and only adds the fast-changing bits.
Because the base is already built and pushed to Artifact Registry, a normal service
deploy pulls it as a **single cached layer** — no apt, no pip, no model download.

### What goes where

| Layer                                    | Changes when…                          | Image         |
| ---------------------------------------- | -------------------------------------- | ------------- |
| CUDA/torch base (`pytorch/pytorch:…`)    | torch/CUDA upgrade (rare)              | **vlm-base**  |
| `build-essential` (C compiler)           | ~never                                 | **vlm-base**  |
| pip deps (transformers/peft/bnb/…)       | dependency bump (occasional)          | **vlm-base**  |
| torch/torchvision force-reinstall pin    | torch upgrade (rare)                  | **vlm-base**  |
| **base model download (~17.5 GB)**       | base-model swap (rare)                | **vlm-base**  |
| non-root user + `HF_HOME`                | ~never                                 | **vlm-base**  |
| shared wrapper `modeling/chartqa/*`      | wrapper edits (sometimes)             | **vlm-service** |
| **LoRA adapter** (`qwen3vl-lora-final2`) | model retrain (sometimes)             | **vlm-service** |
| `vlm_service/*` (`server.py`)            | serving-code edits (often)            | **vlm-service** |
| runtime `ENV` + `CMD`                    | config tweaks (often)                 | **vlm-service** |

> The shared wrapper `modeling/chartqa` is `pip install -e` (editable) in the base, so the
> service image just **re-copies** the source over it — Python imports from the files on
> disk, no reinstall needed. Its `pyproject.toml` deps are baked in the base; only a
> *dependency* change (not a code change) to chartqa needs a base rebuild.

### Container architecture

```
                         Artifact Registry (us-central1)
                         ────────────────────────────────
  BUILT RARELY                                     BUILT ON EVERY DEPLOY
  (deps / model swap)                              (code / adapter / config)

  ┌───────────────────────────────┐    FROM        ┌────────────────────────────┐
  │  vlm-base:latest   (~30 GB)    │◀───────────────│  vlm-service:latest        │
  │ ───────────────────────────── │                │ ────────────────────────── │
  │ • pytorch/cuda12.4 runtime     │                │ FROM vlm-base:latest       │
  │ • build-essential (cc)         │                │  + modeling/chartqa (src)  │
  │ • pip: transformers/peft/      │                │  + LoRA adapter (~170 MB)  │
  │        accelerate/bitsandbytes │                │  + vlm_service/ (server)   │
  │ • torch/torchvision pinned     │                │  + ENV config + CMD        │
  │ • Qwen3-VL-8B base (~17.5 GB)   │                │                            │
  │ • appuser (uid 10001), HF_HOME │                │  (shared base layers are   │
  └───────────────────────────────┘                │   deduped — thin delta)    │
        ▲                                           └────────────────────────────┘
        │ built by                                          │ built by
        │ scripts/gcloud_build_vlm_base.sh                  │ scripts/gcloud_deploy_vlm.sh
        │ (~20-29 min, run MANUALLY, rarely)                │ (~2-3 min, every deploy)
        │                                                   ▼
        │                                       ┌────────────────────────────┐
        └────────── one-time / on dep bump      │ Cloud Run: chartqa-vlm     │
                                                │  GPU nvidia-l4, min=0       │
                                                │  (private, ID-token auth)   │
                                                └────────────────────────────┘
```

### Build/deploy flow (before → after)

```
BEFORE (single image, every deploy):
  gcloud_deploy_vlm.sh
    └─ Cloud Build: apt + pip + torch pin + DOWNLOAD 17.5GB model + copy code
       └─ ~20-29 min ───────────────────────────────────────────────────► deploy

AFTER:
  gcloud_build_vlm_base.sh   (run once, and only on dep/model change)
    └─ Cloud Build: apt + pip + torch pin + DOWNLOAD 17.5GB model
       └─ ~20-29 min ─────────────────────────────► push vlm-base:latest

  gcloud_deploy_vlm.sh       (every normal deploy)
    └─ Cloud Build: FROM vlm-base (cached) + copy code/adapter/config
       └─ ~2-3 min ─────────────────────────────────────────────────────► deploy
```

## Files

- **New `vlm_service/Dockerfile.base`** — the expensive half (base image + deps + model
  download + non-root user). Build context = repo root (needs `modeling/`).
- **Rewrite `vlm_service/Dockerfile`** — now `FROM ${VLM_BASE_IMAGE}` (an `ARG` so the
  registry path is injected at build time), then `COPY --chown` the shared wrapper +
  adapter + service code, set `ENV` + `CMD`. No apt/pip/model download.
- **New `vlm_service/cloudbuild.base.yaml`** — builds `Dockerfile.base` → `vlm-base`.
  (Keeps the raised `timeout`/`diskSizeGb`, which the thin service build no longer needs.)
- **Rewrite `vlm_service/cloudbuild.yaml`** — builds the thin `Dockerfile`, passes
  `--build-arg VLM_BASE_IMAGE=…` via substitution. Normal timeout.
- **New `scripts/gcloud_build_vlm_base.sh`** — build+push `vlm-base` (rare, manual).
  Mirrors the existing deploy scripts' shape (`_gcloud_common.sh`, arg parsing, cleanup).
- **Edit `scripts/gcloud_deploy_vlm.sh`** — build the thin image against `vlm-base:latest`
  and deploy. Fail fast with a clear message if `vlm-base` doesn't exist yet.
- **Docs** — this file; a pointer in `docs/REVIEW_AND_ROADMAP.md` §3.7 and the
  `gcloud-deploy` skill (deploy order now: build base once → guard → app; service deploys
  are cheap).

## Trade-offs (stated honestly)

- **+1 Dockerfile and +1 script to maintain.** The split is only worth it because the
  service half now redeploys in single-digit minutes.
- **The base is not auto-rebuilt.** If you bump torch/transformers/bitsandbytes or swap
  the base model, you must remember to run `gcloud_build_vlm_base.sh` first — the deploy
  script can't detect a stale base. Mitigation: both the base Dockerfile and the deploy
  script name the exact trigger conditions in comments, and the deploy script prints a
  reminder.
- **Storage is ~unchanged.** `vlm-service` is `FROM vlm-base`, so Artifact Registry
  **dedupes** the shared ~30 GB — total ≈ base + a thin (~170 MB) delta, not 2×30 GB.
- **The ~170 MB adapter still uploads** in the service build's source tarball each deploy
  (~30 s). Externalizing the adapter is a possible later step, out of scope here.

## Verification

1. `bash -n` all new/changed scripts; `docker`/Cloud Build config is YAML-linted by
   `gcloud builds submit` itself.
2. Build the base once (`gcloud_build_vlm_base.sh`), confirm it pushes `vlm-base:latest`.
3. Run `gcloud_deploy_vlm.sh`, confirm the build is now **minutes** and the deployed
   revision boots (check `gcloud run services logs read chartqa-vlm` — no
   `Failed to find C compiler`, no mangled adapter path, model loads from local disk).
4. `/health` returns ready; a real `/predict` returns an answer.
