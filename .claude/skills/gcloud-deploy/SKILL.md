---
name: gcloud-deploy
description: Deploy Chart-Visual-QA to Google Cloud Run (CPU app + GPU model + CPU guard), and the operational commands around it — cost safety (billing budget, image cleanup, daily VLM count cap, global rate-limit via Redis), auth between the CPU backend and the private GPU/guard services, required Google login, status/logs checks, and teardown/rollback. Use this whenever the user asks to deploy to Cloud Run, check cloud costs/status, or roll back a Cloud Run deploy.
---
# Cloud Run deploy — Chart-Visual-QA

Three independent deploys, three scripts, deployed in this order:

1. **`scripts/gcloud_deploy_vlm.sh`** — the GPU model (`vlm_service/`), Qwen3-VL-8B +
   LoRA, on Cloud Run GPU (`nvidia-l4`, scale-to-zero). **Deployed PRIVATE** (no
   `--allow-unauthenticated`) — the expensive endpoint must never be callable from the
   public internet.
2. **`scripts/gcloud_deploy_guard.sh`** — Guard Layer 3 (`guard/`, Llama Guard 3 1B via
   Ollama) on Cloud Run CPU, scale-to-zero. Also **deployed PRIVATE** — same reasoning as
   the GPU, just cheaper compute. Optional: the app still works with Guard Layer 3 off
   (Layers 1/2 — rules + toxicity/injection/PII encoders — run regardless).
3. **`scripts/gcloud_deploy_app.sh --vlm-url <step-1-URL>/predict [--guard-url <step-2-URL>] [--redis-url <upstash-url>] [--google-client-id <id>]`**
   — the CPU app (backend + frontend), public. Creates the backend's own service
   account, grants it `run.invoker` on whichever of the GPU/guard services are wired up,
   and sets `VLM_AUTH=gcp_id_token` / `GUARD_LLM_AUTH=gcp_id_token` so the backend can
   call them (`backend/gcp_auth.py` fetches a Google-signed ID token from the Cloud Run
   metadata server — no extra dependency, shared by both call sites). `--google-client-id`
   turns on required Google sign-in (see "Required Google login" below); omit it to
   deploy with no login wall.

Read all three scripts' own header comments before running them — they're the source of
truth for flags. This skill is the *why* and the *operational* commands around them.

## Pre-flight checklist (do this before running either script)

```bash
gcloud config get-value account          # confirm the right Google account
gcloud config get-value project          # confirm the right project
gcloud billing projects describe <PROJECT> --format="value(billingEnabled)"  # must be True
```

**Cloud Run GPU needs an explicit quota** (new projects start at 0). Check/request in the
console: IAM & Admin → Quotas → filter `Cloud Run Admin API` + the target region (default
`us-central1`) → **"Total Nvidia L4 GPU allocation"**. Can be instant or take hours —
check this FIRST, it's the long-lead-time item.

## Deploy sequence

```bash
# 1. GPU model (private). Builds a ~30GB image (base model is BAKED IN, see below) via
#    Cloud Build — 10-30 min, mostly the model download + push.
./scripts/gcloud_deploy_vlm.sh --project <PROJECT> --region us-central1

# 2. Guard Layer 3 (private, CPU, small image — a few minutes). Optional but recommended.
./scripts/gcloud_deploy_guard.sh --project <PROJECT> --region us-central1

# 3. CPU app, pointed at both private services' printed URLs.
./scripts/gcloud_deploy_app.sh --project <PROJECT> --region us-central1 \
  --vlm-url https://chartqa-vlm-XXXX.run.app/predict \
  --guard-url https://chartqa-guard-XXXX.run.app \
  --redis-url rediss://default:<password>@<upstash-host>:<port> \
  --google-client-id <your-oauth-client-id>.apps.googleusercontent.com
```

All three scripts print the next command / the deployed URL at the end — follow what
they say. `--redis-url` and `--google-client-id` are each independently optional —
`gcloud_deploy_app.sh`'s own header comment documents the fallback behavior when either
is omitted.

**Manual prerequisites (one-time, can't be scripted):**
- **Upstash Redis** (for `--redis-url`): create a free database at
  [upstash.com](https://upstash.com) (console), copy its `rediss://` connection URL.
  Chosen over GCP Memorystore because Memorystore needs a VPC Serverless Access
  connector and costs $35-50+/month even at the smallest tier — incompatible with a
  R$50/month budget. Zero backend code change needed: `backend/redis_client.py` already
  auto-detects TLS from the `rediss://` scheme.
- **Google OAuth Client ID** (for `--google-client-id`): Google Cloud Console → APIs &
  Services → Credentials → Create Credentials → OAuth Client ID → "Web application",
  Authorized JavaScript origins = the frontend's Cloud Run URL (get it after the FIRST
  `gcloud_deploy_app.sh` run with no `--google-client-id`, then re-run with it once you
  have the Client ID — or pre-guess the URL, Cloud Run URLs are deterministic per
  service+project+region). No client secret needed — ID-token verification only needs
  the Client ID as the expected audience.

## Why the base model is baked into the image

`vlm_service/Dockerfile` downloads the ~17.5GB Qwen3-VL-8B base model at BUILD time
(`HF_HUB_OFFLINE=1` at runtime — no network call on cold start). Without this, a
scale-to-zero (`min=0`) cold start would download 17.5GB from Hugging Face on every cold
start — slow (minutes) and prone to timing out or hitting HF rate limits. The trade-off:
the image is ~30GB and the build/push is slow. `vlm_service/cloudbuild.yaml` has the
timeout (3600s) and disk (100GB) raised to match.

## Cost safety — three independent layers, don't rely on just one

1. **GCP Billing Budget + alert (the real $ backstop).** This is the ONLY thing here that
   actually looks at money — everything else caps request *volume*, not spend.

   ```bash
   gcloud services enable billingbudgets.googleapis.com --project <PROJECT>
   gcloud billing budgets create \
     --billing-account=<BILLING_ACCOUNT_ID> \
     --display-name="<project> monthly cap" \
     --budget-amount=<AMOUNT><CURRENCY>  \    # e.g. 50.00BRL — no space, currency code suffix
     --filter-projects=projects/<PROJECT_NUMBER> \
     --threshold-rule=percent=0.5 --threshold-rule=percent=0.9 --threshold-rule=percent=1.0
   ```

   Get `<PROJECT_NUMBER>` from `gcloud projects describe <PROJECT> --format="value(projectNumber)"`
   and `<BILLING_ACCOUNT_ID>` from `gcloud billing accounts list`. Alerts email the
   billing account's admins/users by default (no Pub/Sub topic needed for a simple
   email alert). **This does NOT automatically stop spending** — it's a notification.
   If the user wants an automatic hard-stop, that needs a Pub/Sub-triggered Cloud
   Function that disables the service/billing at a threshold — treat that as a separate,
   explicitly-requested, higher-risk task (it can also kill the whole demo/project);
   don't build it unprompted.
2. **`VLM_DAILY_BUDGET` (app-level, request COUNT — not dollars).** Set via
   `gcloud_deploy_app.sh`'s `BACKEND_ENV`. Once this many real VLM answers happen in a
   UTC day, `/api/ask` returns 429 *before* touching the GPU. Cheap safety net, but it
   caps volume, not spend — don't confuse the number with a currency amount (a past
   session mistake: `VLM_DAILY_BUDGET=200` was misread as "$200/day").
3. **Image storage cleanup (`scripts/_gcloud_common.sh`'s `cleanup_old_images`).** Every
   deploy script always builds under the same `:latest` tag, so a re-deploy doesn't
   remove the OLD digest from Artifact Registry — it becomes untagged/dangling and still
   costs storage forever. All three deploy scripts call `cleanup_old_images`
   automatically AFTER each `gcloud run deploy` succeeds (safe ordering: the new
   revision is live before the old image is removed). This matters most for
   `vlm-service` — its image is ~30GB, so a leftover old digest is real, ongoing cost
   with zero usage. Don't skip this when writing new deploy automation for this project.
4. **Required Google login (abuse prevention, not a $ control per se).** When
   `--google-client-id` is set, `/api/ask` and `/api/vlm/warm` reject any request
   without a valid Google ID token — a much bigger lever against automated abuse than
   per-IP rate-limiting, since Google accounts aren't free to mint at scale. Combined
   with `--redis-url`, the rate-limit/budget keys switch from per-IP to per-Google-`sub`
   (stable identity, works across an attacker's IP rotation).

## Auth model (why the GPU and guard are private)

- GPU service (`chartqa-vlm`) and guard service (`chartqa-guard`): both deployed with
  `--no-allow-unauthenticated`.
- Backend service account (`chartqa-backend@<PROJECT>.iam.gserviceaccount.com`, created
  by `gcloud_deploy_app.sh`) is granted `roles/run.invoker` on whichever of the two is
  actually wired up this deploy (via `--vlm-url` / `--guard-url`).
- Backend is deployed `--service-account <that SA>`, so the Cloud Run metadata server
  mints ID tokens for that identity. `backend/gcp_auth.py`'s `fetch_id_token()` fetches
  one (audience = the target service's own URL) — used by both `model_adapter.py` (VLM
  calls, `VLM_AUTH=gcp_id_token`) and `guard_llm.py` (guard calls,
  `GUARD_LLM_AUTH=gcp_id_token`) — and sends it as `Authorization: Bearer <token>`.
- `VLM_AUTH=none` / `GUARD_LLM_AUTH=none` (the default everywhere else — RunPod dev,
  docker-compose, a big-GPU dev box) skip all of this; only Cloud Run sets
  `gcp_id_token` for either.
- Frontend + backend themselves stay `--allow-unauthenticated` (it's a public demo) —
  the GPU and guard are locked down because they're the expensive/abusable parts. Human
  users authenticate at the frontend layer instead, via required Google login (below).

## Required Google login

When `gcloud_deploy_app.sh` is run with `--google-client-id`, the backend sets
`AUTH_ENABLED=1` and the frontend build bakes in `VITE_GOOGLE_CLIENT_ID` (via
`frontend/cloudbuild.yaml` — Vite's `import.meta.env.VITE_*` is compile-time, not
runtime, unlike the nginx template's `PORT`/`BACKEND_URL`). With auth on:

- The frontend shows a "Sign in with Google" gate (Google Identity Services script tag,
  no npm dependency) before the question form renders.
- Every `/api/ask` and `/api/vlm/warm` call carries `Authorization: Bearer <google-id-token>`.
- The backend verifies the token's signature on **every request**
  (`backend/auth.py`, `google-auth`'s `verify_oauth2_token`, audience = the configured
  Client ID) — no session cookie, no server-side session store, no new DB table.
  Rejects with `401` on missing/expired/wrong-audience tokens.
- `/api/health` and `/metrics` stay open (needed for uptime/monitoring probes).
- The rate-limit key switches from client IP to the authenticated user's Google `sub`.

Omit `--google-client-id` to deploy with `AUTH_ENABLED=0` (no login wall) — matches the
local dev / docker-compose default, useful for a quick mock-mode demo deploy.

## Status / logs / cost checks

```bash
gcloud run services list --project <PROJECT> --region us-central1
gcloud run services describe chartqa-vlm --project <PROJECT> --region us-central1 \
  --format="value(status.url,status.conditions)"
gcloud run services logs read chartqa-vlm --project <PROJECT> --region us-central1 --limit 50
gcloud billing accounts list   # find the billing account id for cost dashboards
```

Cost/usage dashboards live in the console (Billing → Reports) — `gcloud` doesn't have a
clean CLI for "how much have I spent so far this month."

## Rollback / teardown

```bash
# Roll back to the previous Cloud Run revision (traffic split), without a new build:
gcloud run services update-traffic <SERVICE> --project <PROJECT> --region <REGION> \
  --to-revisions=<PREVIOUS_REVISION>=100

# Stop a service from costing anything (scale-to-zero already does this when idle, but
# to be sure nothing is min-instances>0):
gcloud run services update <SERVICE> --project <PROJECT> --region <REGION> --min-instances=0

# Full teardown (irreversible — confirm with the user first, this is destructive):
gcloud run services delete <SERVICE> --project <PROJECT> --region <REGION>
```

## Known gotchas (from the first two real deploy sessions, 2026-07-09/10)

- `gcloud billing budgets create` fails the first time with `SERVICE_DISABLED` if
  `billingbudgets.googleapis.com` was never enabled on the project — enable it and retry
  (propagates in seconds, not the usual API-enable delay).
- `gcloud billing accounts describe` does NOT show the account's currency in its default
  output — either check the console or just try `--budget-amount=<N><CURRENCY>` and read
  the error if the currency is wrong (gcloud reports the expected currency on mismatch).
- Empty-array bash expansion (`${ARR[@]}`) errors under `set -u` when the array is empty
  and never assigned — use `${ARR[@]+"${ARR[@]}"}` (see `gcloud_deploy_app.sh`'s
  `SA_ARGS` handling) for an optional `gcloud run deploy` flag block.
- **`.gcloudignore` is a separate file from `.gitignore`/`.dockerignore`, and its absence
  is a silent trap in BOTH directions.** `gcloud builds submit` falls back to
  `.gitignore` when no `.gcloudignore` exists for that build context. That fallback can
  silently *exclude* things that ARE actually needed (`vlm_service`'s repo-root build:
  `.gitignore`'s blanket `checkpoints/` rule excluded the force-added LoRA adapter,
  causing `COPY failed: file not found in build context`), or *include* things that
  should never be uploaded (`backend`/`frontend` subdir builds with no
  `.gcloudignore`: the entire `.venv/`/`node_modules/` got tarred up — 55k+ files, 6GB —
  because neither is in those directories' own `.gitignore` scope the same way). Every
  build context directory in this repo (root, `backend/`, `frontend/`) has its own
  `.gcloudignore` now, mirroring the matching `.dockerignore`. If you add a new Cloud
  Build context, add its `.gcloudignore` too — verify with
  `gcloud meta list-files-for-upload <dir>` before trusting a build.
- **`bitsandbytes`/`triton` need a real C compiler at runtime, even on "runtime" (non-
  devel) CUDA base images.** `chartqa-vlm` crashed on every cold start with
  `RuntimeError: Failed to find C compiler` inside triton's CUDA driver JIT-compilation
  step — confirmed via `gcloud run services logs read chartqa-vlm`, not guessed (the
  earlier hypothesis was a cold-start timeout; the traceback showed a 100%-reproducible
  import-time crash instead). Fixed by installing `build-essential` in
  `vlm_service/Dockerfile` before the pip installs. If a similar quantized-model image
  crashes only on GPU/Cloud Run (not locally, where `devel` images are more common),
  suspect a missing compiler first and check the logs.
- **A GPU Cloud Run deploy without `--no-gpu-zonal-redundancy` prompts interactively**
  ("deploy with no zonal redundancy instead? Y/n") on a project with no zonal-redundancy
  quota — this hangs a non-interactive/scripted run. `gcloud_deploy_vlm.sh` passes the
  flag explicitly; do the same for any new GPU service script.
- **Ollama's base image (`ollama/ollama:latest`) hardcodes its bind address to
  `127.0.0.1:11434` / `0.0.0.0:11434` via its own default, NOT Cloud Run's injected
  `$PORT`.** Cloud Run requires the container to listen on exactly the `$PORT` it
  injects (usually 8080) or the revision never becomes ready. `guard/Dockerfile`
  overrides the base image's `ENTRYPOINT`/`CMD` with
  `CMD ["/bin/sh", "-c", "OLLAMA_HOST=0.0.0.0:${PORT:-11434} exec ollama serve"]` —
  falls back to 11434 when `$PORT` is unset (local docker-compose), binds to Cloud Run's
  port otherwise. Any other Ollama-based service deployed to Cloud Run needs the same
  wrapper.
- Large source uploads to GCS (`gcloud builds submit`'s tarball step, e.g. the ~180MB
  `vlm_service` repo-root context) occasionally hit a transient `ReadTimeoutError` —
  observed as plain network flakiness, not a code bug; retrying the same command
  succeeded. Don't over-diagnose a single timeout on a large upload before retrying once.
