---
name: gcloud-deploy
description: Deploy Chart-Visual-QA to Google Cloud Run (CPU app + GPU model), and the operational commands around it — cost safety (billing budget, image cleanup, daily VLM count cap), auth between the CPU backend and the private GPU service, status/logs checks, and teardown/rollback. Use this whenever the user asks to deploy to Cloud Run, check cloud costs/status, or roll back a Cloud Run deploy.
---

# Cloud Run deploy — Chart-Visual-QA

Two independent deploys, two scripts, deployed in this order:

1. **`scripts/gcloud_deploy_vlm.sh`** — the GPU model (`vlm_service/`), Qwen3-VL-8B +
   LoRA, on Cloud Run GPU (`nvidia-l4`, scale-to-zero). **Deployed PRIVATE** (no
   `--allow-unauthenticated`) — the expensive endpoint must never be callable from the
   public internet.
2. **`scripts/gcloud_deploy_app.sh --vlm-url <step-1-URL>/predict`** — the CPU app
   (backend + frontend), public. Creates the backend's own service account, grants it
   `run.invoker` on the GPU service, and wires `VLM_AUTH=gcp_id_token` so the backend can
   actually call the private GPU (`backend/model_adapter._vlm_auth_header` fetches a
   Google-signed ID token from the Cloud Run metadata server — no extra dependency).

Read both scripts' own header comments before running them — they're the source of
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

# 2. CPU app, pointed at the GPU service's printed URL.
./scripts/gcloud_deploy_app.sh --project <PROJECT> --region us-central1 \
  --vlm-url https://chartqa-vlm-XXXX.run.app/predict
```

Both scripts print the next command / the deployed URL at the end — follow what they say.

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
   costs storage forever. Both deploy scripts call `cleanup_old_images` automatically
   AFTER each `gcloud run deploy` succeeds (safe ordering: the new revision is live
   before the old image is removed). This matters most for `vlm-service` — its image is
   ~30GB, so a leftover old digest is real, ongoing cost with zero usage. Don't skip
   this when writing new deploy automation for this project.

## Auth model (why the GPU is private)

- GPU service (`chartqa-vlm`): deployed with `--no-allow-unauthenticated`.
- Backend service account (`chartqa-backend@<PROJECT>.iam.gserviceaccount.com`, created
  by `gcloud_deploy_app.sh`) is granted `roles/run.invoker` on `chartqa-vlm` only.
- Backend is deployed `--service-account <that SA>`, so the Cloud Run metadata server
  mints ID tokens for that identity. `model_adapter._vlm_auth_header()` fetches one
  (audience = the GPU service's own URL) and sends it as `Authorization: Bearer <token>`.
- `VLM_AUTH=none` (the default everywhere else — RunPod dev, docker-compose, a big-GPU
  dev box) skips all of this; only Cloud Run sets `VLM_AUTH=gcp_id_token`.
- Frontend + backend themselves stay `--allow-unauthenticated` (it's a public demo) —
  only the GPU is locked down, because that's the expensive part.

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

## Known gotchas (from the first real deploy prep session, 2026-07-09)

- `gcloud billing budgets create` fails the first time with `SERVICE_DISABLED` if
  `billingbudgets.googleapis.com` was never enabled on the project — enable it and retry
  (propagates in seconds, not the usual API-enable delay).
- `gcloud billing accounts describe` does NOT show the account's currency in its default
  output — either check the console or just try `--budget-amount=<N><CURRENCY>` and read
  the error if the currency is wrong (gcloud reports the expected currency on mismatch).
- Empty-array bash expansion (`${ARR[@]}`) errors under `set -u` when the array is empty
  and never assigned — use `${ARR[@]+"${ARR[@]}"}` (see `gcloud_deploy_app.sh`'s
  `SA_ARGS` handling) for an optional `gcloud run deploy` flag block.
