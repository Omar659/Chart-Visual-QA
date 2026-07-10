#!/usr/bin/env bash
# Deploys the webapp (backend + frontend) to Cloud Run — CPU only, no GPU. Separate
# from scripts/gcloud_deploy_vlm.sh (the GPU model service) on purpose: the app changes
# far more often than the model, and redeploying it should never require rebuilding the
# multi-GB CUDA image. Wire the two together with --vlm-url once the model is deployed.
#
# Usage:
#   ./scripts/gcloud_deploy_app.sh [--project PROJECT] [--region REGION] [--vlm-url URL]
#                                  [--guard-url URL] [--redis-url URL]
#                                  [--google-client-id ID]
#
#   --vlm-url URL          Point the backend at a deployed vlm_service (its Cloud Run URL
#                           + /predict, e.g. https://chartqa-vlm-xxxx.run.app/predict —
#                           see gcloud_deploy_vlm.sh's own output). Omit to deploy in
#                           USE_MOCK=1 instead (a self-contained, testable app deploy with
#                           no GPU cost). When given, this script creates a backend
#                           service account, grants it run.invoker on the PRIVATE GPU
#                           service (chartqa-vlm), and sets VLM_AUTH=gcp_id_token.
#                           → Run gcloud_deploy_vlm.sh FIRST so chartqa-vlm exists.
#   --guard-url URL         Point the backend at a deployed guard/ service (its Cloud Run
#                           URL, e.g. https://chartqa-guard-xxxx.run.app — see
#                           gcloud_deploy_guard.sh's own output; no path suffix, same
#                           shape as GUARD_LLM_URL=http://guard:11434 locally). When
#                           given, grants the SAME backend service account run.invoker on
#                           the PRIVATE guard service too, and sets GUARD_LLM_ENABLED=1,
#                           GUARD_LLM_AUTH=gcp_id_token. Omit to keep Guard Layer 3 off
#                           (fails open; Layers 1/2 still run).
#                           → Run gcloud_deploy_guard.sh first so chartqa-guard exists.
#   --redis-url URL         A rediss:// URL (e.g. from Upstash) for GLOBAL rate-limit and
#                           daily-VLM-budget counters shared across backend instances.
#                           Omit to keep the in-memory per-instance fallback (fine for a
#                           single instance; under-counts the true global rate once
#                           --max-instances allows more than one).
#   --google-client-id ID   Google OAuth 2.0 Web Client ID (from Google Cloud Console →
#                           Credentials). When given, sets AUTH_ENABLED=1 + the backend's
#                           GOOGLE_CLIENT_ID, and builds the frontend with
#                           VITE_GOOGLE_CLIENT_ID baked in (via frontend/cloudbuild.yaml)
#                           so it renders the Google sign-in gate. Omit to deploy with no
#                           login wall (AUTH_ENABLED=0, matches local dev default).
#
# The frontend and backend are public (--allow-unauthenticated) — a public demo — but the
# GPU and guard services are NOT (only this backend's SA can invoke them). The backend's
# own rate-limit + daily VLM budget (3.6/3.7), and required Google login (3.7) when
# --google-client-id is given, cap cost/abuse on the public path.
#
# Prerequisites: gcloud CLI installed and authenticated (`gcloud auth login`).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/_gcloud_common.sh"

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
REPO="chartqa"
BACKEND_SERVICE="chartqa-backend"
FRONTEND_SERVICE="chartqa-frontend"
VLM_SERVICE="chartqa-vlm"                 # the private GPU service (gcloud_deploy_vlm.sh)
GUARD_SERVICE="chartqa-guard"             # the private guard service (gcloud_deploy_guard.sh)
SA_NAME="chartqa-backend"                 # backend's own service account (calls GPU + guard)
VLM_URL=""
GUARD_URL=""
REDIS_URL=""
GOOGLE_CLIENT_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --vlm-url) VLM_URL="$2"; shift 2 ;;
    --guard-url) GUARD_URL="$2"; shift 2 ;;
    --redis-url) REDIS_URL="$2"; shift 2 ;;
    --google-client-id) GOOGLE_CLIENT_ID="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "No GCP project set. Pass --project <id> or: gcloud config set project <id>" >&2
  exit 1
fi

BACKEND_IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/backend:latest"
FRONTEND_IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/frontend:latest"
echo "[deploy-app] project=$PROJECT region=$REGION"

echo "[deploy-app] enabling required APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com iam.googleapis.com --project "$PROJECT" --quiet

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1; then
  echo "[deploy-app] creating Artifact Registry repo $REPO..."
  gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location "$REGION" --project "$PROJECT" \
    --description "Chart-Visual-QA images"
fi

echo "[deploy-app] building + pushing backend image..."
gcloud builds submit backend --project "$PROJECT" --tag "$BACKEND_IMAGE"

SA_ARGS=()   # extra `gcloud run deploy` args for the backend (service account, when real)
NEEDS_SA=0
[[ -n "$VLM_URL" || -n "$GUARD_URL" ]] && NEEDS_SA=1

if [[ "$NEEDS_SA" -eq 1 ]]; then
  # The backend calls PRIVATE services (GPU, guard) with a Google ID token, so it needs
  # its own service account. Create it once (idempotent) and grant run.invoker on
  # whichever of the two private services is actually being wired up this run. Deploying
  # with --service-account makes the metadata server mint ID tokens for THIS identity
  # (backend/gcp_auth.py).
  SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
    echo "[deploy-app] creating backend service account $SA_EMAIL..."
    gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" \
      --display-name "Chart-Visual-QA backend (invokes the GPU VLM + guard services)"
  fi
  SA_ARGS=(--service-account "$SA_EMAIL")
fi

if [[ -n "$VLM_URL" ]]; then
  USE_MOCK=0
  VLM_PROVIDER=cloudrun
  VLM_AUTH=gcp_id_token
  echo "[deploy-app] real inference: VLM_URL=$VLM_URL (authenticated call to the private GPU)"
  echo "[deploy-app] granting $SA_EMAIL run.invoker on $VLM_SERVICE..."
  gcloud run services add-iam-policy-binding "$VLM_SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --member "serviceAccount:$SA_EMAIL" --role roles/run.invoker --quiet
else
  USE_MOCK=1
  VLM_PROVIDER=none
  VLM_AUTH=none
  echo "[deploy-app] no --vlm-url given: deploying in USE_MOCK=1 (mock answers)."
fi

if [[ -n "$GUARD_URL" ]]; then
  GUARD_LLM_ENABLED=1
  GUARD_LLM_AUTH=gcp_id_token
  echo "[deploy-app] Guard Layer 3: GUARD_LLM_URL=$GUARD_URL (authenticated call to the private guard)"
  echo "[deploy-app] granting $SA_EMAIL run.invoker on $GUARD_SERVICE..."
  gcloud run services add-iam-policy-binding "$GUARD_SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --member "serviceAccount:$SA_EMAIL" --role roles/run.invoker --quiet
else
  GUARD_LLM_ENABLED=0
  GUARD_LLM_AUTH=none
  GUARD_URL="http://localhost:11434"   # unused when disabled; keeps env var non-empty
  echo "[deploy-app] no --guard-url given: Guard Layer 3 stays off (fails open; Layers 1/2 still run)."
fi

if [[ -n "$GOOGLE_CLIENT_ID" ]]; then
  AUTH_ENABLED=1
  echo "[deploy-app] required Google login: AUTH_ENABLED=1 (GOOGLE_CLIENT_ID=$GOOGLE_CLIENT_ID)"
else
  AUTH_ENABLED=0
  echo "[deploy-app] no --google-client-id given: deploying with no login wall (AUTH_ENABLED=0)."
fi

BACKEND_ENV="USE_MOCK=${USE_MOCK},MOCK_DELAY_S=0,MOCK_REVEAL=0"
BACKEND_ENV+=",VLM_URL=${VLM_URL},VLM_TIMEOUT=120,VLM_PROVIDER=${VLM_PROVIDER},VLM_AUTH=${VLM_AUTH}"
BACKEND_ENV+=",QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct,QWEN_ADAPTER_PATH=,QWEN_QUANTIZATION=none"
BACKEND_ENV+=",QWEN_MAX_NEW_TOKENS=64,QWEN_ANSWER_SUFFIX= Please answer directly."
BACKEND_ENV+=",HOST=0.0.0.0,FLASK_DEBUG=0,CORS_ORIGINS=*,MAX_UPLOAD_MB=10,MIN_QUESTION_ALNUM=3"
BACKEND_ENV+=",ANSWER_CACHE_ENABLED=1,ANSWER_CACHE_MAX=512,ANSWER_CACHE_TTL_S=3600"
# Data layer + cost controls (3.6/3.7). REDIS_URL empty = in-memory per-instance fallback
# (fine for a single instance; under-counts the true global rate once --max-instances
# allows more than one — pass --redis-url, e.g. an Upstash rediss:// URL, to fix that).
# VLM_DAILY_BUDGET is a COUNT of real VLM answers per day, NOT a dollar amount — it's an
# app-level soft cap, not a spend cap. The actual $ backstop for this project is a GCP
# Billing Budget + alert (set one up for your project; see docs/REVIEW_AND_ROADMAP.md
# §3.7). 20/day is plenty for a portfolio demo; raise it if you expect real traffic.
BACKEND_ENV+=",REDIS_URL=${REDIS_URL},RATELIMIT_ENABLED=1,RATELIMIT_PER_MINUTE=30,VLM_DAILY_BUDGET=20"
BACKEND_ENV+=",GUARD_ENABLED=1,GUARD_TOXICITY_THRESHOLD=0.7,GUARD_INJECTION_THRESHOLD=0.8,GUARD_PII_THRESHOLD=0.6"
BACKEND_ENV+=",GUARD_TOXICITY_MODEL=original,GUARD_INJECTION_MODEL=protectai/deberta-v3-base-prompt-injection-v2"
BACKEND_ENV+=",GUARD_LLM_ENABLED=${GUARD_LLM_ENABLED},GUARD_LLM_URL=${GUARD_URL},GUARD_LLM_AUTH=${GUARD_LLM_AUTH},GUARD_LLM_MODEL=llama-guard3:1b,GUARD_LLM_TIMEOUT=60"
BACKEND_ENV+=",CHART_CLIP_MODEL=openai/clip-vit-base-patch32,CHART_CLIP_THRESHOLD=0.5,CHART_MIN_DATA_DIGITS=2,CHART_BLOCK_THRESHOLD=0.4"
BACKEND_ENV+=",CHART_SAMPLE_SIZE=128,CHART_MIN_BACKGROUND_RATIO=0.18,CHART_MAX_DISTINCT_COLORS=48,TESSERACT_CMD="
# Required Google login (3.7). AUTH_ENABLED=0 (default) = no login wall, matches local
# dev. GOOGLE_CLIENT_ID is public (it's the OAuth audience, not a secret) — safe as a
# plain env var, same as everything else here.
BACKEND_ENV+=",AUTH_ENABLED=${AUTH_ENABLED},GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}"

echo "[deploy-app] deploying backend to Cloud Run..."
gcloud run deploy "$BACKEND_SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$BACKEND_IMAGE" \
  --cpu=1 --memory=4Gi \
  --min-instances=0 --max-instances=3 \
  --timeout=300 \
  --cpu-boost \
  --set-env-vars="$BACKEND_ENV" \
  ${SA_ARGS[@]+"${SA_ARGS[@]}"} \
  --allow-unauthenticated

BACKEND_URL=$(gcloud run services describe "$BACKEND_SERVICE" --project "$PROJECT" \
  --region "$REGION" --format='value(status.url)')
echo "[deploy-app] backend deployed: $BACKEND_URL"
cleanup_old_images "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/backend"

echo "[deploy-app] building + pushing frontend image (VITE_GOOGLE_CLIENT_ID baked in at"
echo "  build time, see frontend/cloudbuild.yaml)..."
gcloud builds submit frontend --project "$PROJECT" \
  --config frontend/cloudbuild.yaml \
  --substitutions="_IMAGE=${FRONTEND_IMAGE},_VITE_GOOGLE_CLIENT_ID=${GOOGLE_CLIENT_ID}"

echo "[deploy-app] deploying frontend to Cloud Run (proxies /api -> $BACKEND_URL)..."
gcloud run deploy "$FRONTEND_SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$FRONTEND_IMAGE" \
  --cpu=1 --memory=512Mi \
  --min-instances=0 --max-instances=3 \
  --set-env-vars="BACKEND_URL=${BACKEND_URL},DNS_RESOLVER=8.8.8.8" \
  --allow-unauthenticated

FRONTEND_URL=$(gcloud run services describe "$FRONTEND_SERVICE" --project "$PROJECT" \
  --region "$REGION" --format='value(status.url)')
echo "[deploy-app] frontend deployed: $FRONTEND_URL"
cleanup_old_images "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/frontend"

echo "[deploy-app] done. Open $FRONTEND_URL"
