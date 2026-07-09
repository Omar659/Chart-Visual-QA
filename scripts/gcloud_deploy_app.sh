#!/usr/bin/env bash
# Deploys the webapp (backend + frontend) to Cloud Run — CPU only, no GPU. Separate
# from scripts/gcloud_deploy_vlm.sh (the GPU model service) on purpose: the app changes
# far more often than the model, and redeploying it should never require rebuilding the
# multi-GB CUDA image. Wire the two together with --vlm-url once the model is deployed.
#
# Usage:
#   ./scripts/gcloud_deploy_app.sh [--project PROJECT] [--region REGION] [--vlm-url URL]
#
#   --vlm-url URL   Point the backend at a deployed vlm_service (its Cloud Run URL +
#                   /predict, e.g. https://chartqa-vlm-xxxx.run.app/predict — see
#                   gcloud_deploy_vlm.sh's own output). Omit to deploy in USE_MOCK=1
#                   instead (a self-contained, testable app deploy with no GPU cost).
#                   When given, this script creates a backend service account, grants it
#                   run.invoker on the PRIVATE GPU service (chartqa-vlm), and sets
#                   VLM_AUTH=gcp_id_token so the backend authenticates its GPU calls.
#                   → Run gcloud_deploy_vlm.sh FIRST so chartqa-vlm exists.
#
# The frontend and backend are public (--allow-unauthenticated) — a public demo — but the
# GPU is NOT (only this backend's SA can invoke it). The backend's own rate-limit + daily
# VLM budget (3.6/3.7) cap cost on the public path.
#
# Prerequisites: gcloud CLI installed and authenticated (`gcloud auth login`).
#
# NOTE: the input-guard's Layer 3 (Llama Guard) isn't deployed by this script — it's
# disabled here (GUARD_LLM_ENABLED=0, fails open per the existing design) rather than
# half-wired to a guard service that doesn't exist yet. Layer 1/2 (rules + toxicity/
# injection/PII encoders, baked into the backend image) still run.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
REPO="chartqa"
BACKEND_SERVICE="chartqa-backend"
FRONTEND_SERVICE="chartqa-frontend"
VLM_SERVICE="chartqa-vlm"                 # the private GPU service (gcloud_deploy_vlm.sh)
SA_NAME="chartqa-backend"                 # backend's own service account (calls the GPU)
VLM_URL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --vlm-url) VLM_URL="$2"; shift 2 ;;
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
if [[ -n "$VLM_URL" ]]; then
  USE_MOCK=0
  VLM_PROVIDER=cloudrun
  VLM_AUTH=gcp_id_token
  echo "[deploy-app] real inference: VLM_URL=$VLM_URL (authenticated call to the private GPU)"

  # The backend calls the PRIVATE GPU service with a Google ID token, so it needs its own
  # service account that is granted run.invoker on that service. Create it (idempotent)
  # and bind the role. Deploying with --service-account makes the metadata server mint
  # ID tokens for THIS identity (model_adapter._vlm_auth_header).
  SA_EMAIL="${SA_NAME}@${PROJECT}.iam.gserviceaccount.com"
  if ! gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
    echo "[deploy-app] creating backend service account $SA_EMAIL..."
    gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" \
      --display-name "Chart-Visual-QA backend (invokes the GPU VLM service)"
  fi
  echo "[deploy-app] granting $SA_EMAIL run.invoker on $VLM_SERVICE..."
  gcloud run services add-iam-policy-binding "$VLM_SERVICE" \
    --project "$PROJECT" --region "$REGION" \
    --member "serviceAccount:$SA_EMAIL" --role roles/run.invoker --quiet
  SA_ARGS=(--service-account "$SA_EMAIL")
else
  USE_MOCK=1
  VLM_PROVIDER=none
  VLM_AUTH=none
  echo "[deploy-app] no --vlm-url given: deploying in USE_MOCK=1 (mock answers)."
fi

BACKEND_ENV="USE_MOCK=${USE_MOCK},MOCK_DELAY_S=0,MOCK_REVEAL=0"
BACKEND_ENV+=",VLM_URL=${VLM_URL},VLM_TIMEOUT=120,VLM_PROVIDER=${VLM_PROVIDER},VLM_AUTH=${VLM_AUTH}"
BACKEND_ENV+=",QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct,QWEN_ADAPTER_PATH=,QWEN_QUANTIZATION=none"
BACKEND_ENV+=",QWEN_MAX_NEW_TOKENS=64,QWEN_ANSWER_SUFFIX= Please answer directly."
BACKEND_ENV+=",HOST=0.0.0.0,FLASK_DEBUG=0,CORS_ORIGINS=*,MAX_UPLOAD_MB=10,MIN_QUESTION_ALNUM=3"
BACKEND_ENV+=",ANSWER_CACHE_ENABLED=1,ANSWER_CACHE_MAX=512,ANSWER_CACHE_TTL_S=3600"
# Data layer + cost controls (3.6/3.7). REDIS_URL empty = in-memory fallbacks (Cloud Run
# has no Redis wired here yet — a Memorystore URL would go here). The daily VLM budget is
# the real GPU-cost backstop for a public demo; tune before going live.
BACKEND_ENV+=",REDIS_URL=,RATELIMIT_ENABLED=1,RATELIMIT_PER_MINUTE=30,VLM_DAILY_BUDGET=200"
BACKEND_ENV+=",GUARD_ENABLED=1,GUARD_TOXICITY_THRESHOLD=0.7,GUARD_INJECTION_THRESHOLD=0.8,GUARD_PII_THRESHOLD=0.6"
BACKEND_ENV+=",GUARD_TOXICITY_MODEL=original,GUARD_INJECTION_MODEL=protectai/deberta-v3-base-prompt-injection-v2"
BACKEND_ENV+=",GUARD_LLM_ENABLED=0,GUARD_LLM_URL=http://localhost:11434,GUARD_LLM_MODEL=llama-guard3:1b,GUARD_LLM_TIMEOUT=20"
BACKEND_ENV+=",CHART_CLIP_MODEL=openai/clip-vit-base-patch32,CHART_CLIP_THRESHOLD=0.5,CHART_MIN_DATA_DIGITS=2"
BACKEND_ENV+=",CHART_SAMPLE_SIZE=128,CHART_MIN_BACKGROUND_RATIO=0.18,CHART_MAX_DISTINCT_COLORS=48,TESSERACT_CMD="

echo "[deploy-app] deploying backend to Cloud Run..."
gcloud run deploy "$BACKEND_SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$BACKEND_IMAGE" \
  --cpu=1 --memory=2Gi \
  --min-instances=0 --max-instances=3 \
  --timeout=300 \
  --set-env-vars="$BACKEND_ENV" \
  ${SA_ARGS[@]+"${SA_ARGS[@]}"} \
  --allow-unauthenticated

BACKEND_URL=$(gcloud run services describe "$BACKEND_SERVICE" --project "$PROJECT" \
  --region "$REGION" --format='value(status.url)')
echo "[deploy-app] backend deployed: $BACKEND_URL"

echo "[deploy-app] building + pushing frontend image..."
gcloud builds submit frontend --project "$PROJECT" --tag "$FRONTEND_IMAGE"

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
echo "[deploy-app] done. Open $FRONTEND_URL"
