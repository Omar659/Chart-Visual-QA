#!/usr/bin/env bash
# Deploys guard/ (Llama Guard 3 1B via Ollama) to Cloud Run — Guard Layer 3, CPU-only.
# SAME image `docker compose up guard` runs locally (guard/Dockerfile); this script only
# handles the Cloud Run side. Much smaller/simpler than gcloud_deploy_vlm.sh: no GPU, no
# LoRA adapter, small baked model — a plain `gcloud builds submit --tag` is enough, no
# custom cloudbuild.yaml needed.
#
# Usage:
#   ./scripts/gcloud_deploy_guard.sh [--project PROJECT] [--region REGION]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (`gcloud auth login`).
#
# AUTH: deployed PRIVATE (no --allow-unauthenticated), same reasoning as the GPU service —
# an unauthenticated classifier endpoint is still a free compute sink for anyone who finds
# the URL. Only the backend calls it, with a Google-signed ID token
# (backend/guard_llm.py, GUARD_LLM_AUTH=gcp_id_token); its service account is granted
# run.invoker on this service by scripts/gcloud_deploy_app.sh. Deploy this AFTER
# gcloud_deploy_vlm.sh and BEFORE gcloud_deploy_app.sh --guard-url <this service's URL>.
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/_gcloud_common.sh"

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
REPO="chartqa"
SERVICE="chartqa-guard"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "No GCP project set. Pass --project <id> or: gcloud config set project <id>" >&2
  exit 1
fi

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/guard:latest"
echo "[deploy-guard] project=$PROJECT region=$REGION image=$IMAGE"

echo "[deploy-guard] enabling required APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com --project "$PROJECT" --quiet

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1; then
  echo "[deploy-guard] creating Artifact Registry repo $REPO..."
  gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location "$REGION" --project "$PROJECT" \
    --description "Chart-Visual-QA images"
fi

echo "[deploy-guard] building + pushing image via Cloud Build (bakes llama-guard3:1b,"
echo "  a few GB — much smaller than the VLM image, but still not instant)..."
gcloud builds submit guard --project "$PROJECT" --tag "$IMAGE"

echo "[deploy-guard] deploying to Cloud Run (min-instances=0, scale-to-zero, PRIVATE)..."
# --no-allow-unauthenticated: private, same pattern as chartqa-vlm. CPU-only (no --gpu
# flags), 2Gi is plenty for a 1B model. guard/Dockerfile's CMD honors $PORT (Cloud Run
# injects it, usually 8080) instead of Ollama's hardcoded default :11434 — see the
# Dockerfile comment for why that override was needed.
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" \
  --cpu=2 --memory=2Gi \
  --min-instances=0 --max-instances=2 --concurrency=4 \
  --timeout=60 \
  --no-allow-unauthenticated

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)')
echo "[deploy-guard] deployed (private): ${URL}"

cleanup_old_images "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/guard"

echo "[deploy-guard] next: ./scripts/gcloud_deploy_app.sh --project ${PROJECT} --region ${REGION} \\"
echo "                       --vlm-url <vlm-url>/predict --guard-url ${URL}"
echo "[deploy-guard]   (that script grants the backend service account run.invoker on"
echo "                  '${SERVICE}' too, and sets GUARD_LLM_ENABLED=1, GUARD_LLM_AUTH=gcp_id_token.)"
