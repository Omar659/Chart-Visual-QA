#!/usr/bin/env bash
# Deploys vlm_service/ (Qwen3-VL-8B + LoRA) to Cloud Run GPU — the production serving
# path. SAME Docker image RunPod dev pods run (vlm_service/Dockerfile); this script only
# handles the Cloud Run side: build via Cloud Build, push, deploy with GPU + scale-to-zero.
#
# Usage:
#   ./scripts/gcloud_deploy_vlm.sh [--project PROJECT] [--region REGION]
#
# Prerequisites:
#   - gcloud CLI installed and authenticated (`gcloud auth login`).
#   - A GCP project with billing enabled AND Cloud Run GPU quota for the target region
#     (request via console if `gcloud run deploy` errors with a quota message —
#     GPU-enabled regions include us-central1, europe-west1, asia-southeast1 at time of
#     writing; check current availability: `gcloud run regions list`).
#
# NOTE on auth: deployed with --allow-unauthenticated so it matches the CURRENT
# unauthenticated `backend/model_adapter._predict_remote` (a plain POST, no ID token).
# That's a known gap for a "production" deploy — see docs/REVIEW_AND_ROADMAP.md's
# security checklist. Locking this down needs model_adapter.py to attach a Google-signed
# identity token, which isn't wired up yet; flagged rather than half-implemented.
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
REPO="chartqa"
SERVICE="chartqa-vlm"
ADAPTER_DIR="qwen3vl-lora-final2"

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

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/vlm-service:latest"
echo "[deploy-vlm] project=$PROJECT region=$REGION image=$IMAGE"

echo "[deploy-vlm] enabling required APIs..."
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  cloudbuild.googleapis.com --project "$PROJECT" --quiet

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1; then
  echo "[deploy-vlm] creating Artifact Registry repo $REPO..."
  gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location "$REGION" --project "$PROJECT" \
    --description "Chart-Visual-QA images"
fi

echo "[deploy-vlm] building + pushing image via Cloud Build (repo root context; the base"
echo "  CUDA/torch image is large, this can take 10-20 minutes)..."
gcloud builds submit . \
  --project "$PROJECT" \
  --config vlm_service/cloudbuild.yaml \
  --substitutions="_IMAGE=${IMAGE}"

echo "[deploy-vlm] deploying to Cloud Run GPU (min-instances=0, scale-to-zero)..."
gcloud run deploy "$SERVICE" \
  --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" \
  --gpu=1 --gpu-type=nvidia-l4 --cpu=4 --memory=16Gi \
  --no-cpu-throttling \
  --min-instances=0 --max-instances=1 --concurrency=1 \
  --timeout=300 \
  --set-env-vars="QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct,QWEN_ADAPTER_PATH=/app/modeling/checkpoints/${ADAPTER_DIR},QWEN_QUANTIZATION=4bit,QWEN_MAX_NEW_TOKENS=64,QWEN_ANSWER_SUFFIX= Please answer directly." \
  --allow-unauthenticated

URL=$(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --format='value(status.url)')
echo "[deploy-vlm] deployed: ${URL}"
echo "[deploy-vlm] set in the backend's prod env: VLM_URL=${URL}/predict  VLM_PROVIDER=cloudrun"
