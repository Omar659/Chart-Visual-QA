#!/usr/bin/env bash
# Builds + pushes the vlm-base image (vlm_service/Dockerfile.base) to Artifact Registry.
# This is the EXPENSIVE half of the VLM image: CUDA/torch deps + the baked ~17.5 GB base
# model (~30 GB image, ~20-29 min build). See docs/VLM_IMAGE_SPLIT.md.
#
# RUN THIS RARELY — only when one of these changes:
#   - the CUDA/torch base image tag,
#   - a pip dependency (transformers/peft/accelerate/bitsandbytes or the torch pin),
#   - the base model (QWEN_MODEL_ID).
# A normal code/adapter/config deploy does NOT need this — it just reruns
# scripts/gcloud_deploy_vlm.sh, which builds the thin `FROM vlm-base` image in minutes.
#
# Usage:
#   ./scripts/gcloud_build_vlm_base.sh [--project PROJECT] [--region REGION]
#
# Prerequisites: gcloud CLI installed and authenticated (`gcloud auth login`).
set -euo pipefail
cd "$(dirname "$0")/.."
source "$(dirname "$0")/_gcloud_common.sh"

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
REPO="chartqa"

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

IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/vlm-base:latest"
echo "[build-vlm-base] project=$PROJECT region=$REGION image=$IMAGE"

echo "[build-vlm-base] enabling required APIs..."
gcloud services enable artifactregistry.googleapis.com cloudbuild.googleapis.com \
  --project "$PROJECT" --quiet

if ! gcloud artifacts repositories describe "$REPO" --location "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1; then
  echo "[build-vlm-base] creating Artifact Registry repo $REPO..."
  gcloud artifacts repositories create "$REPO" --repository-format=docker \
    --location "$REGION" --project "$PROJECT" \
    --description "Chart-Visual-QA images"
fi

echo "[build-vlm-base] building + pushing base image via Cloud Build (bakes the ~17.5 GB"
echo "  base model + CUDA/torch deps — this can take 20-30 minutes; run rarely)..."
gcloud builds submit . \
  --project "$PROJECT" \
  --config vlm_service/cloudbuild.base.yaml \
  --substitutions="_IMAGE=${IMAGE}"

echo "[build-vlm-base] pushed: ${IMAGE}"

# vlm-base is ~30 GB per digest; a leftover old digest from a previous base build is real,
# ongoing storage cost. Clean up the untagged ones now that the new :latest is pushed.
cleanup_old_images "${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/vlm-base"

echo "[build-vlm-base] next: ./scripts/gcloud_deploy_vlm.sh --project ${PROJECT} --region ${REGION}"
echo "[build-vlm-base]   (the thin service deploy now builds against this base in minutes.)"
