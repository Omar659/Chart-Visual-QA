#!/usr/bin/env bash
# Applies scripts/registry_cleanup_policy.json as the Artifact Registry cleanup policy of
# the `chartqa` Docker repo, so old image versions are deleted server-side on a schedule
# instead of relying only on cleanup_old_images (which runs only when someone deploys).
#
# Policy (Keep rules always win over Delete rules):
#   - delete-untagged-older-than-1d: every deploy re-pushes :latest, so the previous
#     digest becomes untagged — delete it once it is more than a day old.
#   - keep-most-recent-1: never delete the newest version of any package, even untagged.
# Net effect: each package (backend, frontend, guard, vlm-base, vlm-service, …) keeps only
# its current :latest. Artifact Registry dedupes layers by digest, so deleting an old
# vlm-service never removes layers vlm-base (or the live vlm-service) still references.
#
# Trade-off: no old digest survives past ~1 day, so a Cloud Run rollback to an older
# revision only works within that window — after that, roll forward with a rebuild.
#
# DRY-RUN BY DEFAULT: the policy is installed but only logs what it would delete (Cloud
# Audit Logs, ~daily, asynchronous). Pass --apply to enforce it for real.
#
# Usage:
#   ./scripts/gcloud_registry_cleanup.sh [--project PROJECT] [--region REGION] [--apply]
#
# Prerequisites: gcloud CLI installed and authenticated (`gcloud auth login`).
set -euo pipefail
cd "$(dirname "$0")/.."

PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${GCP_REGION:-us-central1}"
REPO="chartqa"
POLICY_FILE="scripts/registry_cleanup_policy.json"
DRY_RUN_FLAG="--dry-run"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project) PROJECT="$2"; shift 2 ;;
    --region) REGION="$2"; shift 2 ;;
    --apply) DRY_RUN_FLAG="--no-dry-run"; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "$PROJECT" ]]; then
  echo "No GCP project set. Pass --project <id> or: gcloud config set project <id>" >&2
  exit 1
fi

echo "[registry-cleanup] project=$PROJECT region=$REGION repo=$REPO mode=${DRY_RUN_FLAG#--}"
gcloud artifacts repositories set-cleanup-policies "$REPO" \
  --project "$PROJECT" --location "$REGION" \
  --policy "$POLICY_FILE" "$DRY_RUN_FLAG"

gcloud artifacts repositories describe "$REPO" --project "$PROJECT" --location "$REGION" \
  --format="yaml(cleanupPolicies,cleanupPolicyDryRun)"

if [[ "$DRY_RUN_FLAG" == "--dry-run" ]]; then
  echo "[registry-cleanup] dry-run only — nothing is deleted. Re-run with --apply to enforce."
fi
