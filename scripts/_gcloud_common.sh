#!/usr/bin/env bash
# Shared helpers for the gcloud deploy scripts (scripts/gcloud_deploy_vlm.sh /
# gcloud_deploy_app.sh). Sourced, not executed directly — one implementation instead of
# copy-pasting the same cleanup logic into both scripts.

# Every deploy script here always builds+pushes under the SAME tag (:latest). Pushing a
# new build doesn't remove the OLD digest that used to hold that tag — it becomes
# "untagged" (dangling) in Artifact Registry and keeps costing storage forever unless
# deleted. This matters a lot for vlm_service specifically: its image bakes the ~17.5GB
# base model, so each leftover old digest is ~30GB of pure waste, billed monthly whether
# or not the service is ever invoked (independent of Cloud Run's own scale-to-zero).
#
# Call this AFTER the new `gcloud run deploy` has succeeded (not right after the push) —
# that way the new revision is already confirmed live before we remove anything the
# previous revision might otherwise need to pull again.
cleanup_old_images() {
  local image_path="$1"   # e.g. REGION-docker.pkg.dev/PROJECT/REPO/NAME (no :tag)
  echo "[cleanup] removing untagged (dangling) images under ${image_path}..."
  local digests
  digests=$(gcloud artifacts docker images list "$image_path" \
    --include-tags --filter="-tags:*" --format="get(name)" 2>/dev/null || true)
  if [[ -z "$digests" ]]; then
    echo "[cleanup] nothing to remove."
    return 0
  fi
  while IFS= read -r digest; do
    [[ -z "$digest" ]] && continue
    echo "[cleanup] deleting ${digest}..."
    gcloud artifacts docker images delete "$digest" --delete-tags --quiet \
      || echo "[cleanup] WARNING: failed to delete ${digest} (continuing)"
  done <<< "$digests"
}
