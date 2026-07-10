#!/usr/bin/env bash
# Shared helpers for the gcloud deploy scripts (scripts/gcloud_deploy_vlm.sh /
# gcloud_deploy_guard.sh / gcloud_deploy_app.sh). Sourced, not executed directly — one
# implementation instead of copy-pasting the same cleanup logic into every script.

# On Windows Git Bash (MSYS2), any command-line arg that looks like a POSIX absolute
# path gets silently rewritten to a Windows path before the target .exe ever sees it —
# e.g. QWEN_ADAPTER_PATH=/app/modeling/checkpoints/... inside a --set-env-vars string
# became "C:/Program Files/Git/app/modeling/checkpoints/..." on the deployed Cloud Run
# service (confirmed 2026-07-10: the container crashed in PeftModel.from_pretrained
# because the path was mangled — real bug, not hypothetical, and it was silently masked
# by an unrelated earlier boot crash in the previous deploy).
#
# Fix scope matters: MSYS_NO_PATHCONV=1 (disable ALL conversion) seems like the obvious
# fix but it's too broad — gcloud's own Windows wrapper (gcloud.cmd) relies on MSYS path
# conversion internally to locate its bundled Python, so disabling it globally breaks
# `gcloud` itself ("python.exe: can't open file 'D:\c\Users\...\gcloud.py'", confirmed by
# testing). MSYS2_ARG_CONV_EXCL scoped to just the --set-env-vars flag avoids that: it
# only suppresses conversion for arguments starting with that literal prefix, leaving
# gcloud's own internal invocation untouched. No-op outside Git Bash, so always safe to
# export.
export MSYS2_ARG_CONV_EXCL="--set-env-vars"

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
