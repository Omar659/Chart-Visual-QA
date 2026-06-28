#!/usr/bin/env bash
# Build a per-sample <dir>_results.json from each errors-dir. Comment out the
# lines you don't need; add more as you create new errors folders (not the diff
# ones).
#
# The whole body lives in main() so bash parses the entire script before running
# it; this keeps it safe even if the editor re-saves the file mid-run.
main() {
    set -euo pipefail
    cd "$(dirname "$0")/.."   # modeling/ project root (package importable as chartqa)

    # --------------------------- Config --------------------------------------
    QWEN_ZERO_SHOT_EXACT="outputs/errors/errors_qwen_zero_shot_exact"
    QWEN_LORA_EXACT="outputs/errors/errors_qwen_lora_finetune_exact"
    QWEN_ZERO_SHOT_RELAXED="outputs/errors/errors_qwen_zero_shot_relaxed"
    QWEN_LORA_RELAXED="outputs/errors/errors_qwen_lora_finetune_relaxed"
    BLIP2_ZERO_SHOT_EXACT="outputs/errors/errors_blip2_zero_shot_exact"
    BLIP2_LORA_EXACT="outputs/errors/errors_blip2_lora_finetune_exact"
    BLIP2_ZERO_SHOT_RELAXED="outputs/errors/errors_blip2_zero_shot_relaxed"
    BLIP2_LORA_RELAXED="outputs/errors/errors_blip2_lora_finetune_relaxed"

    # ------------------------- Per-errors-dir --------------------------------
    echo "[results] $QWEN_ZERO_SHOT_EXACT"
    python -m chartqa.analysis.results_from_errors --errors-dir "$QWEN_ZERO_SHOT_EXACT"

    echo "[results] $QWEN_LORA_EXACT"
    python -m chartqa.analysis.results_from_errors --errors-dir "$QWEN_LORA_EXACT"

    echo "[results] $QWEN_ZERO_SHOT_RELAXED"
    python -m chartqa.analysis.results_from_errors --errors-dir "$QWEN_ZERO_SHOT_RELAXED"

    echo "[results] $QWEN_LORA_RELAXED"
    python -m chartqa.analysis.results_from_errors --errors-dir "$QWEN_LORA_RELAXED"

    echo "[results] $BLIP2_ZERO_SHOT_EXACT"
    python -m chartqa.analysis.results_from_errors --errors-dir "$BLIP2_ZERO_SHOT_EXACT"

    echo "[results] $BLIP2_LORA_EXACT"
    python -m chartqa.analysis.results_from_errors --errors-dir "$BLIP2_LORA_EXACT"

    echo "[results] $BLIP2_ZERO_SHOT_RELAXED"
    python -m chartqa.analysis.results_from_errors --errors-dir "$BLIP2_ZERO_SHOT_RELAXED"

    echo "[results] $BLIP2_LORA_RELAXED"
    python -m chartqa.analysis.results_from_errors --errors-dir "$BLIP2_LORA_RELAXED"
}

main "$@"
