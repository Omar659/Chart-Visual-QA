#!/usr/bin/env bash
# Evaluate zero-shot (base) vs LoRA-finetuned checkpoint for each model, saving
# each run's wrong images, then build the disagreement set (one technique wrong,
# the other right). Edit the config block below (e.g. METRIC=exact) as needed.
#
# The whole body lives in main() so bash parses the entire script before running
# it; this keeps it safe even if the editor re-saves the file mid-run.
main() {
    set -euo pipefail
    cd "$(dirname "$0")/.."   # modeling/ project root (package importable as chartqa)

    # --------------------------- Config --------------------------------------
    METRIC="relaxed"   # relaxed | exact (always written explicitly in the dir name)

    QWEN_CKPT="./checkpoints/qwen3vl-lora-final2"
    QWEN_ZERO_SHOT_ERR="outputs/errors/errors_qwen_zero_shot_${METRIC}"
    QWEN_LORA_ERR="outputs/errors/errors_qwen_lora_finetune_${METRIC}"
    QWEN_DIFF_DIR="outputs/errors/errors_qwen_diff_${METRIC}"

    BLIP2_CKPT="./checkpoints/blip2-lora-final"
    BLIP2_ZERO_SHOT_ERR="outputs/errors/errors_blip2_zero_shot_${METRIC}"
    BLIP2_LORA_ERR="outputs/errors/errors_blip2_lora_finetune_${METRIC}"
    BLIP2_DIFF_DIR="outputs/errors/errors_blip2_diff_${METRIC}"

    # --------------------------- Qwen3-VL ------------------------------------
    echo "=================================================="
    echo "[Qwen 1/3] Zero-shot (base) | ${METRIC}"
    echo "=================================================="
    python -m chartqa.evaluation.evaluate --model qwen --metric "$METRIC" --errors-dir "$QWEN_ZERO_SHOT_ERR"

    echo "=================================================="
    echo "[Qwen 2/3] LoRA checkpoint ($QWEN_CKPT) | ${METRIC}"
    echo "=================================================="
    python -m chartqa.evaluation.evaluate --model qwen --metric "$METRIC" --checkpoint "$QWEN_CKPT" --errors-dir "$QWEN_LORA_ERR"

    echo "=================================================="
    echo "[Qwen 3/3] Building disagreement set"
    echo "=================================================="
    python -m chartqa.analysis.compare_errors --zero-shot-dir "$QWEN_ZERO_SHOT_ERR" --lora-dir "$QWEN_LORA_ERR" --out-dir "$QWEN_DIFF_DIR"

    # ---------------------------- BLIP-2 -------------------------------------
    echo "=================================================="
    echo "[BLIP-2 1/3] Zero-shot (base) | ${METRIC}"
    echo "=================================================="
    python -m chartqa.evaluation.evaluate --model blip2 --metric "$METRIC" --errors-dir "$BLIP2_ZERO_SHOT_ERR"

    echo "=================================================="
    echo "[BLIP-2 2/3] LoRA checkpoint ($BLIP2_CKPT) | ${METRIC}"
    echo "=================================================="
    python -m chartqa.evaluation.evaluate --model blip2 --metric "$METRIC" --checkpoint "$BLIP2_CKPT" --errors-dir "$BLIP2_LORA_ERR"

    echo "=================================================="
    echo "[BLIP-2 3/3] Building disagreement set"
    echo "=================================================="
    python -m chartqa.analysis.compare_errors --zero-shot-dir "$BLIP2_ZERO_SHOT_ERR" --lora-dir "$BLIP2_LORA_ERR" --out-dir "$BLIP2_DIFF_DIR"

    echo "Done."
}

main "$@"
