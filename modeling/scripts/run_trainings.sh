#!/usr/bin/env bash
# Run the two LoRA trainings back to back: BLIP-2 first, then Qwen3-VL. Stops on
# the first failure (set -e). Each run writes to checkpoint_train/<run>/.
#
# The whole body lives in main() so bash parses the entire script before running
# it; this keeps it safe even if the editor re-saves the file mid-run.
main() {
    set -euo pipefail
    cd "$(dirname "$0")/.."   # modeling/ project root (package importable as chartqa)

    echo "=================================================="
    echo "[1/2] BLIP-2 LoRA fine-tuning"
    echo "=================================================="
    python -m chartqa.training.finetune_lora --model blip2

    echo "=================================================="
    echo "[2/2] Qwen3-VL LoRA fine-tuning"
    echo "=================================================="
    python -m chartqa.training.finetune_lora --model qwen

    echo "All trainings completed."
}

main "$@"
