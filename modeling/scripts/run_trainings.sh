#!/usr/bin/env bash
# Run the two LoRA trainings back to back: BLIP-2 first, then Qwen3-VL. Stops on
# the first failure (set -e). Each run writes to checkpoint_train/<run>/.
#
# The whole body lives in main() so bash parses the entire script before running
# it; this keeps it safe even if the editor re-saves the file mid-run.
main() {
    set -euo pipefail
    cd "$(dirname "$0")/.."   # modeling/ project root (package importable as chartqa)

    # BLIP-2: 884 optimizer steps, recovered from the committed
    # checkpoints/blip2-lora-final/trainer_state.json (= one full epoch of the train
    # split). This reproduces the committed adapter's step budget.
    BLIP2_MAX_STEPS="${BLIP2_MAX_STEPS:-884}"

    # Qwen3-VL: 200 optimizer steps, recovered from the committed
    # checkpoints/qwen3vl-lora-final2/training_args.bin (max_steps=200, the effective
    # step budget since max_steps>0 overrides epochs). Reproduction target only -- the
    # committed run used an HF Trainer with a linear scheduler while this repo's loop
    # uses cosine (see MODELCARD / constants.py), so it is not a bit-for-bit guarantee.
    # Override to explore other budgets, e.g. QWEN_MAX_STEPS=<n> bash scripts/run_trainings.sh
    QWEN_MAX_STEPS="${QWEN_MAX_STEPS:-200}"

    echo "=================================================="
    echo "[1/2] BLIP-2 LoRA fine-tuning (max_steps=${BLIP2_MAX_STEPS})"
    echo "=================================================="
    python -m chartqa.training.finetune_lora --model blip2 --max-steps "$BLIP2_MAX_STEPS"

    echo "=================================================="
    echo "[2/2] Qwen3-VL LoRA fine-tuning (max_steps=${QWEN_MAX_STEPS})"
    echo "=================================================="
    python -m chartqa.training.finetune_lora --model qwen --max-steps "$QWEN_MAX_STEPS"

    echo "All trainings completed."
}

main "$@"
