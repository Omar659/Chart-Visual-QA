"""Central configuration for the modeling pipeline.

Every tunable value (model ids, hyperparameters, LoRA setup, evaluation
settings, paths and file names) lives here so the scripts stay declarative and
there is a single place to edit. Pure presentation details (e.g. matplotlib
figure size) are intentionally left next to the plotting code.
"""

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
# HuggingFaceM4/ChartQA ships train/val/test with columns:
#   image, query (question), label (answer, stored as a single-element list),
#   human_or_machine.
DATASET_NAME = "HuggingFaceM4/ChartQA"

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
MODEL_IDS = {
    "qwen": "Qwen/Qwen3-VL-8B-Instruct",
    "blip2": "Salesforce/blip2-flan-t5-xl",
}
MODEL_NAMES = list(MODEL_IDS)  # selectable via --model
DEFAULT_MODEL = "qwen"
TRAIN_DTYPE = "bfloat16"

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
ANSWER_SUFFIX = " Please answer directly."  # nudges short answers (helps exact match)
RELAXED_TOLERANCE = 0.05  # 5% numeric tolerance for the ChartQA relaxed metric
DEFAULT_SPLIT = "test"
DEFAULT_METRIC = "relaxed"
METRIC_NAMES = ["relaxed", "exact"]
# Default generation budget for the (short) answer, per model.
EVAL_MAX_NEW_TOKENS = {"qwen": 1024, "blip2": 32}
# Opt-in bitsandbytes quantization for model loading (--quantization).
# "4bit" = NF4 + double quant + bf16 compute; "8bit" = LLM.int8().
QUANTIZATION_MODES = ["none", "8bit", "4bit"]
DEFAULT_QUANTIZATION = "none"
# Per-sample result labels written by results_from_errors.py.
RESULT_SUCCESS = "success"
RESULT_FAILURE = "failure"

# --------------------------------------------------------------------------- #
# Training hyperparameters (per model)
# --------------------------------------------------------------------------- #
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

HPARAMS = {
    "qwen": {
        # RECOVERED schedule: the values below are read from the committed
        # checkpoints/qwen3vl-lora-final2/training_args.bin (a genuine HF
        # transformers.TrainingArguments saved by an HF Trainer run). Effective budget =
        # max_steps=200 optimizer steps (max_steps>0 overrides epochs), lr 5e-5,
        # warmup_steps 0, per_device_train_batch_size 1 * grad_accum 16 = eff. batch 16.
        # PROVENANCE CAVEAT: the committed adapter came from an HF *Trainer* run
        # (training_args.bin present + NO trainer_state.json = a Trainer final
        # save_model()), whereas this file's training script (finetune_lora.py) is a
        # *custom* PyTorch loop. Two mismatches make these the reproduction TARGET, not a
        # bit-for-bit guarantee: (1) SCHEDULER — training_args.bin says
        # lr_scheduler_type=linear with 0 warmup, but finetune_lora.py hardcodes
        # get_cosine_schedule_with_warmup (cosine) in its "OPTIMIZER AND SCHEDULER"
        # block; the scheduler type is NOT read from these constants, so changing
        # warmup_steps here does NOT switch cosine -> linear. (2) the training script
        # itself differs (HF Trainer vs this custom loop).
        "batch_size": 1,          # per_device_train_batch_size (from training_args.bin)
        "grad_accum": 16,         # 1 * 16 = effective batch 16 (from training_args.bin)
        "learning_rate": 5e-5,    # training_args.bin learning_rate
        "num_epochs": 3,          # training_args.bin num_train_epochs (capped by max_steps=200)
        "max_steps": 200,         # training_args.bin max_steps: recovered optimizer-step budget
        "warmup_steps": 0,        # training_args.bin warmup_steps=0 (see SCHEDULER caveat above)
        "save_steps": 10,         # Save a checkpoint every N optimizer steps (repo-loop cadence)
        "logging_steps": 1,       # Record the loss every N optimizer steps (repo-loop cadence)
        "eval_steps": 20,         # Run validation every N optimizer steps (repo-loop only; committed run used eval_strategy=no)
        "eval_max_batches": None,  # Cap val batches per eval for speed (None = full val split)
    },
    "blip2": {
        "batch_size": 2,          # 224x224 images are light; raise if VRAM allows
        "grad_accum": 16,
        "learning_rate": 5e-5,
        "num_epochs": 1,
        # Recovered from checkpoints/blip2-lora-final/trainer_state.json: 884 optimizer
        # steps (= one full epoch of the train split; final train loss 2.946, val loss
        # 3.536 @ step 20 -> 2.558 @ step 880 best). This reproduces the committed
        # adapter's step budget. (trainer_state records only step/loss/eval_loss, so the
        # lr / warmup / batch below are the repo's declared config, not in that file.)
        "max_steps": 884,
        "warmup_steps": 20,
        "save_steps": 10,
        "logging_steps": 1,
        "eval_steps": 20,
        "eval_max_batches": None,
    },
}

# --------------------------------------------------------------------------- #
# LoRA (target modules / task type differ per architecture)
# --------------------------------------------------------------------------- #
# LoRA rank/alpha are PER MODEL — the two committed adapters were trained with
# different ranks, so these are dicts keyed by model (mirroring LORA_TARGET_MODULES /
# LORA_TASK_TYPE below), not global scalars. Values are pinned to each checkpoint's
# committed adapter_config.json (the source of truth for reproduction):
#   qwen  -> checkpoints/qwen3vl-lora-final2/adapter_config.json  (r=16, alpha=32)
#   blip2 -> checkpoints/blip2-lora-final/adapter_config.json     (r=32, alpha=64)
LORA_R = {"qwen": 16, "blip2": 32}
LORA_ALPHA = {"qwen": 32, "blip2": 64}
LORA_DROPOUT = 0.05  # matches both committed adapters
LORA_BIAS = "none"   # matches both committed adapters
LORA_TARGET_MODULES = {
    # Qwen3-VL: the committed qwen3vl-lora-final2 adapter adapted the LANGUAGE MODEL
    # ONLY (these 7 modules) — the vision tower was NOT targeted. This is the exact
    # config that produced the reported 84.60% -> 86.08% relaxed (75.36% -> 77.00%
    # exact). Do NOT re-add the vision modules (qkv/proj/linear_fc1/linear_fc2): that
    # changes the adapter and no longer reproduces the committed checkpoint.
    "qwen": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # BLIP-2: LoRA on the Flan-T5 language model only (7 modules, order-independent;
    # matches checkpoints/blip2-lora-final/adapter_config.json). The ViT ("qkv"/
    # "projection") and Q-Former ("query"/"key"/"value"/"dense") use different names,
    # so these T5 names leave both frozen automatically.
    "blip2": ["q", "k", "v", "o", "wi_0", "wi_1", "wo"],
}
LORA_TASK_TYPE = {"qwen": "CAUSAL_LM", "blip2": "SEQ_2_SEQ_LM"}

# --------------------------------------------------------------------------- #
# Paths / artifacts
# --------------------------------------------------------------------------- #
# Curated, hand-picked best checkpoints (committed to git).
CHECKPOINTS_DIR = "checkpoints"
# Local training runs (gitignored): each run folder holds checkpoint-<N>,
# checkpoint-best and checkpoint-final, plus the live loss plot.
CHECKPOINT_TRAIN_DIR = "checkpoint_train"

# Generated analysis artifacts (relative to the modeling/ project root).
OUTPUTS_DIR = "outputs"
ERRORS_DIR = f"{OUTPUTS_DIR}/errors"      # evaluate.py wrong-prediction dumps
RESULTS_DIR = f"{OUTPUTS_DIR}/results"    # results_from_errors.py per-sample JSON
QUESTIONS_DIR = f"{OUTPUTS_DIR}/questions"  # dumped questions + category table

BEST_CHECKPOINT_NAME = "checkpoint-best"
FINAL_CHECKPOINT_NAME = "checkpoint-final"
CHECKPOINT_PREFIX = "checkpoint-"  # regular checkpoints: checkpoint-<step>
LIVE_LOSS_PNG_NAME = "loss_during_train.png"
LOSS_PNG_NAME = "loss.png"
TRAINER_STATE_FILE = "trainer_state.json"

# JSON file names produced/consumed by the evaluation + analysis scripts.
ERRORS_JSON_FILE = "errors.json"
COMPARISON_JSON_FILE = "comparison.json"
QUESTIONS_CATEGORIZED_FILE = "questions_categorized.json"
# Per-category accuracy table produced by create_table.py.
CATEGORY_TABLE_FILE = "category_table.json"
