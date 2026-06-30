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
        "batch_size": 2,          # BS=1/2 on a 4090 because of the large images
        "grad_accum": 16,         # Simulates a total batch size of (2 * 16) = 32
        "learning_rate": 1e-4,
        "num_epochs": 1,          # Full passes over the training set
        "max_steps": 20,          # Cap on optimizer steps (None = full epochs). NOTE: 20 is a quick-test value.
        "warmup_steps": 1,        # Fixed LR warmup steps (cosine schedule)
        "save_steps": 10,         # Save a checkpoint every N optimizer steps
        "logging_steps": 1,       # Record the loss every N optimizer steps
        "eval_steps": 20,         # Run validation every N optimizer steps
        "eval_max_batches": None,  # Cap val batches per eval for speed (None = full val split)
    },
    "blip2": {
        "batch_size": 2,          # 224x224 images are light; raise if VRAM allows
        "grad_accum": 16,
        "learning_rate": 5e-5,
        "num_epochs": 1,
        "max_steps": 20,
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
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TARGET_MODULES = {
    "qwen": [
        # Language model
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
        # Vision encoder (Qwen3VLVisionAttention and MLP / merger)
        "qkv", "proj", "linear_fc1", "linear_fc2",
    ],
    # BLIP-2: LoRA on the Flan-T5 language model only. The ViT ("qkv"/"projection")
    # and Q-Former ("query"/"key"/"value"/"dense") use different names, so these
    # T5 names leave both frozen automatically.
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
