# Model card — blip2-lora-final

LoRA adapter for chart visual question answering (ChartQA). Weak baseline-2 contrast
model: BLIP-2 (flan-t5-xl) is architecturally poor at reading chart text / numbers, so
this is a "baseline 1 vs baseline 2" reference, **not** a deployment candidate.

## Base model

- `Salesforce/blip2-flan-t5-xl` (`SEQ_2_SEQ_LM`).

## LoRA configuration (measured — from `adapter_config.json`)

| Field                                   | Value                        |
| --------------------------------------- | ---------------------------- |
| rank `r`                                | 32                           |
| `lora_alpha`                            | 64                           |
| `lora_dropout`                          | 0.05                         |
| `bias`                                  | none                         |
| `task_type`                             | SEQ_2_SEQ_LM                 |
| `peft_version`                          | 0.19.1                       |
| target modules (7 — Flan-T5 LM only)    | `q, k, v, o, wi_0, wi_1, wo` |

## Evaluation (measured — ChartQA test, 2500 samples; roadmap §3 Results)

| Metric           | Zero-shot   | LoRA fine-tuned | Δ        |
| ---------------- | ----------- | --------------- | -------- |
| Relaxed (5% tol) | 8.40% (210) | 12.40% (310)    | +4.00 pp |
| Exact            | 0.84% (21)  | 5.60% (140)     | +4.76 pp |

## Training schedule (partially measured — from `trainer_state.json`)

| Field                          | Value                                              | Source                                         |
| ------------------------------ | -------------------------------------------------- | ---------------------------------------------- |
| optimizer steps (`max_steps`)  | **884** (= one full epoch of the train split)      | measured (trainer_state)                       |
| final train loss (step 884)    | 2.946                                              | measured (trainer_state)                       |
| validation loss                | 3.536 (step 20) → **2.558** (step 880, best)       | measured (44 evals, every 20 steps)            |
| learning rate                  | 5e-5                                               | `constants.py` declared — **not** in trainer_state |
| warmup steps                   | 20                                                 | `constants.py` declared — **not** in trainer_state |
| effective batch                | 32 (batch 2 × grad-accum 16)                       | `constants.py` declared — **not** in trainer_state |

> `trainer_state.json` records only `step`, `loss`, and periodic `eval_loss`. It does
> **not** contain lr / warmup / batch, so those rows are the repo's declared config, not
> independently verified from the checkpoint. The file's `best_*` fields are all null
> (the custom training loop does not populate them); the best validation loss is the
> minimum `eval_loss` above (2.558 @ step 880).

**Measured vs unknown:** base model, all LoRA hyperparameters, the eval metrics, the
884-step budget, and the train/val loss curve are measured. The lr / warmup / batch are
declared-but-unverified.

## Hardware

- Trained on a 24 GB RTX 4090 (roadmap §3 training notes). No quantization
  (full-precision bf16 base). BLIP-2 also fits a 6 GB RTX 4050 at 4-bit for
  inference / eval.
