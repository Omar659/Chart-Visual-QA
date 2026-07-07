# Model card — qwen3vl-lora-final2

LoRA adapter for chart visual question answering (ChartQA). This is the committed,
curated checkpoint that produced the team's best reported result. It is the
reproduction source of truth: `modeling/chartqa/constants.py` is pinned to the values
below (Phase 2.2).

## Base model

- `Qwen/Qwen3-VL-8B-Instruct` (vision-language, `CAUSAL_LM`).

## LoRA configuration (measured — from `adapter_config.json`)

| Field                                          | Value                                                          |
| ---------------------------------------------- | ------------------------------------------------------------- |
| rank `r`                                       | 16                                                            |
| `lora_alpha`                                   | 32                                                            |
| `lora_dropout`                                 | 0.05                                                         |
| `bias`                                         | none                                                          |
| `task_type`                                    | CAUSAL_LM                                                     |
| `peft_version`                                 | 0.19.1                                                       |
| target modules (7 — **language model only**)   | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |

> The vision tower was **not** adapted — only the language model's attention and MLP
> projections. Adding vision modules (`qkv / proj / linear_fc1 / linear_fc2`) changes
> the adapter and will not reproduce this checkpoint. `LORA_TARGET_MODULES["qwen"]` is
> pinned to these 7 modules for that reason.

## Evaluation (measured — ChartQA test, 2500 samples; roadmap §3 Results)

| Metric           | Zero-shot     | LoRA fine-tuned       | Δ        |
| ---------------- | ------------- | --------------------- | -------- |
| Relaxed (5% tol) | 84.60% (2115) | **86.08%** (2152)     | +1.48 pp |
| Exact            | 75.36% (1884) | **77.00%** (1925)     | +1.64 pp |

## Training schedule — recovered from committed `training_args.bin` (HF `TrainingArguments`)

This checkpoint ships a `training_args.bin` next to the adapter (same commit `dcb1d20`,
same mtime as `adapter_config.json`). It is a genuine, parseable
`transformers.TrainingArguments` object; loading it
(`torch.load(path, weights_only=False)` — a trusted in-repo artifact) recovers the
committed Qwen run's real schedule:

| Field                                             | Value                                                                   |
| ------------------------------------------------- | ----------------------------------------------------------------------- |
| `max_steps`                                       | **200** (>0 ⇒ overrides epochs; effective budget = 200 optimizer steps) |
| `learning_rate`                                   | **5e-5**                                                                |
| `warmup_steps`                                    | **0**                                                                   |
| `lr_scheduler_type`                               | **linear**                                                              |
| `per_device_train_batch_size`                     | **1**                                                                   |
| `gradient_accumulation_steps`                     | **16** → effective batch **16** (1 × 16)                                |
| `num_train_epochs`                                | 3.0 (capped by `max_steps=200`)                                         |
| `optim`                                           | `adamw_torch_fused`                                                     |
| `weight_decay`                                    | 0.0                                                                     |
| `max_grad_norm`                                   | 1.0                                                                     |
| `seed`                                            | 42                                                                      |
| `bf16` / `gradient_checkpointing`                 | True / True                                                             |
| `save_steps` / `logging_steps` / `eval_strategy`  | 50 / 5 / no                                                             |

`HPARAMS["qwen"]` in `constants.py` is pinned to these values as the reproduction
target: `max_steps=200`, `learning_rate=5e-5`, `warmup_steps=0`, `batch_size=1` ×
`grad_accum=16` = effective **16**.

### Provenance & reproducibility caveat

These values are the **best-known reproduction target for the committed checkpoint**, not
a guarantee of a bit-for-bit rerun. The committed adapter was evidently produced by an
**HF `Trainer`** run — the signature is `training_args.bin` present **and no
`trainer_state.json`**, i.e. a Trainer *final* `save_model()`. The repo's current
`modeling/chartqa/training/finetune_lora.py` is instead a **custom PyTorch training
loop** with different declared defaults, and two concrete mismatches remain:

- **Scheduler:** the artifact says `lr_scheduler_type = linear` with `warmup_steps = 0`,
  but `finetune_lora.py` hardcodes `get_cosine_schedule_with_warmup` (cosine). This is
  set directly in the "OPTIMIZER AND SCHEDULER" block of that file, **not** read from
  `constants.py`, so aligning `warmup_steps` in constants does not switch cosine → linear.
- **Training script:** the committed run used the HF `Trainer`; this repo runs its own
  loop. Even with identical hyperparameters the two code paths are not guaranteed to
  produce identical weights.

So running the current repo code with the recovered values approximates, but does not
provably reproduce bit-for-bit, the committed adapter.

**Measured-from-artifact vs still-uncertain:**

- *Measured* (`adapter_config.json` + `training_args.bin`): base model, all LoRA
  hyperparameters above, and the full schedule table (steps / lr / warmup / scheduler /
  batch / optim / seed).
- *Still uncertain:* whether re-running today's repo loop with these values reproduces
  the 86.08% relaxed result — the scheduler and training script differ, and there is no
  `trainer_state.json` to confirm the per-step loss trajectory. Confirmation needs the
  GPU re-eval below.

## Hardware

- Trained on a 24 GB RTX 4090 (roadmap §3 training notes). No quantization
  (full-precision bf16 base).

## Reproduction status

- Config reconciled to this adapter in Phase 2.2 (constants pinned to r=16 / alpha=32,
  7 LM-only target modules).
- **GPU re-eval verification** (confirm this adapter re-evaluates to 86.08% relaxed) is
  **deferred to cloud / Phase 1.5** — Qwen3-VL-8B inference does not fit the local 6 GB
  RTX 4050.
