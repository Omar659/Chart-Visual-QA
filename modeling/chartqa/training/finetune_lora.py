import argparse
import json
import os
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm

from chartqa.constants import (
    BEST_CHECKPOINT_NAME,
    CHECKPOINT_PREFIX,
    CHECKPOINT_TRAIN_DIR,
    DEFAULT_MODEL,
    FINAL_CHECKPOINT_NAME,
    HPARAMS,
    LIVE_LOSS_PNG_NAME,
    LORA_ALPHA,
    LORA_BIAS,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    LORA_TASK_TYPE,
    MAX_GRAD_NORM,
    MODEL_IDS,
    MODEL_NAMES,
    TRAIN_DTYPE,
    TRAINER_STATE_FILE,
    WEIGHT_DECAY,
)
from chartqa.models.qwen_vl_chat import QwenVLChat
from chartqa.models.blip2_chat import Blip2Chat
from chartqa.data.chartqa_dataset import ChartQADataset
from chartqa.data.collators import ChartQACollator, Blip2Collator
from chartqa.viz.plot_loss import plot_log_history

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_wrapper(model: str):
    """Instantiate the base model wrapper for the chosen model."""
    if model == "qwen":
        return QwenVLChat(model_name=MODEL_IDS[model], dtype=TRAIN_DTYPE, attn_implementation="sdpa")
    return Blip2Chat(model_name=MODEL_IDS[model], dtype=TRAIN_DTYPE)


def build_lora_config(model: str) -> LoraConfig:
    # LORA_R / LORA_ALPHA are per-model dicts (the committed adapters used different
    # ranks: qwen r=16/alpha=32, blip2 r=32/alpha=64), so index them by model.
    return LoraConfig(
        r=LORA_R[model],
        lora_alpha=LORA_ALPHA[model],
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=LORA_TARGET_MODULES[model],
        task_type=LORA_TASK_TYPE[model],
    )


def build_collator(model: str, processor):
    """Qwen uses the chat-template collator; BLIP-2 the seq2seq one."""
    if model == "qwen":
        return ChartQACollator(processor=processor)
    return Blip2Collator(processor=processor)


def save_trainer_state(directory, log_history):
    """Save the loss history in the format expected by plot_loss.py."""
    state_data = {
        "best_global_step": None,
        "best_metric": None,
        "best_model_checkpoint": None,
        "log_history": log_history,  # Key read by plot_loss.py
    }
    state_path = os.path.join(directory, TRAINER_STATE_FILE)
    with open(state_path, "w") as f:
        json.dump(state_data, f, indent=2)
    tqdm.write(f"Trainer state updated at: {state_path}")


@torch.no_grad()
def run_validation(model, dataloader, max_batches=None):
    """Average cross-entropy loss over the validation set (lower is better).

    Pure forward passes with labels (no generation), so it is cheap. Pass
    `max_batches` to evaluate only a prefix of the val split for speed.
    """
    model.eval()
    total_loss, n_batches = 0.0, 0
    total = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
    for i, batch in enumerate(tqdm(dataloader, total=total, desc="validation", unit="batch", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        inputs = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**inputs).loss
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning on ChartQA (Qwen3-VL or BLIP-2).")
    parser.add_argument("--model", choices=MODEL_NAMES, default=DEFAULT_MODEL, help="Which model to fine-tune.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run folder name under checkpoint_train/. Default: <model>_lora_<timestamp>.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help=(
            "Cap on optimizer steps (overrides HPARAMS[model]['max_steps']). "
            "blip2 default = 884 (recovered from the committed trainer_state.json). "
            "qwen default = 200 (recovered from the committed training_args.bin). Both "
            "are reproduction targets, not bit-for-bit guarantees (see MODELCARD)."
        ),
    )
    args = parser.parse_args()
    model_name = args.model
    hp = HPARAMS[model_name]

    # Resolve the optimizer-step budget: CLI overrides the per-model constants default.
    # This makes the step count a conscious choice instead of an accidental toy value.
    max_steps = args.max_steps if args.max_steps is not None else hp["max_steps"]
    if model_name == "qwen":
        # 200 steps is the recovered budget from the committed qwen3vl-lora-final2
        # training_args.bin, so the step count is known. The remaining provenance gap is
        # the SCHEDULER/SCRIPT: that checkpoint was trained by an HF Trainer with a
        # LINEAR scheduler (lr_scheduler_type=linear, warmup 0), while this custom loop
        # hardcodes get_cosine_schedule_with_warmup (cosine) below -- so even with the
        # right steps/lr this run approximates rather than bit-for-bit reproduces it.
        print(
            f"NOTE: qwen optimizer-step budget = {max_steps} (recovered from the "
            "committed training_args.bin). Provenance caveat: the committed adapter used "
            "an HF Trainer with a LINEAR scheduler; this loop uses cosine, so it is a "
            "reproduction target, not a bit-for-bit guarantee (see MODELCARD)."
        )

    # All run artifacts (checkpoint-<N>, checkpoint-best, checkpoint-final, live
    # loss plot) live in one run folder under checkpoint_train/ (gitignored). Copy
    # the best ones into checkpoints/ by hand to keep/commit them.
    run_name = args.run_name or f"{model_name}_lora_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = os.path.join(CHECKPOINT_TRAIN_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    live_loss_png = os.path.join(run_dir, LIVE_LOSS_PNG_NAME)
    print(f"Run artifacts -> {run_dir}")

    print("Loading the wrapper model in bfloat16...")
    chat_wrapper = build_wrapper(model_name)
    model = chat_wrapper.model
    processor = chat_wrapper.processor

    # VRAM-saving configuration
    model.config.use_cache = False  # Disabled for training (incompatible with gradient checkpointing)
    model.gradient_checkpointing_enable()
    # Required by PEFT when gradient checkpointing is on
    model.enable_input_require_grads()

    # =====================================================================
    # APPLY PEFT (LoRA)
    # =====================================================================
    print("Configuring and applying LoRA...")
    peft_config = build_lora_config(model_name)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Swap the wrapper's model for its PEFT-wrapped version
    chat_wrapper.model = model

    # =====================================================================
    # DATASET AND DATALOADER
    # =====================================================================
    print("Initializing Dataset and DataLoader...")
    train_dataset = ChartQADataset(split="train")
    collator = build_collator(model_name, processor)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hp["batch_size"],
        shuffle=True,
        collate_fn=collator,
        drop_last=True,
    )

    # Validation split for best-model selection (lowest validation loss).
    val_dataset = ChartQADataset(split="val")
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hp["batch_size"],
        shuffle=False,
        collate_fn=collator,
        drop_last=False,
    )

    # =====================================================================
    # OPTIMIZER AND SCHEDULER
    # =====================================================================
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = AdamW(trainable_params, lr=hp["learning_rate"], weight_decay=WEIGHT_DECAY)

    # One optimizer step = grad_accum batches. Derive the number of optimizer
    # steps from the dataset so we cover the whole training set num_epochs times,
    # optionally clamped to the max_steps cap.
    steps_per_epoch = len(train_dataloader) // hp["grad_accum"]
    total_steps = steps_per_epoch * hp["num_epochs"]
    if max_steps is not None:
        total_steps = min(total_steps, max_steps)
    print(
        f"Steps/epoch: {steps_per_epoch} | epochs: {hp['num_epochs']} | "
        f"cap (max_steps): {max_steps} -> total optimizer steps: {total_steps}"
    )

    # Cosine schedule with a fixed warmup.
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=hp["warmup_steps"],
        num_training_steps=total_steps,
    )

    # =====================================================================
    # MAIN TRAINING LOOP (epoch-based)
    # =====================================================================
    print(f"Starting custom training: {hp['num_epochs']} epochs, {total_steps} optimization steps...")
    model.train()

    running_loss = 0.0
    optimizer.zero_grad()

    step_losses = []
    log_history = []
    best_loss = float("inf")  # lowest validation loss seen so far

    optimizer_step = 0
    stop = False

    for epoch in range(hp["num_epochs"]):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{hp['num_epochs']}", unit="batch")
        for batch_idx, batch in enumerate(pbar):
            inputs = {
                k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(**inputs)
                loss = outputs.loss

            (loss / hp["grad_accum"]).backward()
            running_loss += loss.item()

            # Run the real optimization step every grad_accum batches.
            if (batch_idx + 1) % hp["grad_accum"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                optimizer_step += 1

                # Loss of the current optimization step.
                step_loss = running_loss / hp["grad_accum"]
                step_losses.append(step_loss)
                running_loss = 0.0  # Reset the accumulator for the next step

                # Logging: update the progress bar and record the training loss.
                if optimizer_step % hp["logging_steps"] == 0 or optimizer_step == 1:
                    avg_loss = sum(step_losses) / len(step_losses)
                    pbar.set_postfix(
                        step=f"{optimizer_step}/{total_steps}",
                        loss=f"{avg_loss:.4f}",
                        lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    )
                    log_history.append({"step": optimizer_step, "loss": avg_loss})
                    step_losses = []

                # Validation step: keep the checkpoint with the lowest validation loss.
                if optimizer_step % hp["eval_steps"] == 0:
                    eval_loss = run_validation(model, val_dataloader, hp["eval_max_batches"])
                    tqdm.write(f"[step {optimizer_step}] validation loss: {eval_loss:.4f}")
                    log_history.append({"step": optimizer_step, "eval_loss": eval_loss})

                    if eval_loss < best_loss:
                        best_loss = eval_loss
                        best_ckpt_path = os.path.join(run_dir, BEST_CHECKPOINT_NAME)
                        tqdm.write(f"New best validation loss: {best_loss:.4f} -> saving {BEST_CHECKPOINT_NAME}")
                        model.save_pretrained(best_ckpt_path)
                        processor.save_pretrained(best_ckpt_path)
                        save_trainer_state(best_ckpt_path, log_history)

                # Regular checkpoint every save_steps
                if optimizer_step % hp["save_steps"] == 0:
                    ckpt_path = os.path.join(run_dir, f"{CHECKPOINT_PREFIX}{optimizer_step}")
                    tqdm.write(f"Saving regular checkpoint: {ckpt_path}")
                    model.save_pretrained(ckpt_path)
                    processor.save_pretrained(ckpt_path)
                    save_trainer_state(ckpt_path, log_history)

                # Live loss plot, refreshed every optimizer step (inside the run folder).
                plot_log_history(log_history, live_loss_png)

                # max_steps cap: stop once the optimizer-step budget is reached.
                if max_steps is not None and optimizer_step >= max_steps:
                    stop = True
                    break
        pbar.close()
        if stop:
            break

    # =====================================================================
    # SAVE FINAL LORA WEIGHTS (in the same run folder)
    # =====================================================================
    final_path = os.path.join(run_dir, FINAL_CHECKPOINT_NAME)
    print(f"Training complete. Saving final adapter to: {final_path}")
    model.save_pretrained(final_path)
    processor.save_pretrained(final_path)
    save_trainer_state(final_path, log_history)
    print("Save completed successfully.")


if __name__ == "__main__":
    main()
