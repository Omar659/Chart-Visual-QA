import argparse
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")  # headless-safe (WSL); we save to a PNG file
import matplotlib.pyplot as plt

from chartqa.constants import (
    CHECKPOINT_PREFIX,
    CHECKPOINT_TRAIN_DIR,
    LIVE_LOSS_PNG_NAME,
    LOSS_PNG_NAME,
    TRAINER_STATE_FILE,
)


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Return the checkpoint-* directory with the highest step, or None."""
    pattern = re.compile(rf"{re.escape(CHECKPOINT_PREFIX)}(\d+)$")
    ckpts = [
        c for c in glob.glob(os.path.join(output_dir, f"{CHECKPOINT_PREFIX}*"))
        if pattern.search(c)
    ]
    if not ckpts:
        return None
    return max(ckpts, key=lambda c: int(pattern.search(c).group(1)))


def load_log_history(checkpoint: str) -> list:
    """Read `log_history` from a checkpoint dir or a trainer_state.json path."""
    state_path = checkpoint
    if os.path.isdir(checkpoint):
        state_path = os.path.join(checkpoint, TRAINER_STATE_FILE)
    with open(state_path) as f:
        return json.load(f).get("log_history", [])


def plot_log_history(log_history, out=LIVE_LOSS_PNG_NAME, title="Training loss (live)"):
    """Plot train (and eval) loss from an in-memory `log_history` to `out`.

    Built for live plotting during training: it writes a dedicated file and
    closes the figure on every call, so it can be invoked each step without
    leaking figures and without touching loss.png (the per-checkpoint plot).
    """
    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    eval_ = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    if not train:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(*zip(*train), marker="o", label="train loss")
    if eval_:
        plt.plot(*zip(*eval_), marker="s", label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_loss(
    checkpoint: str | None = None,
    output_dir: str = CHECKPOINT_TRAIN_DIR,
    out: str = LOSS_PNG_NAME,
) -> None:
    if checkpoint is None:
        checkpoint = find_latest_checkpoint(output_dir)
        if checkpoint is None:
            raise SystemExit(f"No checkpoint found in {output_dir}")
    print(f"Reading log history from: {checkpoint}")

    log_history = load_log_history(checkpoint)

    # Trainer logs train loss under "loss" and eval loss under "eval_loss".
    train = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
    eval_ = [(e["step"], e["eval_loss"]) for e in log_history if "eval_loss" in e]
    if not train:
        raise SystemExit("No training loss found in log history.")

    plt.figure(figsize=(8, 5))
    plt.plot(*zip(*train), marker="o", label="train loss")
    if eval_:
        plt.plot(*zip(*eval_), marker="s", label="eval loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    print(f"Saved plot to: {os.path.abspath(out)} ({len(train)} points)")


def main():
    parser = argparse.ArgumentParser(description="Plot training loss from a Trainer checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint dir or trainer_state.json. Default: latest in --output-dir.",
    )
    parser.add_argument("--output-dir", default=CHECKPOINT_TRAIN_DIR)
    parser.add_argument("--out", default=LOSS_PNG_NAME)
    args = parser.parse_args()

    plot_loss(args.checkpoint, args.output_dir, args.out)


if __name__ == "__main__":
    main()
