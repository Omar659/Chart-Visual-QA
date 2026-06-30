import argparse
import json
import os
import shutil

from chartqa.constants import (
    ANSWER_SUFFIX,
    DEFAULT_METRIC,
    DEFAULT_MODEL,
    DEFAULT_SPLIT,
    ERRORS_JSON_FILE,
    EVAL_MAX_NEW_TOKENS,
    METRIC_NAMES,
    MODEL_NAMES,
)
from chartqa.evaluation.metrics import METRICS
from chartqa.data.chartqa_dataset import ChartQADataset
from chartqa.models.qwen_vl_chat import QwenVLChat
from chartqa.models.blip2_chat import Blip2Chat

# Both wrappers share the chat(image, text, system_prompt, max_new_tokens) API
# and accept adapter_path, so the evaluation loop is model-agnostic.
WRAPPERS = {"qwen": QwenVLChat, "blip2": Blip2Chat}


def evaluate(
    model: str = DEFAULT_MODEL,
    checkpoint: str | None = None,
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    max_new_tokens: int | None = None,
    metric: str = DEFAULT_METRIC,
    errors_dir: str | None = None,
) -> float:
    match_fn = METRICS[metric]
    # adapter_path=None -> base model; a path -> base model + LoRA checkpoint.
    chat_model = WRAPPERS[model](adapter_path=checkpoint)
    dataset = ChartQADataset(split=split)

    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS[model]

    # Optionally dump the images the model gets wrong + an errors.json. The folder
    # is wiped on every run so it only holds the current run's mistakes.
    errors = []
    if errors_dir:
        if os.path.isdir(errors_dir):
            shutil.rmtree(errors_dir)
        os.makedirs(errors_dir, exist_ok=True)

    n = len(dataset) if limit is None else min(limit, len(dataset))
    correct = 0
    for idx in range(n):
        sample = dataset[idx]
        raw = chat_model.chat(
            image=sample["image"],
            text=sample["question"] + ANSWER_SUFFIX,
            max_new_tokens=max_new_tokens,
        )
        # Strip a leading "Answer:" if the model echoes it.
        pred = raw.split("Answer:")[-1].strip()
        gold = sample["answer"]
        if match_fn(pred, gold):
            correct += 1
        elif errors_dir:
            # Save the misclassified chart image, named by its dataset index.
            img_name = f"{idx}.png"
            sample["image"].save(os.path.join(errors_dir, img_name))
            errors.append({"index": idx, "image": img_name, "gold": gold, "pred": pred})
        print(
            f"[{idx + 1}/{n}] gold={gold!r} pred={pred!r} "
            f"running_acc={correct / (idx + 1) * 100:.2f}%"
        )

    accuracy = correct / n if n else 0.0
    tag = checkpoint if checkpoint else "base model"
    print(
        f"\n{model} {metric} accuracy ({tag}, split={split}): "
        f"{correct}/{n} = {accuracy * 100:.2f}%"
    )

    if errors_dir:
        with open(os.path.join(errors_dir, ERRORS_JSON_FILE), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model": model,
                    "checkpoint": checkpoint or "base model",
                    "split": split,
                    "metric": metric,
                    "n": n,
                    "num_errors": len(errors),
                    "errors": errors,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"Saved {len(errors)} wrong-prediction images + errors.json to: {errors_dir}")

    return accuracy


def main():
    parser = argparse.ArgumentParser(description="Evaluate a VLM (Qwen3-VL or BLIP-2) on ChartQA.")
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default=DEFAULT_MODEL,
        help="Which model wrapper to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a LoRA adapter checkpoint. Omit to evaluate the base model.",
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split: train/val/test.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Evaluate only the first N samples."
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Generation budget. Default: per-model.",
    )
    parser.add_argument(
        "--metric",
        choices=METRIC_NAMES,
        default=DEFAULT_METRIC,
        help="Scoring: 'relaxed' (ChartQA, numeric tolerance) or 'exact' string match.",
    )
    parser.add_argument(
        "--errors-dir",
        default=None,
        help="If set, save wrong-prediction images + errors.json here (folder cleared each run).",
    )
    args = parser.parse_args()

    evaluate(
        args.model, args.checkpoint, args.split, args.limit, args.max_new_tokens,
        args.metric, args.errors_dir,
    )


if __name__ == "__main__":
    main()
