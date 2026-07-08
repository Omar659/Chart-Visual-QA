import argparse
import json
import os
import shutil
import time

from chartqa.constants import (
    ANSWER_SUFFIX,
    DEFAULT_METRIC,
    DEFAULT_MODEL,
    DEFAULT_QUANTIZATION,
    DEFAULT_SPLIT,
    ERRORS_JSON_FILE,
    EVAL_MAX_NEW_TOKENS,
    LORA_ALPHA,
    LORA_R,
    LORA_TARGET_MODULES,
    METRIC_NAMES,
    MODEL_IDS,
    MODEL_NAMES,
    QUANTIZATION_MODES,
)
# tracking imports mlflow (fail-open); kept BEFORE the dataset/model imports so any
# pyarrow it pulls loads ahead of torch (the Windows torch-before-pyarrow crash guard,
# see chartqa/data/chartqa_dataset.py). It never imports torch itself.
from chartqa import tracking
from chartqa.evaluation.metrics import METRICS
from chartqa.data.chartqa_dataset import ChartQADataset
from chartqa.models.qwen_vl_chat import QwenVLChat
from chartqa.models.blip2_chat import Blip2Chat
# torch is already imported (transitively) by the model wrappers above, so this
# explicit import runs AFTER datasets and is a safe no-op re-import. Used for the
# CPU-safe latency/VRAM instrumentation below.
import torch

# Both wrappers share the chat(image, text, system_prompt, max_new_tokens) API
# and accept adapter_path, so the evaluation loop is model-agnostic.
WRAPPERS = {"qwen": QwenVLChat, "blip2": Blip2Chat}


def _hardware_tag() -> str:
    """CUDA device name when a GPU is present, else 'cpu' — the MLflow hardware tag."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
    except Exception:  # noqa: BLE001 - never let a tag lookup break the eval
        pass
    return "cpu"


def _lora_params(model: str) -> dict:
    """Resolved LoRA config for the run params (only meaningful with an adapter)."""
    return {
        "lora_r": LORA_R[model],
        "lora_alpha": LORA_ALPHA[model],
        "lora_target_modules": len(LORA_TARGET_MODULES[model]),
    }


def _dir_size_bytes(path: str) -> int:
    """Total on-disk size of a directory tree (used for the adapter size metric)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted, non-empty list."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = rank - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def _latency_stats(latencies: list[float]) -> dict:
    """Mean/p50/p95 of per-sample latency, discarding the first (warmup) sample."""
    timed = latencies[1:] if len(latencies) > 1 else latencies
    if not timed:
        return {}
    s = sorted(timed)
    return {
        "latency_mean_s": sum(s) / len(s),
        "latency_p50_s": _percentile(s, 50),
        "latency_p95_s": _percentile(s, 95),
    }


def evaluate(
    model: str = DEFAULT_MODEL,
    checkpoint: str | None = None,
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
    max_new_tokens: int | None = None,
    metric: str = DEFAULT_METRIC,
    errors_dir: str | None = None,
    quantization: str = DEFAULT_QUANTIZATION,
) -> float:
    match_fn = METRICS[metric]

    # Peak-VRAM measurement spans model load + inference (the "does it fit on 6 GB"
    # number the quantization study needs). Guarded so a CPU-only box skips it cleanly.
    cuda = False
    try:
        cuda = torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - torch built without CUDA, or a driver hiccup
        cuda = False
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    # adapter_path=None -> base model; a path -> base model + LoRA checkpoint.
    # quantization != "none" loads the base 4-/8-bit (bitsandbytes) and keeps
    # any LoRA adapter attached instead of merging it.
    load_start = time.perf_counter()
    chat_model = WRAPPERS[model](adapter_path=checkpoint, quantization=quantization)
    load_time_s = time.perf_counter() - load_start

    dataset = ChartQADataset(split=split)

    if max_new_tokens is None:
        max_new_tokens = EVAL_MAX_NEW_TOKENS[model]

    n = len(dataset) if limit is None else min(limit, len(dataset))

    run_name = f"{model}-{quantization}-{metric}"
    if limit:
        run_name += f"-limit{limit}"

    # One MLflow run wraps the whole eval; fail-open (no-op if mlflow absent/disabled),
    # so it never changes eval behavior or crashes a run — the results JSON is still
    # written either way.
    with tracking.track_run(run_name, tags={"hardware": _hardware_tag()},
                            experiment="chartqa-eval"):
        run_params = {
            "model": model,
            "base_model": MODEL_IDS[model],
            "checkpoint": checkpoint or "base model",
            "split": split,
            "limit": limit,
            "n": n,
            "metric": metric,
            "quantization": quantization,
            "max_new_tokens": max_new_tokens,
        }
        if checkpoint:
            run_params.update(_lora_params(model))
        tracking.log_params(run_params)

        # Optionally dump the images the model gets wrong + an errors.json. The folder
        # is wiped on every run so it only holds the current run's mistakes.
        errors = []
        if errors_dir:
            if os.path.isdir(errors_dir):
                shutil.rmtree(errors_dir)
            os.makedirs(errors_dir, exist_ok=True)

        correct = 0
        latencies = []
        for idx in range(n):
            sample = dataset[idx]
            t0 = time.perf_counter()
            raw = chat_model.chat(
                image=sample["image"],
                text=sample["question"] + ANSWER_SUFFIX,
                max_new_tokens=max_new_tokens,
            )
            if cuda:
                torch.cuda.synchronize()  # count GPU work, not just kernel launch
            latencies.append(time.perf_counter() - t0)
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

        # Accuracy + the GPU-study extras (latency / peak VRAM / load time / adapter
        # size), each guarded so CPU-only / base-model runs just omit what doesn't apply.
        run_metrics = {
            f"{metric}_accuracy": accuracy,
            "correct": correct,
            "n": n,
            "load_time_s": load_time_s,
        }
        run_metrics.update(_latency_stats(latencies))
        if cuda:
            run_metrics["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1e9
        if checkpoint and os.path.isdir(checkpoint):
            run_metrics["adapter_size_mb"] = _dir_size_bytes(checkpoint) / 1e6
        tracking.log_metrics(run_metrics)

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
            tracking.log_artifact(errors_dir)

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
    parser.add_argument(
        "--quantization",
        choices=QUANTIZATION_MODES,
        default=DEFAULT_QUANTIZATION,
        help="Opt-in bitsandbytes precision: '4bit' = NF4 + double quant + bf16 compute, "
        "'8bit' = LLM.int8(). Default: full precision. With --checkpoint, quantized "
        "runs keep the LoRA adapter attached (a quantized base cannot be merged).",
    )
    args = parser.parse_args()

    evaluate(
        args.model, args.checkpoint, args.split, args.limit, args.max_new_tokens,
        args.metric, args.errors_dir, args.quantization,
    )


if __name__ == "__main__":
    main()
