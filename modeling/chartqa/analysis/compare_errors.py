import argparse
import json
import os
import shutil

from chartqa.constants import COMPARISON_JSON_FILE, ERRORS_JSON_FILE


def load_errors(errors_dir: str):
    """Load an evaluate.py errors.json -> (dict index->entry, full metadata)."""
    with open(os.path.join(errors_dir, ERRORS_JSON_FILE), encoding="utf-8") as f:
        data = json.load(f)
    return {e["index"]: e for e in data["errors"]}, data


def main():
    parser = argparse.ArgumentParser(
        description="Build the set of samples where exactly one of two evaluate runs failed."
    )
    parser.add_argument("--zero-shot-dir", required=True, help="errors-dir of the zero-shot run.")
    parser.add_argument("--lora-dir", required=True, help="errors-dir of the LoRA-finetuned run.")
    parser.add_argument("--out-dir", required=True, help="Output folder (cleared each run).")
    args = parser.parse_args()

    zs_err, zs_meta = load_errors(args.zero_shot_dir)
    lora_err, lora_meta = load_errors(args.lora_dir)

    zs_failed = set(zs_err)
    lora_failed = set(lora_err)
    # Symmetric difference: exactly one technique failed (the other got it right).
    disagree = sorted(zs_failed ^ lora_failed)

    # Clear the output folder on every run.
    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    for idx in disagree:
        if idx in zs_failed:
            # zero-shot wrong, LoRA right -> under exact, the right pred == gold.
            entry = zs_err[idx]
            src_dir, failed_by = args.zero_shot_dir, "zero_shot"
            gold = entry["gold"]
            pred_zero_shot, pred_lora_finetune = entry["pred"], gold
        else:
            entry = lora_err[idx]
            src_dir, failed_by = args.lora_dir, "lora_finetune"
            gold = entry["gold"]
            pred_zero_shot, pred_lora_finetune = gold, entry["pred"]

        img = entry["image"]
        shutil.copyfile(os.path.join(src_dir, img), os.path.join(args.out_dir, img))
        records.append(
            {
                "index": idx,
                "image": img,
                "gold": gold,
                "pred_zero_shot": pred_zero_shot,
                "pred_lora_finetune": pred_lora_finetune,
                "failed_by": failed_by,
            }
        )

    with open(os.path.join(args.out_dir, COMPARISON_JSON_FILE), "w", encoding="utf-8") as f:
        json.dump(
            {
                "zero_shot": zs_meta.get("checkpoint"),
                "lora_finetune": lora_meta.get("checkpoint"),
                "metric": zs_meta.get("metric"),
                "zero_shot_failed_total": len(zs_failed),
                "lora_finetune_failed_total": len(lora_failed),
                "n_disagreements": len(records),
                "note": "Only samples where exactly one technique failed. For the technique "
                "that did NOT fail, pred == gold (exact metric).",
                "disagreements": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(
        f"Disagreements (exactly one failed): {len(records)} "
        f"(zero-shot-only-wrong={len(zs_failed - lora_failed)}, "
        f"lora-only-wrong={len(lora_failed - zs_failed)}) -> {args.out_dir}"
    )


if __name__ == "__main__":
    main()
