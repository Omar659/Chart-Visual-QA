import argparse
import json
import os

from chartqa.constants import DEFAULT_SPLIT, QUESTIONS_DIR
from chartqa.data.chartqa_dataset import ChartQADataset


def main():
    parser = argparse.ArgumentParser(
        description="Dump all (id, question) pairs of a ChartQA split to JSON."
    )
    parser.add_argument("--split", default=DEFAULT_SPLIT, help="Dataset split: train/val/test.")
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: <QUESTIONS_DIR>/questions_<split>.json.",
    )
    args = parser.parse_args()

    dataset = ChartQADataset(split=args.split)
    records = [
        {"id": idx, "question": dataset[idx]["question"]} for idx in range(len(dataset))
    ]

    out = args.out or os.path.join(QUESTIONS_DIR, f"questions_{args.split}.json")
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(records)} (id, question) records -> {out}")


if __name__ == "__main__":
    main()
