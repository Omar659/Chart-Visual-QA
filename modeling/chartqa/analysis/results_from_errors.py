import argparse
import json
import os

from chartqa.constants import ERRORS_JSON_FILE, RESULT_FAILURE, RESULT_SUCCESS, RESULTS_DIR
from chartqa.data.chartqa_dataset import ChartQADataset

# Build a per-sample results JSON (id, result, question) from an evaluate
# errors-dir. errors.json lists only the FAILED indices (+ the total `n` and the
# `split`), so successes are reconstructed as every index in 0..n-1 that is not
# among the failures.


def main():
    parser = argparse.ArgumentParser(
        description="Per-sample results (id, result, question) from an evaluate errors-dir."
    )
    parser.add_argument(
        "--errors-dir",
        required=True,
        help="An evaluate errors-dir containing errors.json (e.g. "
        "errors/errors_qwen_zero_shot_relaxed). NOT an errors_*_diff_* dir "
        "(those have comparison.json instead).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: <RESULTS_DIR>/results_<name>.json, where "
        "<name> is the errors-dir without its 'errors_' prefix.",
    )
    args = parser.parse_args()

    errors_json = os.path.join(args.errors_dir, ERRORS_JSON_FILE)
    if not os.path.isfile(errors_json):
        raise SystemExit(
            f"No errors.json in {args.errors_dir!r}. Point --errors-dir at an evaluate "
            f"errors folder (not a diff folder)."
        )

    with open(errors_json, encoding="utf-8") as f:
        meta = json.load(f)

    n = meta["n"]
    split = meta["split"]
    failed = {e["index"] for e in meta["errors"]}

    dataset = ChartQADataset(split=split)

    records = [
        {
            "id": idx,
            "result": RESULT_FAILURE if idx in failed else RESULT_SUCCESS,
            "question": dataset[idx]["question"],
        }
        for idx in range(n)
    ]

    if args.out:
        out = args.out
    else:
        # errors_qwen_zero_shot_exact -> results_qwen_zero_shot_exact.json
        base = os.path.basename(args.errors_dir.rstrip("/"))
        stem = base[len("errors_"):] if base.startswith("errors_") else base
        out = os.path.join(RESULTS_DIR, f"results_{stem}.json")
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    n_fail = len(failed)
    print(
        f"Wrote {len(records)} records ({len(records) - n_fail} {RESULT_SUCCESS} / "
        f"{n_fail} {RESULT_FAILURE}) [split={split}, metric={meta.get('metric')}] -> {out}"
    )


if __name__ == "__main__":
    main()
