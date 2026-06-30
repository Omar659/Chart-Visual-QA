import json
import os

from chartqa.constants import (
    CATEGORY_TABLE_FILE,
    QUESTIONS_CATEGORIZED_FILE,
    QUESTIONS_DIR,
)


def main():
    categorized_path = os.path.join(QUESTIONS_DIR, QUESTIONS_CATEGORIZED_FILE)
    with open(categorized_path, "r") as f:
        data = json.load(f)

    results = {}
    for file_name, stats in data.items():
        file_results = {}
        for category, sample in stats["success"].items():
            file_results[category] = len(sample)
        for category, sample in stats["failure"].items():
            file_results[category] = round(
                file_results[category] / (file_results[category] + len(sample)) * 100, 2
            )
        results[file_name] = file_results

    print(json.dumps(results, indent=4))

    os.makedirs(QUESTIONS_DIR, exist_ok=True)
    out_path = os.path.join(QUESTIONS_DIR, CATEGORY_TABLE_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"Wrote table -> {out_path}")


if __name__ == "__main__":
    main()
