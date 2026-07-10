# Import `datasets` (which pulls in pyarrow) BEFORE torch: on Windows, torch
# 2.6.0+cu124 loaded first corrupts the DLL-load order so a later
# `import pyarrow.dataset` access-violation-crashes the process (exit 139).
# This module is the first in both the eval and analysis import chains to touch
# either library, so ordering it here fixes both entrypoints (mirrors the lazy
# `datasets` import in chartqa/models/qwen_vl_chat.py).
from datasets import load_dataset
from torch.utils.data import Dataset

from chartqa.constants import DATASET_NAME


class ChartQADataset(Dataset):
    """ChartQA wrapped as a torch Dataset yielding raw samples.

    Each item is a dict: {"image": PIL.Image, "question": str, "answer": str}.
    Available splits: "train", "val", "test".
    """

    def __init__(self, split: str = "train"):
        self.dataset = load_dataset(DATASET_NAME, split=split)

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _to_text(value) -> str:
        # The answer is stored as a single-element list.
        if isinstance(value, (list, tuple)):
            return str(value[0]) if value else ""
        return str(value)

    def __getitem__(self, idx: int) -> dict:
        row = self.dataset[idx]
        return {
            "image": row["image"],
            "question": self._to_text(row["query"]),
            "answer": self._to_text(row["label"]),
        }
