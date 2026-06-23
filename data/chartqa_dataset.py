from torch.utils.data import Dataset
from datasets import load_dataset

# HuggingFaceM4/ChartQA ships train/val/test with columns:
#   image, query (question), label (answer, stored as a single-element list),
#   human_or_machine.
DATASET_NAME = "HuggingFaceM4/ChartQA"


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
