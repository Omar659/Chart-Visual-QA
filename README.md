# Chart-Visual-QA

Ask text questions about an image (e.g. a chart) using **Qwen3-VL-4B-Instruct**.

The model runs **locally via 🤗 Transformers** (no server/Docker). A small wrapper
class, `QwenVLChat`, loads the model once and answers an image + question through a
`chat()` method. A `ChartQADataset` wrapper exposes the
[ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) dataset as a torch
`Dataset`.

## Project structure

| Path | Purpose |
| --- | --- |
| `model/qwen_vl_chat.py` | `QwenVLChat` — loads Qwen3-VL and answers an image + question. |
| `data/chartqa_dataset.py` | `ChartQADataset` — ChartQA as a torch `Dataset` yielding `{image, question, answer}`. |

## Requirements

- **Python 3.10+**
- An **NVIDIA GPU with CUDA** (developed on an RTX 4090 / 24 GB; WSL2 works).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch transformers accelerate pillow requests datasets
```

## Usage

### Ask a question about an image

```python
from model.qwen_vl_chat import QwenVLChat

model = QwenVLChat()  # downloads Qwen/Qwen3-VL-4B-Instruct on first run (~8 GB)

answer = model.chat(
    image="chart.png",                       # URL, local path, or PIL.Image
    text="What is the highest value in this chart?",
    system_prompt="You are a precise data analyst.",  # optional
)
print(answer)
```

`QwenVLChat.chat`:

```python
chat(image, text, system_prompt=None, max_new_tokens=128) -> str
```

Constructor defaults: `QwenVLChat(model_name="Qwen/Qwen3-VL-4B-Instruct",
dtype="auto", device_map="auto", attn_implementation="sdpa")`.

### Load the ChartQA dataset

```python
from data.chartqa_dataset import ChartQADataset

train = ChartQADataset(split="train")   # splits: "train", "val", "test"
print(len(train))
sample = train[0]                       # {"image": PIL.Image, "question": str, "answer": str}
```
