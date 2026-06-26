"""Real-model adapter — Qwen3-VL (+ optional LoRA) behind the webapp's seam.

Implements the contract from docs/PLAN.md §5:

    predict(image_bytes: bytes, question: str) -> str

`backend/inference.py` imports this lazily, only when ``USE_MOCK=0``, so mock mode
stays free of heavy ML deps. All knobs come from ``.env`` (no in-code defaults),
matching ``env_config``:

    USE_MOCK=0
    QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct            # base VLM (downloaded from HF)
    QWEN_ADAPTER_PATH=checkpoints/qwen3vl-lora-final2   # LoRA dir; '' = base model
    QWEN_MAX_NEW_TOKENS=64
    QWEN_ANSWER_SUFFIX=" Please answer directly."

Requires a CUDA GPU and the deps in requirements.txt (torch, transformers, peft,
accelerate, pillow). The model is loaded once per process and cached.
"""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from PIL import Image

from env_config import env_int, env_str

# Repo root (backend/ -> repo). The backend process runs with cwd=backend/, so a
# relative QWEN_ADAPTER_PATH is resolved against the repo root, not backend/.
_ROOT = Path(__file__).resolve().parent.parent


def _resolve_adapter_path(raw: str) -> str | None:
    """Resolve QWEN_ADAPTER_PATH ('' -> None; relative -> repo-root-relative)."""
    raw = raw.strip()
    if not raw:
        return None
    p = Path(raw)
    return str(p if p.is_absolute() else _ROOT / p)


@lru_cache(maxsize=1)
def _load_model():
    """Load Qwen3-VL (+ LoRA adapter) once and cache it for the process."""
    from qwen_vl_chat import QwenVLChat  # heavy import kept local to mock mode

    model_id = env_str("QWEN_MODEL_ID")
    adapter_path = _resolve_adapter_path(env_str("QWEN_ADAPTER_PATH"))
    return QwenVLChat(model_name=model_id, adapter_path=adapter_path)


def predict(image_bytes: bytes, question: str) -> str:
    """Return a short answer (1-10 words) for a chart image + question."""
    chat = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    answer = chat.chat(
        image=image,
        text=question.strip() + env_str("QWEN_ANSWER_SUFFIX"),
        max_new_tokens=env_int("QWEN_MAX_NEW_TOKENS"),
    )
    # If the model echoes an "Answer:" prefix (CoT-style prompts), keep the tail.
    return answer.split("Answer:")[-1].strip()
