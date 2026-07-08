"""Real-model adapter — Qwen3-VL (+ optional LoRA) behind the webapp's seam.

Implements the contract from docs/PLAN.md §5:

    predict(image_bytes: bytes, question: str) -> str

`backend/inference.py` imports this lazily, only when ``USE_MOCK=0``, so mock mode
stays free of heavy ML deps. All knobs come from ``.env`` (no in-code defaults),
matching ``env_config``:

    USE_MOCK=0
    VLM_URL=                                            # empty = in-process; URL = remote service
    VLM_TIMEOUT=120                                     # remote call timeout (survives cold start)
    QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct            # base VLM (downloaded from HF)
    QWEN_ADAPTER_PATH=checkpoints/qwen3vl-lora-final2   # LoRA dir; '' = base model
    QWEN_QUANTIZATION=none                              # none|8bit|4bit (bitsandbytes)
    QWEN_MAX_NEW_TOKENS=64
    QWEN_ANSWER_SUFFIX=" Please answer directly."

Two serving modes:
  * **In-process** (``VLM_URL`` empty): load Qwen3-VL once and generate here. Needs a
    CUDA GPU + the heavy deps (torch, transformers, peft, accelerate). Good for dev on a
    big GPU. Qwen-8B does NOT fit the 6 GB 4050 (roadmap §B2).
  * **Remote** (``VLM_URL`` set): POST ``{image, question}`` to a separate VLM HTTP
    service (the production scale-to-zero GPU path, see ``vlm_service/``). The backend
    then needs neither torch nor the model — only ``requests``.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import requests
from PIL import Image

from env_config import env_float, env_int, env_str

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
    # Opt-in bitsandbytes quantization ('none' = full precision). With 8bit/4bit
    # the wrapper keeps the LoRA adapter attached instead of merging it (a
    # quantized base cannot be merged) and fails loudly if bitsandbytes is missing.
    return QwenVLChat(
        model_name=model_id,
        adapter_path=adapter_path,
        quantization=env_str("QWEN_QUANTIZATION"),
    )


def predict(image_bytes: bytes, question: str) -> str:
    """Return a short answer (1-10 words) for a chart image + question.

    Routes to the remote VLM service when ``VLM_URL`` is set, else runs in-process.
    The frontend/API contract is identical either way.
    """
    url = env_str("VLM_URL").strip()
    if url:
        return _predict_remote(url, image_bytes, question)
    return _predict_local(image_bytes, question)


def _predict_local(image_bytes: bytes, question: str) -> str:
    """In-process inference: load Qwen3-VL once (cached) and generate here."""
    chat = _load_model()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    answer = chat.chat(
        image=image,
        text=question.strip() + env_str("QWEN_ANSWER_SUFFIX"),
        max_new_tokens=env_int("QWEN_MAX_NEW_TOKENS"),
    )
    # If the model echoes an "Answer:" prefix (CoT-style prompts), keep the tail.
    return answer.split("Answer:")[-1].strip()


def _predict_remote(url: str, image_bytes: bytes, question: str) -> str:
    """Delegate to a remote VLM service: POST ``{image, question}`` -> ``{answer}``.

    The service owns the model + generation config (suffix, max_new_tokens, "Answer:"
    stripping), so the contract stays tiny and the backend carries no ML deps. This is
    **not** fail-open: the VLM produces THE answer, so an unreachable/slow service
    surfaces as an error (HTTP 5xx to the client) rather than a fabricated result. The
    timeout is generous to survive a scale-to-zero cold start (mirrors GUARD_LLM_TIMEOUT).
    """
    payload = {
        "image": base64.b64encode(image_bytes).decode("ascii"),
        "question": question.strip(),
    }
    resp = requests.post(url, json=payload, timeout=env_float("VLM_TIMEOUT"))
    resp.raise_for_status()
    return resp.json()["answer"]
