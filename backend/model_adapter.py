"""Real-model adapter — owned by the model team (Susanne & Omar).

This is the single file you need to fill in to plug the fine-tuned VLM into the
webapp. The backend never imports this module while running in mock mode, so it
stays out of the way (and out of the dependency tree) until you're ready.

To go live:
    1. Add your model's runtime deps to requirements.txt (transformers, torch,
       peft, pillow, ...).
    2. Implement ``predict`` below: decode the image bytes, build the prompt,
       run the model, post-process, and return a SHORT string (1-10 words).
    3. Start the backend with USE_MOCK=0 (env var). Nothing else changes.

Contract (must match docs/PLAN.md §5 exactly):
    predict(image_bytes: bytes, question: str) -> str
"""

from __future__ import annotations

# Suggested skeleton (uncomment + adapt when the model lands):
#
# import io
# from functools import lru_cache
# from PIL import Image
#
# @lru_cache(maxsize=1)
# def _load_model():
#     # Load the VLM + LoRA adapter once and cache it for the process.
#     ...
#     return processor, model
#
# def predict(image_bytes: bytes, question: str) -> str:
#     processor, model = _load_model()
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     prompt = f"Question: {question} Answer:"
#     ...  # run inference
#     answer = ...  # decode + post-process
#     return answer.strip()


def predict(image_bytes: bytes, question: str) -> str:
    """Return a short answer for a chart image + question.

    Replace this body with the real model call. See the skeleton above.
    """
    raise NotImplementedError(
        "Real model not wired in yet. Implement predict() in model_adapter.py "
        "and run with USE_MOCK=0 to use it."
    )
