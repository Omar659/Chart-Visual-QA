"""Is this image a chart?  Two-stage gate.

Stage 1 (preferred): a CLIP zero-shot classifier. We embed the image and a set
of "chart" vs "not a chart" text prompts and see which group the image is closer
to. This is far more robust than pixel statistics — it recognises charts by
content, not by "has a white background".

Stage 2 (fallback): a cheap pixel heuristic (flat-background ratio + limited
color palette). Used only when CLIP can't run — torch/transformers not
installed, the model can't be downloaded, or inference fails. So the gate keeps
working (degraded) on a machine without the ML deps.

The model team can later replace the CLIP stage with their own fine-tuned
classifier; the contract is unchanged: ``looks_like_chart(bytes) -> (bool, float)``.

Fails open everywhere: if nothing can decide, we assume it IS a chart
(confidence 1.0) so we never show a false "unreliable" warning.
"""

from __future__ import annotations

import io
import os
from collections import Counter

# --- CLIP zero-shot stage ------------------------------------------------

# Both configurable via env so the model and cutoff can be tuned without code
# changes. Defaults per the project doc.
_CLIP_MODEL = os.environ.get("CHART_CLIP_MODEL", "openai/clip-vit-base-patch32")
_CLIP_THRESHOLD = float(os.environ.get("CHART_CLIP_THRESHOLD", "0.5"))

# Zero-shot label set. Index 0 is the "chart" class; the rest are what a
# non-chart upload usually is. CLIP softmaxes over all four labels and we read
# off the probability mass on the chart label.
_CLIP_LABELS = [
    "a chart, graph, or data plot",
    "a photograph",
    "a screenshot of text or a document",
    "a drawing or illustration",
]

# Lazy singleton: None = not tried yet, False = unavailable, else (model, proc, torch).
_clip = None


def _load_clip():
    """Load CLIP once. Returns (model, processor, torch) or None if unavailable."""
    global _clip
    if _clip is not None:
        return _clip or None
    try:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        model = CLIPModel.from_pretrained(_CLIP_MODEL)
        model.eval()
        processor = CLIPProcessor.from_pretrained(_CLIP_MODEL)
        _clip = (model, processor, torch)
        return _clip
    except Exception:
        _clip = False  # don't retry on every request
        return None


def _clip_chart_prob(image_bytes: bytes):
    """Return P(image is a chart) from CLIP zero-shot, or None if CLIP can't run.

    Reads the image, compares it against ``_CLIP_LABELS`` with CLIP, and returns
    the softmax probability on the first (chart) label.
    """
    clip = _load_clip()
    if clip is None:
        return None
    model, processor, torch = clip
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        inputs = processor(text=_CLIP_LABELS, images=img, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits_per_image  # (1, n_labels)
        probs = logits.softmax(dim=1)[0]
        return float(probs[0])  # P(chart) — index 0 is the chart label
    except Exception:
        return None


# --- Pixel heuristic stage (fallback) ------------------------------------

_SAMPLE_SIZE = 128            # downscale to this square before analysis
_MIN_BACKGROUND_RATIO = 0.18  # charts usually have a big flat background
_MAX_DISTINCT_COLORS = 48     # of 512 coarse buckets; photos blow past this


def _heuristic_chart(image_bytes: bytes) -> tuple[bool, float]:
    """Cheap pixel-stats verdict: flat background + limited palette => chart."""
    try:
        from PIL import Image
    except Exception:
        return True, 1.0  # dependency missing -> never warn

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((_SAMPLE_SIZE, _SAMPLE_SIZE))
    except Exception:
        return True, 1.0  # undecodable -> let the model decide

    # Quantize to the top 3 bits per channel so near-identical shades collapse.
    quant = [(r >> 5, g >> 5, b >> 5) for (r, g, b) in img.getdata()]
    total = len(quant)
    counts = Counter(quant)

    dominant_share = counts.most_common(1)[0][1] / total
    distinct = len(counts)

    is_chart = dominant_share >= _MIN_BACKGROUND_RATIO and distinct <= _MAX_DISTINCT_COLORS
    palette_score = max(0.0, 1.0 - distinct / 512)
    confidence = round((dominant_share + palette_score) / 2, 3)
    return is_chart, confidence


# --- Public API ----------------------------------------------------------

def looks_like_chart(image_bytes: bytes) -> tuple[bool, float]:
    """Return ``(is_chart, confidence)``.

    Tries CLIP zero-shot first; falls back to the pixel heuristic if CLIP is
    unavailable. ``confidence`` is a rough 0..1 score for logging; the webapp
    only needs the boolean.
    """
    prob = _clip_chart_prob(image_bytes)
    if prob is not None:
        return prob >= _CLIP_THRESHOLD, round(prob, 3)
    return _heuristic_chart(image_bytes)
