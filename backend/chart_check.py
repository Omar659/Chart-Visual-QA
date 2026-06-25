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

# Optional path to the Tesseract engine for non-PATH installs (Windows, pinned
# container path). Empty -> rely on PATH (the norm on Linux/containers).
_TESSERACT_CMD = os.environ.get("TESSERACT_CMD", "")

# Zero-shot label set, split into a "chart" group and a "non-chart" group.
# CLIP softmaxes over ALL labels; P(chart) is the summed mass on the chart group.
#
# The chart prompts deliberately stress DATA — numeric values plus labeled axes,
# a legend, or labeled segments — because a chart is only a chart if it encodes
# values. The non-chart group includes "a diagram/figure with no data" so images
# that look chart-like but carry no values (flowcharts, schematics, geometric
# figures, generic illustrations) are pulled toward non-chart.
_CHART_LABELS = [
    "a bar chart with labeled axes and numeric values",
    "a line graph showing data values over time",
    "a pie chart with labeled segments and percentages",
    "a scatter plot of data points with labeled axes",
    "a dashboard with multiple bar, line, or pie charts",
    "several plotted charts or graphs shown together",
]
_NONCHART_LABELS = [
    "a photograph",
    "a diagram, schematic, or flowchart with no data values",
    "a drawing, illustration, or geometric figure without data",
    "a screenshot of text, a table of numbers, or a document",
]
_CLIP_LABELS = _CHART_LABELS + _NONCHART_LABELS

# "Has data values" gate. CLIP recognises chart *structure* (axes, frames, even
# chart-shaped icons) but can't tell whether real DATA is present. A real chart
# shows numbers — axis ticks, data labels, percentages; an infographic of
# chart-type icons, an empty frame, or a labels-only diagram has none. So a chart
# must be CLIP-chart AND contain numeric data, read via OCR. This enforces the
# rule "no data -> not a chart" no matter how chart-like the image looks or how
# many panels it has. Requires the Tesseract binary (see requirements/README);
# fails open if OCR is unavailable.
_MIN_DATA_DIGITS = int(os.environ.get("CHART_MIN_DATA_DIGITS", "2"))  # min numeric chars for a real chart

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
    the summed softmax probability over the chart group (``_CHART_LABELS``).
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
        return float(probs[: len(_CHART_LABELS)].sum())  # P(chart) over chart labels
    except Exception:
        return None


# --- Pixel heuristic stage (fallback) ------------------------------------

_SAMPLE_SIZE = int(os.environ.get("CHART_SAMPLE_SIZE", "128"))            # downscale square before analysis
_MIN_BACKGROUND_RATIO = float(os.environ.get("CHART_MIN_BACKGROUND_RATIO", "0.18"))  # flat-background share
_MAX_DISTINCT_COLORS = int(os.environ.get("CHART_MAX_DISTINCT_COLORS", "48"))     # of 512 coarse buckets


def _has_data_values(image_bytes: bytes) -> bool:
    """True if the image contains numeric data values (axis ticks, data labels,
    percentages), read via OCR.

    A real chart shows numbers; an infographic of chart-type icons, an empty
    frame, or a labels-only diagram has none. Small images are upscaled first so
    OCR can resolve small axis numbers.

    Fails OPEN (returns True) if Pillow/Tesseract is unavailable or OCR errors,
    so it never vetoes on its own failure — the CLIP verdict stands in that case.
    """
    try:
        import pytesseract
        from PIL import Image

        # Optional override for non-PATH installs (e.g. Windows, or a pinned path
        # in a container). On Linux/containers tesseract is usually on PATH already.
        if _TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        return True
    try:
        if min(img.size) < 600:  # upscale small images so OCR can read numbers
            scale = 600 // min(img.size) + 1
            img = img.resize((img.width * scale, img.height * scale))
        text = pytesseract.image_to_string(img)
    except Exception:
        return True
    return sum(ch.isdigit() for ch in text) >= _MIN_DATA_DIGITS


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
        # Chart iff CLIP says chart AND the image shows real numeric data.
        is_chart = prob >= _CLIP_THRESHOLD and _has_data_values(image_bytes)
        return is_chart, round(prob, 3)
    return _heuristic_chart(image_bytes)
