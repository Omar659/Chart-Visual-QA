"""Heuristic "is this image a chart?" gate.

PLACEHOLDER: this is a cheap, transparent heuristic so the webapp can warn when
a user uploads something that doesn't look like a chart. The model team can
replace ``looks_like_chart`` with a real classifier later — the return contract
(``(is_chart: bool, confidence: float)``) is what app.py depends on.

The heuristic: charts (bar/line/pie/scatter) tend to have a large flat
background region and a small, limited color palette. Photos and natural images
have many smoothly-varying colors and no single dominant flat area. So we
quantize the image to a coarse palette and look at:
  - dominant_share: fraction of pixels in the single most common color bucket
  - distinct: number of occupied color buckets
An image is "chart-like" when there is a dominant flat region AND the palette
stays small.

Fails open: if Pillow is missing or the bytes can't be decoded, we assume it IS
a chart (confidence 1.0) so we never show a false "unreliable" warning.
"""

from __future__ import annotations

import io
from collections import Counter

# Tuning constants (coarse on purpose; this is a placeholder).
_SAMPLE_SIZE = 128            # downscale to this square before analysis
_MIN_BACKGROUND_RATIO = 0.18  # charts usually have a big flat background
_MAX_DISTINCT_COLORS = 48     # of 512 coarse buckets; photos blow past this


def looks_like_chart(image_bytes: bytes) -> tuple[bool, float]:
    """Return ``(is_chart, confidence)`` for the given image bytes.

    confidence is a rough 0..1 score for logging/debugging; the webapp only
    needs the boolean.
    """
    try:
        from PIL import Image
    except Exception:
        return True, 1.0  # dependency missing -> never warn

    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((_SAMPLE_SIZE, _SAMPLE_SIZE))
    except Exception:
        return True, 1.0  # undecodable -> let the model decide

    # Quantize to the top 3 bits per channel (8*8*8 = 512 buckets) so near-
    # identical shades (anti-aliasing, JPEG noise) collapse together.
    quant = [(r >> 5, g >> 5, b >> 5) for (r, g, b) in img.getdata()]
    total = len(quant)
    counts = Counter(quant)

    dominant_share = counts.most_common(1)[0][1] / total
    distinct = len(counts)

    is_chart = dominant_share >= _MIN_BACKGROUND_RATIO and distinct <= _MAX_DISTINCT_COLORS

    # Confidence: blend "has a flat background" with "palette is limited".
    palette_score = max(0.0, 1.0 - distinct / 512)
    confidence = round((dominant_share + palette_score) / 2, 3)
    return is_chart, confidence
