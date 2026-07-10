"""Upload safety: validate + re-encode an uploaded image to strip hidden payloads.

An uploaded file can carry a valid image *plus* trailing or embedded junk (polyglot
files, metadata payloads, appended scripts). Decoding with PIL and re-encoding from the
pixel data produces a clean image containing only the pixels — no original file structure
survives. It also yields a canonical byte string, which the answer cache keys on.
"""
from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError


class InvalidImage(ValueError):
    """The uploaded bytes are not a decodable image."""


def sanitize_image(image_bytes: bytes) -> bytes:
    """Return clean PNG bytes re-encoded from the decoded pixels, or raise InvalidImage.

    ``convert("RGB")`` forces a full decode, so a corrupt/truncated image raises here
    (rejected) rather than reaching the model; re-saving to PNG drops everything that
    wasn't pixel data.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImage(str(exc)) from exc
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()
