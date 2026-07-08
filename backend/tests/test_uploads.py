"""Tests for upload sanitization (real PIL decode + re-encode, no mocking)."""
import io

import pytest
from PIL import Image

from uploads import InvalidImage, sanitize_image

_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _png(color=(255, 0, 0), size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def test_valid_image_reencodes_to_png():
    out = sanitize_image(_png())
    assert out.startswith(_PNG_SIG)
    img = Image.open(io.BytesIO(out))
    img.load()
    assert img.size == (8, 8)


def test_trailing_payload_is_stripped():
    raw = _png() + b"<?php system($_GET[0]); ?>"  # polyglot-style trailing junk
    out = sanitize_image(raw)
    assert b"<?php" not in out                     # re-encode dropped everything non-pixel
    Image.open(io.BytesIO(out)).load()             # still a valid image


def test_jpeg_input_is_accepted():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (0, 128, 255)).save(buf, "JPEG")
    out = sanitize_image(buf.getvalue())
    assert out.startswith(_PNG_SIG)                # normalized to PNG


def test_non_image_raises():
    with pytest.raises(InvalidImage):
        sanitize_image(b"this is definitely not an image")
