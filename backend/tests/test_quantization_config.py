"""Unit tests for `build_quantization_config` (qwen_vl_chat.py).

These exercise the REAL function at its real boundary — no monkeypatching of the
unit under test. `bitsandbytes` is not installed in `backend/.venv`, so the
"quantization requested but the dep is missing" path is genuinely taken here
(fail-loud, not fail-open). Importing `qwen_vl_chat` pulls in torch/transformers,
which the backend venv has; no model is downloaded (the function never loads
weights).
"""

import importlib.util

import pytest
from transformers import BitsAndBytesConfig

from qwen_vl_chat import build_quantization_config

_HAS_BNB = importlib.util.find_spec("bitsandbytes") is not None


@pytest.mark.parametrize("value", ["none", "", "NONE", " none ", " NONE ", None])
def test_none_like_modes_return_none(value):
    # None, empty, and any case/whitespace variant of "none" mean full precision.
    assert build_quantization_config(value) is None


@pytest.mark.parametrize("value", ["fp4", "16bit", "int4", "4-bit", "true", "yes"])
def test_unknown_mode_raises_value_error(value):
    with pytest.raises(ValueError) as excinfo:
        build_quantization_config(value)
    # The message echoes the offending value and lists the accepted modes.
    assert repr(value) in str(excinfo.value)
    assert "4bit" in str(excinfo.value)


@pytest.mark.parametrize("mode", ["4bit", "8bit", " 4BIT ", "8Bit"])
@pytest.mark.skipif(_HAS_BNB, reason="bitsandbytes installed; RuntimeError path not reachable")
def test_quant_requested_without_bitsandbytes_raises_runtime_error(mode):
    # bitsandbytes genuinely absent -> fail loudly with an actionable install hint.
    with pytest.raises(RuntimeError) as excinfo:
        build_quantization_config(mode)
    message = str(excinfo.value)
    assert "bitsandbytes" in message
    assert "pip install bitsandbytes" in message


@pytest.mark.parametrize("mode", ["4bit", "8bit"])
@pytest.mark.skipif(not _HAS_BNB, reason="bitsandbytes not installed; config cannot be built")
def test_quant_modes_build_config_when_bitsandbytes_present(mode):
    # When the dep is present the requested mode yields a real BitsAndBytesConfig.
    config = build_quantization_config(mode)
    assert isinstance(config, BitsAndBytesConfig)
    if mode == "4bit":
        assert config.load_in_4bit is True
        assert config.bnb_4bit_quant_type == "nf4"
    else:
        assert config.load_in_8bit is True
