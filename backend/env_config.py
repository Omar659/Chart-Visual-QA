"""Required-env-var helpers — config lives in .env, not in code defaults.

Backend modules read their config through these so there is a single source of
truth (.env / .env.example) and no value is silently duplicated in code. A
missing key fails fast at import with a clear message instead of a raw KeyError.

The app loads .env at startup (backend/app.py -> python-dotenv); tests load
.env.example (backend/tests/conftest.py) so the full key set is always present.
"""

from __future__ import annotations

import os

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _require(name: str) -> str:
    val = os.environ.get(name)
    if val is None:
        raise RuntimeError(
            f"Missing required env var {name!r}. Define it in .env "
            f"(copy .env.example). Backend config has no in-code defaults."
        )
    return val


def env_str(name: str) -> str:
    """Required string. May be empty (e.g. TESSERACT_CMD='' means 'use PATH')."""
    return _require(name)


def env_int(name: str) -> int:
    return int(_require(name))


def env_float(name: str) -> float:
    return float(_require(name))


def env_bool(name: str) -> bool:
    val = _require(name).strip().lower()
    if val in _TRUTHY:
        return True
    if val in _FALSY:
        return False
    raise RuntimeError(f"Env var {name!r}={val!r} is not a boolean (use 1/0, true/false).")
