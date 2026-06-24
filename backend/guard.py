"""Layer-2 input guard: small local classifiers (see docs/PLAN.md §6).

Three independent signals, each its own small encoder model:
  - toxicity        (detoxify / unitary toxic-bert)
  - prompt injection (protectai deberta-v3 prompt-injection)
  - PII             (Microsoft Presidio)

Design constraints:
  * **Lazy + fail-open.** Each detector imports its (heavy) ML dependency only on
    first use and returns ``None`` if the dependency or model isn't available, so
    the backend still boots and serves in dev without torch/transformers/presidio.
    Install the real models with:  pip install -r backend/requirements-guard.txt
  * **Detectors are module-level callables** so tests can monkeypatch them with no
    ML deps installed.
  * Runs AFTER Layer 1 (cheap rules in app.py), BEFORE the model. The guard never
    sees an image here — it screens the question text.

Contract: ``guard(question) -> GuardResult``. Fail-closed for safety categories
(toxic / prompt_injection); PII blocks only on high-risk entity types.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

# --- Tunables (env-overridable) ---------------------------------------------
TOXICITY_THRESHOLD = float(os.environ.get("GUARD_TOXICITY_THRESHOLD", "0.7"))
INJECTION_THRESHOLD = float(os.environ.get("GUARD_INJECTION_THRESHOLD", "0.8"))
PII_SCORE_THRESHOLD = float(os.environ.get("GUARD_PII_THRESHOLD", "0.6"))
GUARD_ENABLED = os.environ.get("GUARD_ENABLED", "1").lower() not in ("0", "false", "no", "off")

# Block only on high-risk identifiers; ignore PERSON/LOCATION/ORG/DATE to avoid
# false positives on ordinary chart questions ("What was John's revenue?").
_SENSITIVE_PII = {
    "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD", "IBAN_CODE", "CRYPTO",
    "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "IP_ADDRESS", "MEDICAL_LICENSE",
}


@dataclass
class GuardResult:
    allowed: bool
    category: str = "ok"      # "ok" | "toxic" | "prompt_injection" | "pii"
    reason: str = ""          # short, user-safe message


# --- Device selection (use CUDA when available, else CPU) -------------------
@lru_cache(maxsize=1)
def _cuda_device_index() -> int:
    """0 for the first CUDA GPU, -1 for CPU (transformers `device` convention)."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except Exception:  # noqa: BLE001 — torch not installed
        return -1


# --- Model loaders (cached; return None if unavailable) ---------------------
@lru_cache(maxsize=1)
def _load_toxicity():
    try:
        from detoxify import Detoxify
        device = "cuda" if _cuda_device_index() == 0 else "cpu"
        return Detoxify("original", device=device)
    except Exception:  # noqa: BLE001 — dep missing or model download failed
        return None


@lru_cache(maxsize=1)
def _load_injection():
    try:
        from transformers import pipeline
        return pipeline(
            "text-classification",
            model="protectai/deberta-v3-base-prompt-injection-v2",
            truncation=True,
            device=_cuda_device_index(),
        )
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=1)
def _load_pii():
    try:
        from presidio_analyzer import AnalyzerEngine
        return AnalyzerEngine()
    except Exception:  # noqa: BLE001
        return None


# --- Detectors (return a score / entities, or None if unavailable) ----------
def toxicity_score(text: str) -> float | None:
    """P(toxic) in [0,1], or None if the detector isn't available."""
    model = _load_toxicity()
    if model is None:
        return None
    try:
        return float(model.predict(text)["toxicity"])
    except Exception:  # noqa: BLE001
        return None


def injection_score(text: str) -> float | None:
    """P(prompt injection) in [0,1], or None if unavailable."""
    clf = _load_injection()
    if clf is None:
        return None
    try:
        out = clf(text)[0]
        label = str(out.get("label", "")).upper()
        score = float(out.get("score", 0.0))
        # Model returns the winning label + its confidence; convert to P(INJECTION).
        return score if label == "INJECTION" else 1.0 - score
    except Exception:  # noqa: BLE001
        return None


def pii_hits(text: str) -> list[str] | None:
    """High-risk PII entity types found, or None if the detector isn't available."""
    engine = _load_pii()
    if engine is None:
        return None
    try:
        results = engine.analyze(text=text, language="en")
        return sorted({
            r.entity_type for r in results
            if r.entity_type in _SENSITIVE_PII and r.score >= PII_SCORE_THRESHOLD
        })
    except Exception:  # noqa: BLE001
        return None


# --- Orchestrator -----------------------------------------------------------
def guard(question: str) -> GuardResult:
    """Screen the question through the Layer-2 classifiers, then the Layer-3 LLM.

    Returns ``allowed=True`` when nothing fires OR when a detector is unavailable
    (fail-open). Order: toxicity -> prompt injection -> PII -> Layer-3 LLM.
    """
    if not GUARD_ENABLED:
        return GuardResult(True)

    tox = toxicity_score(question)
    if tox is not None and tox >= TOXICITY_THRESHOLD:
        return GuardResult(False, "toxic", "This question looks abusive — please rephrase.")

    inj = injection_score(question)
    if inj is not None and inj >= INJECTION_THRESHOLD:
        return GuardResult(
            False, "prompt_injection",
            "That looks like an attempt to override the assistant's instructions.",
        )

    hits = pii_hits(question)
    if hits:
        return GuardResult(
            False, "pii",
            "Please remove personal data from your question (e.g. " + ", ".join(hits) + ").",
        )

    # --- Layer 3: LLM input boundary filter (Llama Guard) — last, gated, fail-open ---
    # Lazy import avoids a circular import (guard_llm imports GuardResult from here).
    try:
        from guard_llm import llm_classify
        verdict = llm_classify(question)
    except Exception:  # noqa: BLE001
        verdict = None
    if verdict is not None and not verdict.allowed:
        return verdict

    return GuardResult(True)


def is_available() -> dict[str, bool]:
    """Which detectors actually have their model loaded (for /api/health, debugging)."""
    return {
        "toxicity": _load_toxicity() is not None,
        "prompt_injection": _load_injection() is not None,
        "pii": _load_pii() is not None,
    }
