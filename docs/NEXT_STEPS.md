\# Next Steps — Webapp Team (Victor & Min)

> A focused working plan for the two of us. The model (VLM, fine-tuning) stays with
> Susanne & Omar; everything here is webapp/guard work, behind the stable
> `run_inference` / `/api/ask` seam. See [PLAN.md](PLAN.md) and [ROBUSTNESS.md](ROBUSTNESS.md).

## Where we are

- **Layer 1** (cheap rules + chart heuristic) — ✅ Min's work, merged into `feat/webapp`.
- **Layer 2** (toxicity / prompt-injection / PII encoders) — ✅ `backend/guard.py`, on branch
  `feat/guard-layer2` (PR open into `feat/webapp`). Lazy + fail-open, real models opt-in.
- **Next, in parallel:**
  - **Task A — Min:** upgrade the chart detector (heuristic → **CLIP zero-shot**).
  - **Task B — Victor:** add **Layer 3** input boundary filter (**Llama Guard 3 1B**) —
    full plan in [TASK_B_LAYER3.md](TASK_B_LAYER3.md).
- Both are **open source / local / zero per-token cost**. Both follow the same house style:
  one branch per task → PR into `feat/webapp`; **lazy load + fail-open**; env-config; tests
  that pass *without* the heavy models installed (monkeypatch).

---

## Task A — Min: chart detector, heuristic → CLIP zero-shot

### Why
`backend/chart_check.py` today is a **color-histogram heuristic**: it downscales to 128×128,
quantizes to 512 color buckets, and calls it a chart when one bucket dominates
(`dominant_share ≥ 0.18`) and the palette is small (`distinct ≤ 48`). It's µs-fast and
transparent, but fragile — a photo with a white background, or a chart over a photo, fools it.

**CLIP zero-shot** is a big robustness upgrade with **no training and no dataset**: CLIP
embeds the image and a few text labels and picks the closest. It is *not* a generative VLM —
it's a small image–text encoder (MIT licensed, weights public). ~150 MB, one forward,
milliseconds on the RTX 4050 (or CPU).

### Design (keep the contract, keep the heuristic as fallback)
- **Do not change the public function.** `looks_like_chart(image_bytes) -> (is_chart: bool,
  confidence: float)` stays — so `app.py` needs zero changes.
- Internally: **try CLIP first; if unavailable, fall back to Min's heuristic.** Same
  lazy + fail-open pattern as `guard.py` (`lru_cache` loader returning `None` on ImportError).
- Reuses the **existing** `requirements-guard.txt` deps (torch + transformers) — no new heavy
  dependency. Without those installed, it just uses the heuristic (current behavior).

### Implementation steps
1. Branch off `feat/webapp`: `git switch -c feat/chart-clip`.
2. In `chart_check.py`, **rename** the current body to `_heuristic_chart(image_bytes)` (keep it
   verbatim — it's the fallback).
3. Add a cached, CUDA-aware CLIP loader and a scorer:

   ```python
   import io, os
   from functools import lru_cache

   _CLIP_MODEL = os.environ.get("CHART_CLIP_MODEL", "openai/clip-vit-base-patch32")
   _CLIP_LABELS = [
       "a chart, graph, or data plot",   # index 0 = the "chart" class
       "a photograph",
       "a screenshot of text or a document",
       "a drawing or illustration",
   ]

   @lru_cache(maxsize=1)
   def _load_clip():
       try:
           import torch
           from transformers import CLIPModel, CLIPProcessor
           model = CLIPModel.from_pretrained(_CLIP_MODEL)
           proc = CLIPProcessor.from_pretrained(_CLIP_MODEL)
           device = "cuda" if torch.cuda.is_available() else "cpu"
           return model.to(device).eval(), proc, device
       except Exception:
           return None  # transformers/torch missing -> fall back to heuristic

   def _clip_chart_prob(image_bytes: bytes) -> float | None:
       loaded = _load_clip()
       if loaded is None:
           return None
       try:
           import torch
           from PIL import Image
           model, proc, device = loaded
           img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
           inputs = proc(text=_CLIP_LABELS, images=img,
                         return_tensors="pt", padding=True).to(device)
           with torch.no_grad():
               probs = model(**inputs).logits_per_image.softmax(dim=1)[0]
           return float(probs[0])  # P(chart)
       except Exception:
           return None

   def looks_like_chart(image_bytes: bytes) -> tuple[bool, float]:
       p = _clip_chart_prob(image_bytes)
       if p is not None:
           thr = float(os.environ.get("CHART_CLIP_THRESHOLD", "0.5"))
           return p >= thr, round(p, 3)
       return _heuristic_chart(image_bytes)   # fail-open to the heuristic
   ```

4. Local check (deps already installed in `backend/.venv`):
   ```
   python -c "from chart_check import looks_like_chart, _load_clip; print(_load_clip() is not None)"
   ```
   Then feed it a real chart PNG vs a photo and eyeball the probabilities.

### Tests (no heavy deps needed)
- `monkeypatch` `chart_check._clip_chart_prob` to return `0.9` → `looks_like_chart` returns
  `(True, 0.9)`; `0.1` → `(False, 0.1)`.
- `monkeypatch` it to return `None` → falls back to `_heuristic_chart` (assert it still returns
  a `(bool, float)`).
- Keep one test that the **fallback** path works with `_load_clip` forced to `None`.

### Acceptance criteria
- `looks_like_chart` signature unchanged; `app.py` untouched.
- CLIP used when transformers+torch are present; heuristic used otherwise (no crash, no new
  required dep).
- Threshold + model name are env-configurable (`CHART_CLIP_THRESHOLD`, `CHART_CLIP_MODEL`).
- Tests pass with and without the guard deps installed.
- PR into `feat/webapp`.

> Tip: try **OpenCLIP** (`laion/CLIP-ViT-B-32-laion2B-s34B-b79K`) or **SigLIP**
> (`google/siglip-base-patch16-224`) if the base CLIP mislabels — both are open source and
> often stronger zero-shot. Only the `CHART_CLIP_MODEL` value changes.

### Production & efficiency
- **Precompute the label embeddings once.** The 4 text labels never change — encode them a
  single time and cache the text features; per request only run the **image** encoder + a
  dot-product. Re-encoding the labels every call is the main avoidable cost.
- **Warm-load at boot**, not on the first request, so the first user isn't slow.
- **Cache by image hash** (`sha256(image_bytes)`) — the same upload skips CLIP entirely; share
  the key with the answer cache.
- fp16 on GPU; CLIP-B/32 on CPU is ~tens of ms — fine if the guard runs on a CPU-only box.
- Later if needed: ONNX export / `torch.compile` for lower latency.

---

## Task B — Victor: Layer 3 (LLM input boundary filter, Llama Guard 3)

Layer 3 is the **input boundary filter** (Trust pillar, notes §6) — a small local **Llama Guard 3 1B** that screens the question for **unsafe content / policy / out-of-scope** (jailbreak & prompt-injection stay primarily on Layer 2). Runs after Layers 1–2, on the normalized text, **fail-open**.

**Full plan: [TASK_B_LAYER3.md](TASK_B_LAYER3.md).**

---

## Production & efficiency (applies to both)

Design these to be **deployable**, not just to work locally:

- **Warm-load on boot; never load a model in the request path.** Pre-warm so the first user
  isn't the one who pays the cold start.
- **Short-circuit:** cheapest layer first, early-exit on a block.
- **Cache:** image-hash for the chart detector, normalized-question-hash for the text/LLM checks.
- **Gate the expensive path** (the LLM) behind the cheap ones — most requests should never hit it.
- **Run guard models as their own service(s)** (§8 of PLAN), kept warm; heavy deps stay opt-in.
- **Measure:** log per-stage latency (chart / toxicity / injection / PII / LLM) so the guard's
  real cost is visible and we can trim the slow part.
- When adding any new check, state its **latency/cost impact** in the PR.

---

## Shared conventions (both tasks)
- One branch per task → PR into `feat/webapp`.
- **Lazy load + fail-open** everywhere: a missing model/service must never break the app.
- **Env-config** all knobs; keep heavy deps **opt-in** (`requirements-guard.txt`).
- Tests must pass **without** the heavy models (monkeypatch the model/HTTP calls).
- Keep the `run_inference` / `/api/ask` seam untouched — the model team integrates independently.
