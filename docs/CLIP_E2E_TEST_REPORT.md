# Chart Detector — End-to-End Test Report

**Date:** 2026-06-24
**Branch:** `min`
**Component:** `backend/chart_check.py`
**Tested by:** webapp end-to-end through the frontend proxy

---

## Summary

The chart detector decides whether an uploaded image is a real chart. It runs
for real on every `/api/ask` request (not mocked) and returns `is_chart` and
`chart_confidence` to the UI.

It uses **two gates that must both pass**, plus a fallback:

1. **CLIP zero-shot** — "does this look like a chart?" (classifies by content vs
   a set of chart / non-chart text prompts).
2. **OCR data gate** (`_has_data_values`) — "does it actually contain numeric
   data values?" (reads the image with Tesseract; a real chart shows axis ticks,
   data labels, percentages).
3. **Heuristic fallback** — a pixel heuristic used only if CLIP can't run.

**A chart = looks like a chart AND contains real data.** This enforces the rule
*"no data values → not a chart"*, no matter how chart-like the image looks or
how many panels it has.

**Verdict: ✅ PASS — all real-image cases classify correctly, including an
infographic of chart-type icons that fooled CLIP alone.**

---

## Why two gates

CLIP recognises chart *structure* — axes, frames, even chart-shaped icons — but
can't tell whether real data is present. Two adversarial cases proved this:

- **Empty chart frames** (axes drawn, nothing plotted): CLIP scored 0.98.
- **An infographic of chart-type icons** (pie/bar/line icons with type names, no
  numbers): CLIP scored 0.99.

Both *look* like charts but carry no data. The OCR gate catches them because
they contain **zero digits**, while real charts contain many.

---

## Environment

| Dependency | Version |
| --- | --- |
| torch | 2.12.1 |
| transformers | 5.12.1 |
| Pillow | 11.3.0 |
| pytesseract | 0.3.13 |
| Tesseract (binary) | 5.5.2 |
| Python | 3.14.3 |

| Config (env var) | Value |
| --- | --- |
| `CHART_CLIP_MODEL` | `openai/clip-vit-base-patch32` (default) |
| `CHART_CLIP_THRESHOLD` | `0.5` (default) |
| `_MIN_DATA_DIGITS` | `2` |

Both gates fail open: if Tesseract or torch/transformers are unavailable, that
gate does not veto, so the app never hard-fails on detection.

---

## Test setup

- Requests sent to **`http://localhost:5173/api/ask`** — the Vite proxy, i.e. the
  exact path the browser UI uses (`:5173` → `:5000` → CLIP + OCR).
- Detection rule: `is_chart = P(chart) ≥ 0.5 AND digits_found ≥ 2`.

---

## Results

| # | Input image | CLIP `P(chart)` | has data | `is_chart` | Expected | Result |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Real bar chart | 0.955 | ✓ | `true` | chart | ✅ |
| 2 | Real line chart | 0.997 | ✓ | `true` | chart | ✅ |
| 3 | Dashboard (bar+line+pie) | 0.968 | ✓ | `true` | chart | ✅ |
| 4 | **Infographic of chart-type icons** | **0.994** | **✗** | **`false`** | not chart | ✅ |
| 5 | Empty chart frames (axes, no data) | 0.978 | ✗ | `false` | not chart | ✅ |
| 6 | Multi-panel figure, no data | 0.111 | ✗ | `false` | not chart | ✅ |
| 7 | Single figure, no data | 0.085 | ✗ | `false` | not chart | ✅ |
| 8 | Photograph | 0.023 | ✗ | `false` | not chart | ✅ |
| 9 | Text document / receipt | 0.031 | ✓ | `false` | not chart | ✅ |

Cases 4 and 5 are the key wins: both look like charts to CLIP, but the OCR gate
rejects them because they contain no numeric data.

### Input validation (Layer-1 guard)

| Case | HTTP | Response |
| --- | --- | --- |
| No image uploaded | `400` | `"Please upload an image."` |
| Weak question (`"hi"`) | `400` | `"Please ask a more specific question."` |

---

## Unit tests

`backend/test_chart_check.py` — 7 monkeypatch-based tests that run **without**
torch / transformers / Tesseract (the CLIP and OCR stages are patched, so only
the decision logic is exercised). All pass.

```bash
cd backend && python -m pytest test_chart_check.py
```

Covered: high prob → chart, low prob → not, `None` → heuristic fallback, CLIP
unavailable → still returns `(bool, float)`, threshold inclusive, data gate
vetoes (no data) / allows (data present).

---

## Known limitations

- OCR adds ~0.1–0.3 s per request and requires the Tesseract binary
  (`brew install tesseract`). Without it the data gate fails open (no veto).
- A real chart uploaded at very low resolution may have unreadable numbers and
  could be missed by the OCR gate (false negative). Small images are upscaled to
  mitigate this.
- Robustly distinguishing "a poster of chart icons" from "real charts" is hard;
  the model team's trained classifier can later replace these heuristics behind
  the same `looks_like_chart` contract.

---

## Conclusion

- The detector enforces "no data → not a chart", validated on real images.
- The infographic of chart-type icons (the reported failure) is now correctly
  rejected: CLIP 0.99 but zero data values → not a chart.
- Real charts (single, multi-panel, dashboards) are still accepted.
- Both gates fail open, so the detection step never hard-fails the app.
