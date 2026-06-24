# Task B — Layer 3: LLM Input Boundary Filter (Llama Guard 3)

> Owner: **Victor**. Branch `feat/guard-layer3` → PR into `feat/webapp`. Follows the shared
> conventions in [NEXT_STEPS.md](NEXT_STEPS.md) (lazy + fail-open, env-config, tests that pass
> without the heavy model). The VLM stays with Susanne & Omar; this is webapp/guard work behind
> the `run_inference` / `/api/ask` seam.

## 1. Why — input boundary filtering (Trust pillar)

Layer 3 is our **input boundary filter**: the fast classification layer *before* the main model
described in
[designing_ai_systems_scale_reliability_trust_notes.md](designing_ai_systems_scale_reliability_trust_notes.md)
Pillar 3 §6 ("Input boundary filtering", with **Llama Guard** as the guard model). Layers 1–2 are
cheap and lexical/encoder-bound; Layer 3 is the **semantic** boundary check and the backstop for
evasion (paraphrase / role-play / novel encodings — see [ROBUSTNESS.md](ROBUSTNESS.md) §1.5).

It runs **only after** Layers 1–2 pass, on the **normalized** question, and **fails open** if the\v
guard service is unreachable. Open source (Meta), local, no per-token cost.

## 2. Scope — only what this project needs

No RAG; input = a short question + a chart image. Layer 3 owns only the gaps:

| Boundary concern (notes §6) | Owner here | Layer 3's role |
| --- | --- | --- |
| **Unsafe content** | Layer 2 (toxicity) + **Layer 3** | **primary** — full hazard taxonomy (Llama Guard S1–S14) |
| **Policy violations** | **Layer 3** | catch-all via the taxonomy + custom categories |
| **Out-of-scope requests** | **Layer 3** | "is this a chart/figure question?" — custom category |
| **Jailbreak / role-play** | Layer 2 (deberta) + **Layer 3** | Layer 2 primary; Llama Guard adds coverage |
| Prompt injection | Layer 2 (deberta) | light backstop only |
| PII | Layer 2 (Presidio) | — not re-done |
| Sensitive data leakage | Layer 2 (PII) | minimal — no secrets / RAG in this app |

(Output-side boundary filtering — checking the model's *answer* — is a separate future item.)

## 3. The model — Llama Guard 3 1B (decided)

**Use Llama Guard 3 1B** (Meta, open weights). It's a purpose-built input/output safety classifier
over the MLCommons hazard taxonomy (S1–S13 + S14 code-interpreter abuse), and it **accepts a custom
taxonomy** in the prompt — so we extend it with an off-topic category for the "is this a chart
question?" check. One model, one call: unsafe content + policy + out-of-scope.

It returns plain text — `safe`, or `unsafe` plus a second line of comma-separated category codes:

```
unsafe
S1,S10
```

We map those codes to our `{unsafe, jailbreak, off_topic}` categories + a user-safe reason.

**Notes / honest caveats**
- **Jailbreak/injection** is primarily **Layer 2's deberta**; Llama Guard adds breadth, not a
  dedicated injection model. If we later want a specialized one, Meta's **Prompt Guard** (86M,
  open source) is the option — beyond the MVP.
- Llama Guard is content-safety-trained; **off-topic is a *custom* category** we add — treat its
  off-topic accuracy as best-effort and tune the category text/threshold.

## 4. Architecture

Small local guard model behind HTTP — **same code dev↔prod, only the URL changes** (PLAN §8):
- **Dev:** Ollama (`http://localhost:11434`) — `ollama pull llama-guard3:1b`.
- **Prod:** vLLM (OpenAI-compatible) via `GUARD_LLM_URL`, kept warm, quantized.

## 5. Implementation steps

1. `git switch -c feat/guard-layer3`; `ollama pull llama-guard3:1b`.
2. Add `requests` to `backend/requirements.txt` (tiny).
3. New module `backend/guard_llm.py`:

   ```python
   import os
   from guard import GuardResult   # reuse the dataclass

   GUARD_LLM_ENABLED = os.environ.get("GUARD_LLM_ENABLED", "0").lower() in ("1","true","yes","on")
   GUARD_LLM_URL = os.environ.get("GUARD_LLM_URL", "http://localhost:11434")
   GUARD_LLM_MODEL = os.environ.get("GUARD_LLM_MODEL", "llama-guard3:1b")
   _TIMEOUT = float(os.environ.get("GUARD_LLM_TIMEOUT", "5"))

   # Llama Guard hazard code -> (our category, user-safe reason). Trim to what matters.
   _CODES = {
       "S1":  ("unsafe",    "This request involves violent content."),
       "S2":  ("unsafe",    "This request involves criminal activity."),
       "S3":  ("unsafe",    "This request involves sexual content."),
       "S9":  ("unsafe",    "This request involves weapons or mass harm."),
       "S10": ("unsafe",    "This request involves hateful content."),
       "S14": ("jailbreak", "That looks like an attempt to abuse the tool."),
       "S99": ("off_topic", "Please ask a question about the chart."),   # our custom category
   }

   def llm_classify(question: str):
       """GuardResult, or None if the guard service is unavailable (fail-open)."""
       if not GUARD_LLM_ENABLED:
           return None
       try:
           import requests
           r = requests.post(
               f"{GUARD_LLM_URL}/api/chat", timeout=_TIMEOUT,
               json={"model": GUARD_LLM_MODEL,
                     "messages": [{"role": "user", "content": question}],
                     "stream": False},
           )
           r.raise_for_status()
           out = r.json()["message"]["content"].strip()
           if out.lower().startswith("safe"):
               return GuardResult(True)
           lines = out.splitlines()
           codes = lines[1].replace(" ", "").upper().split(",") if len(lines) > 1 else []
           for c in codes:
               if c in _CODES:
                   cat, reason = _CODES[c]
                   return GuardResult(False, cat, reason)
           return GuardResult(False, "unsafe", "Blocked by the safety classifier.")
       except Exception:
           return None   # service down / bad output -> fail-open
   ```

   > Adding the **off-topic custom category (S99)** means sending Llama Guard a custom-taxonomy
   > prompt (override Ollama's default template). For the MVP you can ship **safety-only** and add
   > the custom category next; document the exact template in the PR.

4. Wire as the **last** step in `guard.py`'s `guard()` (after Layer 2):

   ```python
   from guard_llm import llm_classify
   ...
       verdict = llm_classify(question)   # None when disabled/unreachable
       if verdict is not None and not verdict.allowed:
           return verdict
       return GuardResult(True)
   ```
   Classify the **normalized** text (same text the model will see — ROBUSTNESS §1.5).

5. Enable with `GUARD_LLM_ENABLED=1` + Ollama running; **off by default** so the app and CI run
   without a guard server.

## 6. Tests (no guard server needed)

- `monkeypatch` `guard_llm.requests.post` to return a fake message:
  - `"safe"` → guard allows;
  - `"unsafe\nS10"` → guard blocks with category `unsafe`;
  - exception / `GUARD_LLM_ENABLED=False` → `llm_classify` returns `None` (fail-open → allowed).
- App-level: a question that passes Layers 1–2 but Llama Guard flags → `200 {blocked, category, reason}`.

## 7. Acceptance criteria

- Fails open when no guard server is reachable (default off; CI green without Ollama).
- Maps Llama Guard codes → our categories; runs **after** Layers 1–2; env-config URL/model/timeout.
- Same `{blocked, category, reason}` contract — no frontend change beyond what's already there.
- Branch `feat/guard-layer3` → PR into `feat/webapp`.

## 8. Production & efficiency

- **The LLM is the only expensive layer — gate it.** Don't call it on every request: run it only
  on the **ambiguous subset** (no cheap layer fired but the question is unusual) or behind a risk
  trigger. Keeps median latency near the cheap-layer cost.
- **Cache verdicts** by `sha256(normalized_question)` — identical questions skip the model.
- **Warm model + small output + strict `timeout` + fail-open** so a slow/down guard never blocks.
- **Prod serving:** vLLM (continuous batching, guided decoding) per PLAN §8, a **quantized** model,
  running as its **own warm service** separate from the VLM (keep the guard warm; let the VLM
  scale-to-zero).

## 9. Nice to have (post-MVP) — tune / fine-tune the malicious-text classifier

- **Cheapest lever first:** Llama Guard takes a **custom taxonomy** with no training — refine the
  category definitions (and the off-topic category) before reaching for a fine-tune.
- **Later:** a small **LoRA** fine-tune focused on **malicious-text recognition** (toxicity /
  prompt-injection / jailbreak) raises recall on real evasion **and** lets a smaller/cheaper model
  match it — the production win on the gated layer; it also cuts false positives.
  - *Toxicity:* **Jigsaw Toxic Comments**, **lmsys/toxic-chat**.
  - *Prompt injection:* **deepset/prompt-injections**.
  - *Jailbreak / role-play:* **jackhhao/jailbreak-classification**, **rubend18/ChatGPT-Jailbreak-Prompts**.
  - *Combined safety corpus:* **allenai/wildguardmix**, **PKU-Alignment/BeaverTails**,
    **nvidia/Aegis-AI-Content-Safety**.
  - LoRA 1–2 epochs; **evaluate on a held-out red-team set** (ROBUSTNESS §1.5); track per-category
    recall/precision — don't chase loss. Borrow Susanne & Omar's setup.
- *Alternative:* fine-tune the **Layer-2 encoders** (toxic-bert / deberta) instead — cheaper to
  train and run for fixed-label malicious-text detection.
