# Running the quantization experiments on Kaggle (free T4)

Step-by-step guide to run the **Phase 1.5 Stage B1/C1** accuracy study — Qwen3-VL-8B at
4-bit (and optionally 8-bit), zero-shot vs LoRA fine-tuned — on a **free Kaggle T4 (16 GB)**.
The notebook (`kaggle_quant_eval.ipynb`) holds all the code; this guide covers the manual
Kaggle setup around it.

> **Why Kaggle and not the 4050?** Accuracy is hardware-independent, so the T4 number equals
> what the 4050 would produce. The 4050 (6 GB) can't *hold* Qwen-8B at 4-bit (roadmap §B2),
> but the T4 fits it comfortably with no CPU offload. Free tier = ~30 GPU-h/week.

---

## What you'll produce

A comparison table (auto-built by the last cell + logged to MLflow):

| Config              | Relaxed | Exact | Peak VRAM | p50/p95 latency |
| ------------------- | ------- | ----- | --------- | --------------- |
| 4-bit zero-shot     | ?       | ?     | ?         | ?               |
| 4-bit fine-tuned    | ?       | ?     | ?         | ?               |

Bring those numbers back and we diff them vs the committed **bf16 reference** (zero-shot
84.60% / fine-tuned 86.08% relaxed) to fill the roadmap table.

---

## Step 1 — Build the code dataset (once)

Kaggle notebooks read code/data from **Datasets**. Upload the repo's `modeling/` folder:

1. On your machine, make sure the LoRA adapter is present on disk (not an LFS pointer):
   `modeling/checkpoints/qwen3vl-lora-final2/adapter_model.safetensors` should be a real
   file (tens of MB), not a ~130-byte pointer. If it's a pointer, run `git lfs pull` first.
2. Zip the `modeling/` folder → `modeling.zip` (or you can drag the folder in the uploader).
3. kaggle.com → **Datasets → New Dataset** → upload `modeling.zip` (Kaggle auto-unzips).
   Name it e.g. **`chartqa-modeling`**. Create.

> Big/optional: you can skip the adapter to run **zero-shot only** — the notebook detects a
> missing adapter and just skips the fine-tuned config. But include it to get both numbers.

## Step 2 — Create the notebook

1. kaggle.com → **Code → New Notebook** → **File → Import Notebook** → upload
   `kaggle_quant_eval.ipynb`.
2. Right sidebar:
   - **Accelerator → GPU T4 x1**
   - **Internet → On** (needed to download Qwen from Hugging Face)
   - **Add Input → Datasets →** add your **`chartqa-modeling`** dataset.

## Step 3 — Point the notebook at your dataset (the one line to edit)

In **cell 2**, set `CODE_DIR` to the folder that **contains `chartqa/`**. After adding the
dataset, check the left **Input** panel for the exact path. Typical cases:

- Uploaded `modeling.zip` (folder unzips as `modeling/`): `/kaggle/input/chartqa-modeling/modeling`
- Uploaded the *contents* of `modeling/` (so `chartqa/` is at the top):
  `/kaggle/input/chartqa-modeling`

The cell asserts `chartqa/` exists under `CODE_DIR` and stops early if the path is wrong.

## Step 4 — (optional) Hugging Face token

Faster, rate-limit-free downloads: **Add-ons → Secrets → Add** a secret named **`HF_TOKEN`**
with your HF token. The notebook picks it up automatically. Skippable — downloads still work.

## Step 5 — Smoke test first (cell 4)

Run cells 1→4 in order. **Cell 4** is a 50-sample zero-shot 4-bit run — it also triggers the
one-time **~17.5 GB base-model download** (several minutes on first run). Confirm it prints
running accuracy and a final `... relaxed accuracy ...` line **before** launching the long
runs. This is the cheap "does the real path work on the T4" gate.

## Step 6 — Full runs (cell 5)

Cell 5 runs the full **2500-sample** configs: zero-shot 4-bit, then fine-tuned 4-bit
(relaxed). Budget **~1–2 h per config** on a T4. Notes:

- `relaxed` and `exact` are **separate passes** in the current harness. Run `relaxed` first
  (the headline metric); uncomment the `exact` / `8bit` lines only if you have session time.
- Kaggle sessions run up to ~9–12 h and stop on idle — keep the tab alive, or use
  **Save Version → Save & Run All (Commit)** to run it headless in the background.

## Step 7 — Get the results (cell 6)

Cell 6 prints the roll-up table and zips the tracking dir to `mlruns.zip`. Download it from
the **Output** tab (or **Data → Output**). Send me the printed table (or the zip) and I'll
fill the roadmap comparison table + Δ-vs-bf16.

---

## Troubleshooting

| Symptom | Fix |
| ------- | --- |
| `chartqa/ not found under <CODE_DIR>` | Wrong path — check the Input panel; `CODE_DIR` must contain `chartqa/` (Step 3). |
| `torch.cuda.is_available() == False` | Accelerator isn't GPU — set **GPU T4 x1** and restart the session. |
| Download fails / very slow | **Internet = On**; add the `HF_TOKEN` secret (Step 4). |
| `Qwen3VLForConditionalGeneration` import/attr error | transformers too old — cell 1 does `-U`; if it persists, pin e.g. `pip install -U "transformers==5.12.1"`. |
| Disk full during download | HF cache is already on `/tmp` (cell 3); don't move it to `/kaggle/working` (20 GB output quota). |
| Fine-tuned run skipped | Adapter not in the dataset — include `checkpoints/qwen3vl-lora-final2` (Step 1). |

## Cost / hygiene

- **Free T4 is enough for the whole accuracy study** — no paid GPU needed here.
- Stop the session when idle (Kaggle bills GPU-hours against your weekly quota, not money,
  but the quota is finite). Don't leave it running after the runs finish.
- `--limit` smoke test before every new full config (already done in cell 4).
