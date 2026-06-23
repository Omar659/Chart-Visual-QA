# Multimodal and Domain-Specific NLP

**GPU**

**Note.** This page lays out one full version of the project — the goal, a sample input/output, suggested tools, and a step-by-step plan. Treat it as a reference, not a script. Your team can pick a different angle, swap libraries, narrow the scope, or take the project somewhere we did not anticipate. As long as the final deliverable makes sense for the goal, you are on track.

**Task.** Build a small system that combines two modalities (image + text) in a specific domain. Pick something useful to a real audience: receipts where the user uploads a photo and gets back structured info (vendor, total, line items), scientific charts where the user uploads a figure and the system describes the trend, or product photos where the system writes a short marketing blurb. Test on ~50 examples and evaluate the structured fields or text output against a small hand-annotated ground truth. **You could use** a vision-language model (`Qwen2-VL`, `LLaVA`, `Florence-2`) via HuggingFace `transformers`, an OCR library if useful (`tesseract`, `easyocr`, `docling`), and any LLM for the post-processing step. **Going further (optional).** Build a Streamlit UI for the chosen use case, or extend to a third modality (audio for receipts read aloud, for example).

**Resources:** Colab T4 or 8GB GPU recommended for the vision-language stage.

## What you'll build

Build a Visual Question Answering (VQA) system: given an image and a natural-language question about it, the system returns a short answer. The team will combine a pretrained vision-language model (BLIP-2 or LLaVA) with a small fine-tune on a domain-specific QA dataset (e.g., chart QA, document QA, or biomedical QA — pick one). Compare zero-shot vs fine-tuned on the same eval set and report accuracy plus error patterns.

## What goes in, what comes out

### Input

An image and a natural-language question. For example: a chart and "What was revenue in 2024?" or a medical image and "Which lobe shows abnormal opacity?"

### Output

A short text answer (1–10 words typically). Plus a comparison table of zero-shot vs fine-tuned accuracy.

A few rows from ChartQA

```
[
  {
    "image": "charts/2017-bar-revenue.png",
    "question": "What was the revenue in 2016?",
    "answer": "4.2B"
  },
  {
    "image": "charts/2017-bar-revenue.png",
    "question": "In which year was the revenue highest?",
    "answer": "2018"
  },
  {
    "image": "charts/europe-energy-pie.png",
    "question": "What share of energy came from renewables?",
    "answer": "37%"
  }
]
```

Comparison on the ChartQA val set

```
Approach                      Exact-match  Numeric (±5%)
--------------------------    -----------  -------------
BLIP-2 zero-shot                 0.42          0.55
BLIP-2 + LoRA fine-tune          0.61          0.74

Per question type (fine-tuned):
  Read a single value           0.81
  Compare two values            0.66
  Compute a sum / difference    0.42  ← still hard
  Identify a trend              0.71
```

## Datasets

### Pick one domain-specific VQA dataset [↗](https://huggingface.co/datasets/lmms-lab/ChartQA)

Pick whichever domain interests the team: ChartQA (chart reasoning), DocVQA (document understanding), or PathVQA (biomedical). Each has its own quirks and metric.

**How to get it:** `from datasets import load_dataset; ds = load_dataset("lmms-lab/ChartQA")` or similar for the chosen domain.

**License:** Varies by dataset; check the card.

## Tools you'll need

These are suggestions, not requirements. If your team is more comfortable with a different library, model, or framework that achieves the same goal, use it — and briefly explain the choice in your README.

**Python:** Python 3.10 or newer. **Compute:** A 16 GB GPU is comfortable. Load BLIP-2 in 8-bit; LoRA fine-tuning needs another few GB on top.

Vision-language model

* `transformers` — Provides BLIP-2, LLaVA, and the image processors.
* `torchvision` — Image transforms.
* `pillow` — Image I/O.
* `bitsandbytes` — 8-bit loading so you can fit the model + LoRA on 16 GB VRAM.

Fine-tuning

* `peft` — LoRA on the vision-language model. Cheaper than full fine-tune.
* `accelerate` — Device placement and mixed precision.

Evaluation

* `evaluate` — Exact-match and other simple metrics.
* `datasets` — Streams the QA pairs.

## How to approach it

One reasonable path through the project. Specific tools (UMAP, HDBSCAN, BERTopic, etc.) are examples — feel free to swap them for alternatives you know better.

1. **Pick a domain.** Choose one of ChartQA, DocVQA, PathVQA, or a custom domain. Different domains demand different reasoning.
2. **Load the data.** Inspect the question types. Pick the metric that matches: exact match for short answers, numeric tolerance for numbers, ANLS for documents.
3. **Zero-shot baseline.** Load `Salesforce/blip2-opt-2.7b` (or `llava-hf/llava-1.5-7b-hf`) in 8-bit. For each val example, build the prompt "Question: ... Answer:" and generate. Record accuracy.
4. **LoRA fine-tune.** Attach LoRA adapters to the language-model side of the vision-language model. Fine-tune for 1–2 epochs on the train set.
5. **Generate on val.** With the fine-tuned model. Record accuracy.
6. **Break down by question type.** Classify val questions into categories (single value, comparison, calculation, trend). Report per-type accuracy for both systems.
7. **Error analysis.** Pick 20 wrong answers from the fine-tuned model. Group by failure mode.

## What to deliver

* A notebook with both zero-shot and fine-tuned VQA pipelines.
* A results table with overall and per-question-type accuracy.
* An error analysis: 20 failure cases categorised by type (misread chart axis, wrong arithmetic, OCR error, etc).
* A short README explaining the domain choice and what the team learned.
