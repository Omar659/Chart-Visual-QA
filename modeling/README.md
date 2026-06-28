# ChartQA — VLM fine-tuning & evaluation

LoRA fine-tuning, evaluation and error analysis of vision-language models
(**Qwen3-VL-8B** and **BLIP-2 Flan-T5-XL**) on the
[ChartQA](https://huggingface.co/datasets/HuggingFaceM4/ChartQA) dataset. It
produces the LoRA adapters used by the project's webapp backend.

## Project structure

```
chartqa/              importable package (run with `python -m chartqa.<...>`)
├── constants.py        all configuration: model ids, hyperparameters, paths, …
├── data/               ChartQA dataset wrapper + collators
├── models/             Qwen3-VL / BLIP-2 chat wrappers
├── training/           finetune_lora.py
├── evaluation/         evaluate.py + metrics
├── analysis/           error / question analysis and tables
└── viz/                training-loss plotting
scripts/              shell runners (training, eval + compare, results)
checkpoints/          hand-picked best adapters (committed)
checkpoint_train/     training runs: checkpoint-N / -best / -final (gitignored)
outputs/              generated artifacts: errors/, results/, questions/
```

## Setup

Requires **Python 3.10+** and an **NVIDIA GPU** (CUDA; developed on a 4090 / WSL2).

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

The base models and the dataset are downloaded from Hugging Face on first use.

## Usage

Always run from this directory (`modeling/`). There are two ways to run things:

- **`python -m chartqa.<module> [args]`** — a single program; you pass the arguments.
- **`bash scripts/<name>.sh`** — a convenience wrapper that runs several of those
  commands in sequence with preset arguments. Open and edit it to change models,
  metric or checkpoints.

### Fine-tuning

```bash
python -m chartqa.training.finetune_lora --model qwen
```

- `--model qwen|blip2` — which model to fine-tune.
- `--run-name NAME` — optional; defaults to `<model>_lora_<timestamp>`.

Writes the run to `checkpoint_train/<run-name>/` (`checkpoint-N`, `checkpoint-best`,
`checkpoint-final`, live loss plot). The wrapper `scripts/run_trainings.sh` trains
both models back to back.

### Evaluation

```bash
python -m chartqa.evaluation.evaluate --model qwen --metric relaxed --checkpoint ./checkpoints/qwen3vl-lora-final2 --errors-dir outputs/errors/my_run
```

- `--model qwen|blip2` — which model.
- `--checkpoint PATH` — a LoRA adapter directory. **Omit it** to evaluate the base
  (zero-shot) model.
- `--metric relaxed|exact` — scoring (`relaxed` = ChartQA numeric tolerance).
- `--errors-dir DIR` — optional; if set, saves the misclassified chart images +
  an `errors.json` there.
- `--limit N` — optional; evaluate only the first N samples.

The command prints the accuracy to stdout. The wrapper `scripts/run_eval_compare.sh`
chains the full comparison: zero-shot vs LoRA for both models, then the disagreement
set (samples one model got right and the other wrong).

### Analysis

```bash
python -m chartqa.analysis.dump_questions --split test
python -m chartqa.analysis.results_from_errors --errors-dir outputs/errors/my_run
python -m chartqa.analysis.compare_errors --zero-shot-dir DIR_A --lora-dir DIR_B --out-dir DIR_C
```

- `dump_questions` — dump a split's `(id, question)` pairs to JSON.
- `results_from_errors` — per-sample success/failure table from one errors-dir.
- `compare_errors` — disagreement set between a zero-shot and a LoRA errors-dir.

The wrapper `scripts/run_results_from_errors.sh` builds the result tables for all
the standard error dirs at once.

## Notes

- All hyperparameters and paths live in `chartqa/constants.py`.
- Error/result dirs are named `errors_<model>_<zero_shot|lora_finetune>_<exact|relaxed>`.
