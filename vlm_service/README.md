# vlm_service — GPU VLM inference service (Qwen3-VL-8B)

The production serving path for the real model. The Flask backend (CPU gatekeeper: guards
+ chart gate) runs the cheap checks, then POSTs the surviving `{image, question}` here.
This is a **separate HTTP service** so the 8B VLM lives on a GPU box (**scale-to-zero,
`min=0`**) — it does not fit the 6 GB 4050 (roadmap §B2), so it runs on a cloud GPU or a
big local GPU, never in the backend's CPU-only image.

```
 Frontend ──▶ Backend (CPU: guards + chart gate) ──POST /predict──▶ vlm_service (GPU)
                    │  (VLM_URL set → remote)                              │ Qwen3-VL-8B
                    └─ VLM_URL empty → in-process (dev only, big GPU)      │  (+ LoRA)
```

## Contract

```
POST /predict   {"image": "<base64 PNG/JPEG>", "question": "..."}  →  {"answer": "..."}
GET  /health    →  {"status": "ok", "model": "<id>", "adapter": "<path|null>"}
```

Matches `backend/model_adapter._predict_remote`. The service owns the generation config
(answer suffix, `max_new_tokens`, `Answer:` stripping), so both serving modes return
identical answers.

## Run it

```bash
# 1) Install the wrapper (single source of truth) + this service's serving deps.
pip install -e "./modeling[quant]"        # torch/transformers/peft/accelerate + bitsandbytes
pip install -r vlm_service/requirements.txt

# 2) Config (same QWEN_* keys as the backend real path). Example for a 16 GB cloud GPU:
export QWEN_MODEL_ID=Qwen/Qwen3-VL-8B-Instruct
export QWEN_ADAPTER_PATH=modeling/checkpoints/qwen3vl-lora-final2   # '' = base model
export QWEN_QUANTIZATION=4bit                                       # fits ~6-7 GB
export QWEN_MAX_NEW_TOKENS=64
export QWEN_ANSWER_SUFFIX=" Please answer directly."

# 3) Serve. Dev: `python vlm_service/server.py`. Prod (Linux):
gunicorn -w 1 -t 300 -b 0.0.0.0:8001 --chdir vlm_service server:app
#   -w 1  → one worker = one model copy in VRAM;  -t 300 → long timeout for cold generation.
```

Then point the backend at it: set `VLM_URL=http://<gpu-host>:8001/predict` in the backend's
`.env` and restart the backend. Leave `VLM_URL` empty to run in-process instead.

## Quick test (no backend)

```bash
python - <<'PY'
import base64, requests
img = base64.b64encode(open("dataset/example.png","rb").read()).decode()
r = requests.post("http://localhost:8001/predict",
                  json={"image": img, "question": "What is the highest value?"}, timeout=300)
print(r.json())
PY
```

## Using a free cloud GPU for a demo (Kaggle/Colab + tunnel)

For a one-off demo/screenshot with the accurate model before paying for a box: run this
service inside a Kaggle/Colab GPU notebook, expose it with a tunnel (e.g. `cloudflared`
or `ngrok`), and set the local backend's `VLM_URL` to the tunnel URL. Not production
(notebooks are ephemeral and not meant for serving) — fine for a screenshot.

## Notes / next (Phase 3.1 remainder)

- **Not yet containerized.** A GPU Dockerfile + a `docker-compose` service with
  `deploy.resources.reservations.devices: [{capabilities: [gpu]}]` and `min=0` autoscaling
  is the remaining 3.1 infra work (needs a CUDA host to build/test).
- **vLLM alternative.** For higher throughput, vLLM can serve Qwen3-VL + the LoRA adapter
  with an OpenAI-compatible endpoint (no merge needed). This Flask wrapper is the simple,
  controllable option that reuses our tested `QwenVLChat`; swap later if throughput demands.
