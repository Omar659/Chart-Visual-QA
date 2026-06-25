# Backend: Flask API + CLIP chart detector + Layer-2 encoders + Tesseract OCR.
# Built and run by `python app.py` (production mode) via docker-compose.yml.
FROM python:3.12-slim

# Tesseract OCR engine — the system binary pytesseract shells out to (so the
# "has data values" gate runs without a host install). On PATH here, so the
# backend service sets TESSERACT_CMD="" (see docker-compose.yml).
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU build of torch — the backend runs CLIP on CPU; the GPU is for the vLLM guard.
# Installed first so transformers/detoxify don't pull the much larger CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Layer-2 guard models pull in transformers/presidio/detoxify and (via `-r
# requirements.txt`) the app + CLIP deps. Cached as its own layer.
COPY requirements.txt requirements-guard.txt ./
RUN pip install --no-cache-dir -r requirements-guard.txt \
    && python -m spacy download en_core_web_lg

# Pre-cache CLIP weights at build so the first request isn't a ~600 MB download.
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
