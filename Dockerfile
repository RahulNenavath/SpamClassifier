# Production image — PyTorch 2.5.1 + CUDA 12.4 (Linux/GCP GPU)
# Matches minimum torch version required by transformers 5.x (custom_op API change in 2.5)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir . --no-build-isolation

ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8080"]
