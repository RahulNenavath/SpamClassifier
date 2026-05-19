# Production image — PyTorch 2.5.1 + CUDA 12.4 (Linux/GCP GPU)
# Matches minimum torch version required by transformers 5.x (custom_op API change in 2.5)
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl gnupg \
    && echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
       | tee /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
       | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir . --no-build-isolation

# Download model weights from GCS at build time
ARG MODEL_GCS_PATH=gs://spam-classifier-sms-model/bge_spam_classifier
RUN gsutil -m cp -r ${MODEL_GCS_PATH} src/model/bge_spam_classifier

ENV PYTORCH_ENABLE_MPS_FALLBACK=1
ENV PORT=8080
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.inference:app", "--host", "0.0.0.0", "--port", "8080"]
