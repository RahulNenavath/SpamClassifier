"""
FastAPI inference server for the SMS spam classifier.

Usage:
    uvicorn src.inference:app --host 0.0.0.0 --port 8080 --reload

Routes:
    GET  /       — service info
    GET  /ping   — liveness probe
    POST /predict — {"text": "..."} → {"prediction": "spam"|"not-spam", "confidence": 0.97}
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import Config

app = FastAPI(title="SMS Spam Classifier", version="0.2.0")

_API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def _require_api_key(api_key: str = Security(_api_key_header)) -> None:
    if _API_KEY and api_key != _API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key")

_tokenizer = None
_model = None
_device = None


@app.on_event("startup")
def load_model() -> None:
    global _tokenizer, _model, _device

    if not os.path.isdir(Config.model_save_path):
        raise RuntimeError(
            f"Model not found at '{Config.model_save_path}'. "
            "Run src/train.py first to train and save the model."
        )

    _device = Config.get_device()
    print(f"Loading model from: {Config.model_save_path}  (device={_device})")

    _tokenizer = AutoTokenizer.from_pretrained(Config.model_save_path)
    _model = AutoModelForSequenceClassification.from_pretrained(Config.model_save_path)
    _model.eval()
    _model.to(_device)

    print("Model ready.")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    prediction: str
    confidence: float


@app.get("/", dependencies=[Security(_require_api_key)])
def root():
    return {
        "service": "SMS Spam Classifier",
        "model": Config.hf_model_name,
        "status": "active",
    }


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse, dependencies=[Security(_require_api_key)])
def predict(req: PredictRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    inputs = _tokenizer(
        req.text,
        return_tensors="pt",
        truncation=True,
        max_length=Config.max_seq_length,
        padding=True,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _model(**inputs).logits

    spam_prob = torch.softmax(logits, dim=-1)[0][1].item()
    label = "spam" if spam_prob > Config.decision_threshold else "not-spam"

    return PredictResponse(prediction=label, confidence=round(spam_prob, 4))


def serve() -> None:
    import uvicorn
    uvicorn.run("src.inference:app", host=Config.host, port=Config.port, reload=False)


if __name__ == "__main__":
    serve()
