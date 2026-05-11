# SMS Spam Classifier

Fine-tuned `BAAI/bge-small-en-v1.5` (33.4M param BERT-based model) for binary SMS spam classification. Served via FastAPI, containerised with Docker, trained on Apple Metal GPU (MPS) or CUDA.

## Results

| Model | Accuracy | Spam F1 | Ham F1 | ROC-AUC |
|---|---|---|---|---|
| TF Conv1D + pruning *(previous)* | 98% | 0.96 | 0.98 | — |
| BGE head-only | 93% | 0.88 | 0.95 | 0.984 |
| **BGE last-3-layers fine-tuned** *(current)* | **99%** | **0.98** | **0.99** | **0.9981** |

## Tech Stack

- **Model** — `BAAI/bge-small-en-v1.5` via HuggingFace Transformers 5.x
- **Training** — PyTorch 2.x, Apple MPS (Metal) on macOS / CUDA on Linux
- **Serving** — FastAPI + Uvicorn
- **Containerisation** — Docker (arm64 CPU for local, amd64 CUDA for production)
- **API testing** — Bruno
- **Environment** — Conda + pyproject.toml

## API

```
GET  /         — service info
GET  /ping     — liveness probe
POST /predict  — {"text": "..."} → {"prediction": "spam"|"not-spam", "confidence": 0.97}
```

## Setup

```bash
bash setup.sh
conda activate spam-classifier
```

`setup.sh` creates the `spam-classifier` conda environment, installs MPS-enabled PyTorch on macOS or CUDA 12.4 on Linux, and registers a Jupyter kernel.

## Dataset

Place the dataset in `data/` (gitignored):

```
data/
  spam_dataset.gzip    ← Parquet file, 6,700 rows, columns: target / message
  spam_dataset.xlsx    ← Raw Kaggle format (v1/v2 columns), same rows
```

- **6,700 rows** — Kaggle SMS spam dataset extended with personal SMS
- **73.3% ham / 26.7% spam** — class imbalance compensated via weighted loss
- Raw text is preserved — URLs, phone numbers, and emoji are not stripped

## Training

```bash
python src/train.py --data-path data/spam_dataset.gzip
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--epochs` | 10 | Max training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--unfreeze-layers` | 3 | Last N encoder layers to unfreeze (-1 = full fine-tune) |
| `--patience` | 5 | Early stopping patience (epochs without spam F1 improvement) |
| `--batch-size` | 32 | Batch size |

The best checkpoint (by val spam F1) is saved to `src/model/bge_spam_classifier/`.

## Data Analysis

```bash
python src/data_analysis.py --data-path data/spam_dataset.gzip
```

Generates 5 visualisations in `reports/`: word clouds per class, class distribution, message length histogram, missing values heatmap, and top-20 unigrams per class.

## Inference

**Local (conda, runs on MPS):**
```bash
uvicorn src.inference:app --host 0.0.0.0 --port 8080 --reload
```

**Docker (arm64, CPU — local Mac testing):**
```bash
docker build -f Dockerfile.local -t spam-classifier-local .
docker run -d -p 8080:8080 spam-classifier-local
```

**Docker (amd64, CUDA — production):**
```bash
docker build -t spam-classifier .
docker run -d -p 8080:8080 --gpus all spam-classifier
```

## Testing with Bruno

Open Bruno → **Open Collection** → select the `bruno/` folder. Choose the `local` or `docker` environment (both target `http://localhost:8080`).

| Folder | Request | Description |
|---|---|---|
| Health | Root | `GET /` service info |
| Health | Ping | `GET /ping` liveness |
| Predict | Spam Message | Prize/phone-number spam |
| Predict | Ham Message | Normal conversational text |
| Predict | Spam with URL | Phishing URL (tests raw text preservation) |
| Predict | Empty Text | 422 validation error |

## Project Structure

```
src/
  config.py          — Config class: model name, paths, hyperparams, get_device()
  train.py           — Fine-tuning script
  inference.py       — FastAPI app
  data_analysis.py   — EDA + visualisations
  model/
    bge_spam_classifier/  ← HF SavedModel (gitignored, required for Docker build)
  notebooks/
    Spam_Classifier.ipynb ← Historical TF training notebook
data/                — Dataset files (gitignored)
reports/             — Generated visualisations (gitignored)
bruno/               — Bruno API collection
Dockerfile           — Production (amd64, CUDA 12.4, torch 2.5.1)
Dockerfile.local     — Local Mac (arm64, CPU-only)
setup.sh             — Conda environment setup
pyproject.toml       — Project dependencies
```

## Model Architecture

`BAAI/bge-small-en-v1.5` is a 12-layer BERT encoder (384-dim hidden, 33.4M params). Fine-tuning strategy:

- Layers 0–8: **frozen** (preserve broad language understanding)
- Layers 9–11: **trainable** (adapt to SMS domain)
- Classification head: **trainable** (2-layer linear, randomly initialised)
- **5.3M / 33.4M params trained (16%)**

## Deployment (Phase 2 — GCP, planned)

Google Cloud Run with NVIDIA L4 GPU using the production `Dockerfile`. GitHub Actions CI/CD will push to Artifact Registry on merge to `master`.
