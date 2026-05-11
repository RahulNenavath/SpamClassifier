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

## Historical Baseline — TF v1 (ablation)

The original implementation used TensorFlow 2.11 with two custom architectures trained from scratch on a 1,000-token vocabulary (embedding dim 50, max sequence 150). Text was lowercased and stripped of all punctuation and special characters before tokenisation — URLs and phone numbers were lost entirely.

**Train / test split:** 80/20, no stratification — 5,360 train / 1,340 test.

### Architectures

**BiLSTM**
```
Embedding(1000, 50) → BiLSTM(64, return_seq) → BiLSTM(64) → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

**Conv1D** ← selected for deployment
```
Embedding(1000, 50) → Conv1D(128, k=7, stride=3) → Conv1D(128, k=7, stride=3) → GlobalMaxPool → Dense(128, relu) → Dropout(0.5) → Dense(1, sigmoid)
```

### Results (threshold = 0.75)

| Model | Accuracy | Ham F1 | Spam F1 | Spam Recall |
|---|---|---|---|---|
| BiLSTM | 87% | 0.92 | 0.67 | **0.50** ← critical failure |
| Conv1D | 98% | 0.98 | 0.96 | 0.93 |

### 10-Fold Cross-Validation

| Model | Mean F1 | Median F1 |
|---|---|---|
| BiLSTM | 0.933 | 0.940 |
| Conv1D | 0.930 | 0.935 |

K-fold scores were near-identical across both models. Conv1D was selected because it showed meaningfully better per-class F1 on the held-out test set — particularly spam recall (0.93 vs 0.50 on BiLSTM), which is the more critical metric for a spam classifier.

### Post-training Pruning

Conv1D was pruned using `tensorflow-model-optimization` with polynomial magnitude pruning (50% → 80% sparsity over 10 epochs, batch size 32). This reduced model size significantly with no meaningful accuracy loss. The pruned model was saved as a TF SavedModel with the text standardization layer baked in end-to-end.

> Note: the pruned model reported 100% test accuracy during evaluation, which is a known batching artifact from the eval step count — the true baseline is the unpruned Conv1D at 98%.

---

## Deployment (Phase 2 — GCP, planned)

Google Cloud Run with NVIDIA L4 GPU using the production `Dockerfile`. GitHub Actions CI/CD will push to Artifact Registry on merge to `master`.
