"""
Fine-tune BAAI/bge-small-en-v1.5 for SMS spam classification.

Only the 2-layer classification head is trained; the encoder is frozen.
Supports Apple MPS (Metal), CUDA, and CPU automatically via Config.get_device().

Usage:
    python src/train.py --data-path data/spam_dataset.gzip
    python src/train.py --data-path data/spam_dataset.gzip --epochs 10 --batch-size 32
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import Config

LABEL_COL = "target"
TEXT_COL = "message"
LABEL_MAP = {"ham": 0, "spam": 1}


def load_dataset(path: str) -> pd.DataFrame:
    path = path.strip()
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": LABEL_COL, "v2": TEXT_COL})
    elif path.endswith(".csv") or path.endswith(".csv.gz"):
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path)

    df[LABEL_COL] = df[LABEL_COL].str.strip().str.lower()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    df = df[df[LABEL_COL].isin(LABEL_MAP)].reset_index(drop=True)
    df["label"] = df[LABEL_COL].map(LABEL_MAP)
    return df


class SpamDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_probs, all_preds, all_labels = [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.softmax(logits, dim=-1)[:, 1]  # spam probability
        preds = (probs > Config.decision_threshold).long()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    spam_f1 = f1_score(all_labels, all_preds, pos_label=1, zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_probs)
    return {
        "loss": total_loss / len(loader),
        "spam_f1": spam_f1,
        "roc_auc": roc_auc,
        "labels": all_labels,
        "preds": all_preds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BGE classifier for SMS spam")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--unfreeze-layers", type=int, default=3,
                        help="Number of final encoder layers to unfreeze. Use -1 to unfreeze the full encoder.")
    parser.add_argument("--output-dir", default=Config.model_save_path)
    args = parser.parse_args()

    device = Config.get_device()
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────
    print(f"Loading data from: {args.data_path}")
    df = load_dataset(args.data_path)
    print(f"Dataset: {len(df):,} rows  |  spam={df['label'].sum():,}  ham={(df['label']==0).sum():,}")

    texts = df[TEXT_COL].tolist()
    labels = df["label"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=Config.val_split,
        stratify=labels,
        random_state=Config.random_seed,
    )
    print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}")

    # ── Tokenizer ────────────────────────────────────────────────
    print(f"Loading tokenizer: {Config.hf_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.hf_model_name)

    train_ds = SpamDataset(X_train, y_train, tokenizer, Config.max_seq_length)
    val_ds = SpamDataset(X_val, y_val, tokenizer, Config.max_seq_length)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ── Model ────────────────────────────────────────────────────
    print(f"Loading model: {Config.hf_model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.hf_model_name,
        num_labels=Config.num_labels,
        ignore_mismatched_sizes=True,
    )

    # Freeze everything first, then selectively unfreeze
    for param in model.parameters():
        param.requires_grad = False

    encoder_layers = model.base_model.encoder.layer  # list of transformer blocks
    n_layers = len(encoder_layers)

    if args.unfreeze_layers == -1:
        # Full fine-tune: unfreeze entire encoder
        for param in model.base_model.parameters():
            param.requires_grad = True
        unfreeze_desc = f"all {n_layers} encoder layers (full fine-tune)"
    else:
        # Unfreeze only the last N transformer layers
        n_unfreeze = min(args.unfreeze_layers, n_layers)
        for layer in encoder_layers[-n_unfreeze:]:
            for param in layer.parameters():
                param.requires_grad = True
        unfreeze_desc = f"last {n_unfreeze} of {n_layers} encoder layers"

    # Classifier head is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Unfreezing: {unfreeze_desc} + classifier head")
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    model = model.to(device)

    # ── Class-weighted loss ──────────────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1]),
        y=y_train,
    )
    weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    # ── Training loop ────────────────────────────────────────────
    best_f1 = 0.0
    best_epoch = 0
    epochs_without_improvement = 0

    print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>9}  {'Spam F1':>8}  {'ROC-AUC':>8}  {'':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        improved = val_metrics["spam_f1"] > best_f1
        if improved:
            best_f1 = val_metrics["spam_f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            tag = "✓ saved"
        else:
            epochs_without_improvement += 1
            tag = f"no improv {epochs_without_improvement}/{args.patience}"

        print(
            f"{epoch:>5}  {train_loss:>10.4f}  {val_metrics['loss']:>9.4f}"
            f"  {val_metrics['spam_f1']:>8.4f}  {val_metrics['roc_auc']:>8.4f}  {tag}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no improvement in spam F1 for {args.patience} consecutive epochs.")
            break

    print(f"\nBest checkpoint: epoch {best_epoch}  spam_f1={best_f1:.4f}")
    print(f"Model saved to: {args.output_dir}")

    # ── Final report on val set (best checkpoint) ────────────────
    print("\n── Final classification report (val set, best checkpoint) ──")
    best_model = AutoModelForSequenceClassification.from_pretrained(args.output_dir).to(device)
    best_model.eval()
    final = evaluate(best_model, val_loader, criterion, device)
    print(classification_report(final["labels"], final["preds"], target_names=["Ham", "Spam"]))
    print(f"ROC-AUC: {final['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
