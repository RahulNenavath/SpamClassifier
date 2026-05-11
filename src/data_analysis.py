"""
Data analysis script for the SMS spam dataset.

Usage:
    python src/data_analysis.py --data-path data/spam_dataset.gzip
    python src/data_analysis.py --data-path data/spam_dataset.gzip --output-dir reports/
"""

import argparse
import os
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud

LABEL_COL = "target"
TEXT_COL = "message"
HAM_LABEL = "ham"
SPAM_LABEL = "spam"


def load_dataset(path: str) -> pd.DataFrame:
    path = path.strip()
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
        # Raw Kaggle format: v1=label, v2=message
        if "v1" in df.columns and "v2" in df.columns:
            df = df[["v1", "v2"]].rename(columns={"v1": LABEL_COL, "v2": TEXT_COL})
    elif path.endswith(".csv") or path.endswith(".csv.gz"):
        df = pd.read_csv(path)
    else:
        # Try parquet first (the .gzip file is actually parquet)
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = pd.read_csv(path)

    # Normalise label values: strip whitespace/newlines, lowercase
    df[LABEL_COL] = df[LABEL_COL].str.strip().str.lower()
    df[TEXT_COL] = df[TEXT_COL].astype(str)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape          : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns        : {df.columns.tolist()}")
    print()

    print("── Null counts ─────────────────────────")
    print(df.isnull().sum().to_string())
    print()

    print("── Class distribution ──────────────────")
    vc = df[LABEL_COL].value_counts()
    vcp = df[LABEL_COL].value_counts(normalize=True) * 100
    for label in vc.index:
        print(f"  {label:<8}: {vc[label]:>5,}  ({vcp[label]:.1f}%)")
    ratio = vc.max() / vc.min()
    if ratio >= 2:
        print(f"  [!] Imbalance ratio {ratio:.1f}:1 — class weighting recommended")
    print()

    df["_char_len"] = df[TEXT_COL].str.len()
    df["_word_len"] = df[TEXT_COL].str.split().str.len()
    print("── Message length (characters) ─────────")
    print(df.groupby(LABEL_COL)["_char_len"].describe()[["mean", "50%", "max"]].to_string())
    print()
    print("── Message length (words) ──────────────")
    print(df.groupby(LABEL_COL)["_word_len"].describe()[["mean", "50%", "max"]].to_string())
    print("=" * 60)


def _tokenize(text_series: pd.Series) -> list[str]:
    tokens = []
    for t in text_series:
        tokens.extend(re.findall(r"\b[a-zA-Z]{2,}\b", t.lower()))
    return tokens


def plot_missing_values(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.heatmap(df[[LABEL_COL, TEXT_COL]].isnull(), cbar=False, ax=ax, yticklabels=False)
    ax.set_title("Missing Values Heatmap")
    fig.tight_layout()
    fig.savefig(out_dir / "missing_values.png", dpi=150)
    plt.close(fig)
    print("Saved: missing_values.png")


def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    vc = df[LABEL_COL].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(vc.index, vc.values, color=["#4CAF50", "#F44336"])
    ax.set_title("Class Distribution")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, vc.values):
        pct = val / vc.sum() * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved: class_distribution.png")


def plot_length_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    df["_char_len"] = df[TEXT_COL].str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, color in [(HAM_LABEL, "#4CAF50"), (SPAM_LABEL, "#F44336")]:
        subset = df[df[LABEL_COL] == label]["_char_len"]
        ax.hist(subset, bins=50, alpha=0.6, label=label, color=color)
    ax.set_title("Message Length Distribution (characters)")
    ax.set_xlabel("Character count")
    ax.set_ylabel("Frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "length_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved: length_distribution.png")


def plot_wordclouds(df: pd.DataFrame, out_dir: Path) -> None:
    for label, fname, bg in [
        (SPAM_LABEL, "wordcloud_spam.png", "#1a1a2e"),
        (HAM_LABEL, "wordcloud_ham.png", "#1a2e1a"),
    ]:
        text = " ".join(df[df[LABEL_COL] == label][TEXT_COL])
        wc = WordCloud(
            width=800, height=400,
            background_color=bg,
            colormap="Reds" if label == SPAM_LABEL else "Greens",
            max_words=150,
        ).generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Word Cloud — {label.upper()}", color="white", fontsize=16, pad=10)
        fig.patch.set_facecolor(bg)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150, facecolor=bg)
        plt.close(fig)
        print(f"Saved: {fname}")


def plot_top_words(df: pd.DataFrame, out_dir: Path, top_n: int = 20) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (label, color) in zip(axes, [(SPAM_LABEL, "#F44336"), (HAM_LABEL, "#4CAF50")]):
        tokens = _tokenize(df[df[LABEL_COL] == label][TEXT_COL])
        common = Counter(tokens).most_common(top_n)
        words, counts = zip(*common)
        ax.barh(list(reversed(words)), list(reversed(counts)), color=color)
        ax.set_title(f"Top {top_n} Words — {label.upper()}")
        ax.set_xlabel("Frequency")
    fig.tight_layout()
    fig.savefig(out_dir / "top_words.png", dpi=150)
    plt.close(fig)
    print("Saved: top_words.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="SMS spam dataset analysis")
    parser.add_argument("--data-path", required=True, help="Path to dataset file")
    parser.add_argument("--output-dir", default="reports", help="Directory to save plots")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {args.data_path}")
    df = load_dataset(args.data_path)

    print_summary(df)

    print("\nGenerating visualizations...")
    plot_missing_values(df, out_dir)
    plot_class_distribution(df, out_dir)
    plot_length_distribution(df, out_dir)
    plot_wordclouds(df, out_dir)
    plot_top_words(df, out_dir)

    print(f"\nAll reports saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
