import os

import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class Config:
    # HuggingFace model
    hf_model_name: str = "BAAI/bge-small-en-v1.5"
    model_save_path: str = os.path.join(os.path.dirname(__file__), "model", "bge_spam_classifier")
    num_labels: int = 2
    max_seq_length: int = 128  # SMS is short; 128 tokens is ample

    # Inference — lower than original 0.75; calibrate from ROC curve post-training
    decision_threshold: float = 0.5

    # Training
    learning_rate: float = 2e-4  # high LR is safe because encoder is frozen
    batch_size: int = 32
    epochs: int = 10
    val_split: float = 0.2
    random_seed: int = 42

    # Server
    host: str = "0.0.0.0"
    port: int = int(os.getenv("PORT", "8080"))

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        return torch.device("cpu")
