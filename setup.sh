#!/usr/bin/env bash
set -e

ENV_NAME="spam-classifier"
PYTHON_VERSION="3.11"

echo "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Platform-aware PyTorch install
OS=$(uname -s)
if [[ "$OS" == "Darwin" ]]; then
    echo "Detected macOS — installing PyTorch with MPS (Metal) support..."
    pip install torch torchvision torchaudio
else
    echo "Detected Linux — installing PyTorch with CUDA 12.4 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

echo "Installing project dependencies..."
pip install -e .

echo "Registering Jupyter kernel..."
python -m ipykernel install --user \
    --name "$ENV_NAME" \
    --display-name "Python (spam-classifier)"

echo ""
echo "Setup complete."
echo "  Activate  : conda activate $ENV_NAME"
echo "  Analyze   : python src/data_analysis.py --data-path data/spam_dataset.gzip"
echo "  Train     : python src/train.py --data-path data/spam_dataset.gzip"
echo "  Serve     : uvicorn src.inference:app --reload"
