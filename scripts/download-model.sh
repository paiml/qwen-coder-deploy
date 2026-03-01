#!/bin/bash
set -euo pipefail

MODEL_ID="${1:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
MODEL_DIR="${2:-/home/noah/models/qwen-coder-1.5b}"

echo "Downloading $MODEL_ID to $MODEL_DIR/safetensors/"
mkdir -p "$MODEL_DIR/safetensors"

# Use huggingface-cli if available, else fallback to git clone
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$MODEL_ID" --local-dir "$MODEL_DIR/safetensors"
else
    echo "huggingface-cli not found, using git clone"
    git lfs install
    git clone "https://huggingface.co/$MODEL_ID" "$MODEL_DIR/safetensors"
fi

echo "Download complete: $MODEL_DIR/safetensors/"
ls -lh "$MODEL_DIR/safetensors/"
