#!/bin/bash
set -euo pipefail

MODEL_DIR="${1:-/home/noah/models/qwen-coder-1.5b}"
SAFETENSORS_DIR="$MODEL_DIR/safetensors"
OUTPUT="$MODEL_DIR/qwen-coder-1.5b.apr"

if [ -f "$OUTPUT" ]; then
    echo "APR already exists: $OUTPUT"
    exit 0
fi

echo "Converting SafeTensors to APR v2..."

# Use aprender's CLI for conversion
if ! command -v aprender &>/dev/null; then
    echo "Error: aprender CLI not found. Install with: cargo install aprender"
    exit 1
fi

aprender convert \
    --input "$SAFETENSORS_DIR" \
    --output "$OUTPUT" \
    --format safetensors \
    --compression lz4

echo "APR conversion complete: $OUTPUT"
ls -lh "$OUTPUT"
