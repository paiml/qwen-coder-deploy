#!/bin/bash
set -euo pipefail

MODEL_DIR="${1:-/home/noah/models/qwen-coder-1.5b}"
SAFETENSORS_DIR="$MODEL_DIR/safetensors"
OUTPUT="$MODEL_DIR/qwen-coder-1.5b-q4_k_m.gguf"

if [ -f "$OUTPUT" ]; then
    echo "GGUF already exists: $OUTPUT"
    exit 0
fi

echo "Converting SafeTensors to GGUF (Q4_K_M)..."

# Requires llama.cpp's convert script
CONVERT_SCRIPT="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}/convert_hf_to_gguf.py"

if [ ! -f "$CONVERT_SCRIPT" ]; then
    echo "Error: convert script not found at $CONVERT_SCRIPT"
    echo "Set LLAMA_CPP_DIR to your llama.cpp directory"
    exit 1
fi

# Step 1: Convert to F16 GGUF
F16_GGUF="$MODEL_DIR/qwen-coder-1.5b-f16.gguf"
python3 "$CONVERT_SCRIPT" "$SAFETENSORS_DIR" --outfile "$F16_GGUF" --outtype f16

# Step 2: Quantize to Q4_K_M
QUANTIZE="${LLAMA_CPP_DIR:-$HOME/src/llama.cpp}/build/bin/llama-quantize"

if [ ! -f "$QUANTIZE" ]; then
    echo "Error: llama-quantize not found at $QUANTIZE"
    exit 1
fi

"$QUANTIZE" "$F16_GGUF" "$OUTPUT" Q4_K_M

# Clean up F16 intermediate
rm -f "$F16_GGUF"

echo "GGUF conversion complete: $OUTPUT"
ls -lh "$OUTPUT"
