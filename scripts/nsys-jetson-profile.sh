#!/bin/bash
# Profile a single inference request with nsys on Jetson
# Usage: ssh jetson 'bash /tmp/nsys-jetson-profile.sh'

set -e

# Stop the server
pkill -f 'apr serve' 2>/dev/null || true
sleep 2

# Run apr bench (single inference, no server needed) under nsys
cd /home/noah/src/aprender
SKIP_PARITY_GATE=1 nsys profile \
    --output /tmp/nsys-jetson-decode \
    --force-overwrite true \
    --trace cuda,nvtx \
    --stats true \
    target/release/apr profile \
    /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
    --skip-contract \
    --format json \
    --warmup 1 2>&1 | tail -80

echo "--- nsys report ---"
nsys stats /tmp/nsys-jetson-decode.nsys-rep --report cuda_gpu_kern_sum 2>&1 | head -30
