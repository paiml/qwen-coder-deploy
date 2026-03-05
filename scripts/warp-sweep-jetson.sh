#!/bin/bash
# Warp count sweep for MWV Q4K GEMV on Jetson Orin
set -e

MODEL="/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
APR="/home/noah/src/aprender/target/release/apr"
PARSE='
import sys, json
raw = sys.stdin.read()
# Find first { to skip ANSI progress lines on stdout
start = raw.find("{")
if start < 0:
    print("  NO JSON FOUND", file=sys.stderr)
    sys.exit(1)
d = json.loads(raw[start:])
t = d["timing"]
tps = t["throughput_tok_s"]
avg = t["avg_inference_us"] / 1000
print("  %.1f tok/s, avg=%.1fms" % (tps, avg))
'

echo "=== MWV_WARPS Sweep ==="
for W in 1 2 3 4 6 8; do
    echo "--- MWV_WARPS=$W ---"
    MWV_WARPS=$W SKIP_PARITY_GATE=1 $APR profile "$MODEL" --skip-contract --warmup 1 --format json 2>/dev/null | python3 -c "$PARSE" || echo "  FAILED"
done

echo ""
echo "=== Kernel Variant Sweep ==="
for VAR in "WIDE_Q4K=1" "VECTORIZED_Q4K=1" "DP4A_Q4K=1" "WIDE_Q4K_DISABLE=1"; do
    echo "--- $VAR ---"
    export $VAR
    SKIP_PARITY_GATE=1 $APR profile "$MODEL" --skip-contract --warmup 1 --format json 2>/dev/null | python3 -c "$PARSE" || echo "  FAILED"
    unset ${VAR%%=*}
done
