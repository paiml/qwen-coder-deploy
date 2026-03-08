#!/bin/bash
# Warp sweep on Jetson Orin — test MWV_WARPS={1,2,3,4,6,8} for decode throughput
# Default (3) was tuned for RTX 4090 (128 SMs). Orin has 16 SMs — likely different optimal.
set -euo pipefail

JETSON=192.168.50.53
MODEL=/home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
PORT=8081
PROMPT='{"model":"qwen2.5-coder:1.5b-instruct","messages":[{"role":"user","content":"Write a fibonacci function in Python"}],"max_tokens":50,"stream":false}'

for WARPS in 1 2 3 4 6 8; do
    echo "=== MWV_WARPS=$WARPS ==="

    # Kill any existing server
    ssh noah@$JETSON "for p in \$(pgrep realizr 2>/dev/null); do kill -9 \$p 2>/dev/null; done" 2>/dev/null || true
    sleep 2

    # Start server with this warp count
    ssh noah@$JETSON "SKIP_PARITY_GATE=1 DECODE_TIMING=1 DP4A_Q4K=1 MWV_Q6K=1 MWV_WARPS=$WARPS nohup apr serve -m $MODEL --gpu --host 0.0.0.0 --port $PORT > /tmp/apr-warp-$WARPS.log 2>&1 &"

    # Wait for server
    for i in $(seq 1 30); do
        if curl -sf http://$JETSON:$PORT/health >/dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if ! curl -sf http://$JETSON:$PORT/health >/dev/null 2>&1; then
        echo "  FAILED to start with MWV_WARPS=$WARPS"
        continue
    fi

    # Warmup request
    curl -sf -X POST http://$JETSON:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' -d "$PROMPT" > /dev/null 2>&1

    # Timed request
    START=$(date +%s%N)
    RESULT=$(curl -sf -X POST http://$JETSON:$PORT/v1/chat/completions \
        -H 'Content-Type: application/json' -d "$PROMPT" 2>&1)
    END=$(date +%s%N)

    TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['usage']['completion_tokens'])" 2>/dev/null || echo 0)
    ELAPSED_MS=$(( (END - START) / 1000000 ))

    if [ "$TOKENS" -gt 0 ]; then
        TOKS_PER_SEC=$(python3 -c "print(f'{$TOKENS / ($ELAPSED_MS / 1000):.1f}')")
        echo "  tokens=$TOKENS elapsed=${ELAPSED_MS}ms throughput=${TOKS_PER_SEC} tok/s"
    else
        echo "  FAILED to generate tokens"
    fi

    # Get server-side decode timing (last 3 to see steady-state)
    ssh noah@$JETSON "grep 'DECODE-TIMING' /tmp/apr-warp-$WARPS.log | tail -3" 2>/dev/null || true
    echo ""
done

echo "=== Sweep complete ==="
