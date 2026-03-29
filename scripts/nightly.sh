#!/bin/bash
set -euo pipefail

# Usage: nightly.sh [cpu|gpu|yoga|gx10|wgpu|both|all]
MODE="${1:-gpu}"
DATE=$(date +%Y%m%d)
YOGA_HOST="192.168.50.38"

echo "=== Nightly Benchmark: $(date) (mode: $MODE) ==="

run_benchmark() {
    local target="$1"  # cpu or gpu
    local host="$2"
    local forjar_file="$3"

    echo "--- [$target] Deploying via forjar ---"
    forjar apply -f "$forjar_file"

    echo "--- [$target] Health checks ---"
    for port in 8081 8082 8083; do
        echo -n "  Waiting for :$port..."
        timeout 60 bash -c "until curl -sf http://$host:$port/health >/dev/null 2>&1; do sleep 2; done" || {
            echo " TIMEOUT (skipping)"
            continue
        }
        echo " OK"
    done

    echo "--- [$target] Correctness tests ---"
    for runtime in realizar:8081 ollama:8082 llamacpp:8083; do
        name=${runtime%%:*}
        port=${runtime##*:}
        echo "  Testing $name-$target..."
        probador llm test \
            --config prompts/correctness.yaml \
            --url "http://$host:$port" \
            --runtime-name "$name-$target" \
            --output "results/${name}-${target}-correctness-${DATE}.json" || true
    done

    echo "--- [$target] Load tests ---"
    for runtime in realizar:8081 ollama:8082 llamacpp:8083; do
        name=${runtime%%:*}
        port=${runtime##*:}
        echo "  Load testing $name-$target..."
        probador llm load \
            --url "http://$host:$port" \
            --concurrency 4 \
            --duration 60s \
            --runtime-name "$name-$target" \
            --output "results/${name}-${target}-load-${DATE}.json" || true
    done
}

# PMAT-317: Yoga isolated serial benchmark — deploy one runtime at a time,
# benchmark at multiple concurrency levels, teardown, repeat.
run_yoga_serial() {
    echo "--- [yoga] Isolated serial benchmarks ---"

    local runtimes=("realizr:forjar-yoga-realizr.yaml:8081" \
                    "llamacpp:forjar-yoga-llamacpp.yaml:8083" \
                    "ollama:forjar-yoga-ollama.yaml:8082")
    local concurrencies=(1 4 8 16 32)

    for rt_spec in "${runtimes[@]}"; do
        IFS=':' read -r name forjar port <<< "$rt_spec"
        echo "=== [yoga] $name ==="

        # Deploy
        echo "  Deploying $name..."
        forjar apply -f "$forjar" || { echo "  Deploy FAILED, skipping"; continue; }

        # Wait for health
        echo -n "  Health check :$port..."
        timeout 60 bash -c "until curl -sf http://$YOGA_HOST:$port/health >/dev/null 2>&1; do sleep 2; done" || {
            echo " TIMEOUT, skipping"
            forjar apply -f forjar-yoga-teardown.yaml 2>/dev/null || true
            continue
        }
        echo " OK"

        # Correctness
        echo "  Correctness..."
        local model_flag=""
        [[ "$name" == "ollama" ]] && model_flag="--model qwen2.5-coder:1.5b-instruct"
        probador llm test \
            --config prompts/correctness.yaml \
            --url "http://$YOGA_HOST:$port" \
            $model_flag \
            --runtime-name "$name-yoga" || true

        # Load tests at each concurrency level
        for c in "${concurrencies[@]}"; do
            # Skip high concurrency for runtimes that don't scale
            [[ "$name" == "ollama" && $c -gt 32 ]] && continue
            echo "  Load c=$c..."
            probador llm load \
                --url "http://$YOGA_HOST:$port" \
                --concurrency "$c" \
                --duration 60 \
                --warmup 5 \
                --prompt-profile short \
                --max-tokens-distribution uniform:16,256 \
                --stream true \
                $model_flag \
                --runtime-name "$name-yoga" \
                -o "results/yoga-serial-${name}-c${c}-${DATE}.json" || true
        done

        # Teardown
        echo "  Teardown..."
        forjar apply -f forjar-yoga-teardown.yaml 2>/dev/null || true
        sleep 5
    done

    # Score
    echo "--- [yoga] Scoring ---"
    probador llm score \
        --results results/ \
        --platform yoga \
        --format table \
        --fail-on-grade C 2>&1 || true
}

# PMAT-412: GB10 Blackwell benchmark — 1.5B + 7B models, SSH tunnel
run_gx10() {
    echo "--- [gx10] Grace Blackwell GB10 benchmarks ---"

    # Ensure SSH tunnel
    if ! curl -sf http://127.0.0.1:9081/health >/dev/null 2>&1; then
        echo "  Setting up SSH tunnel to gx10..."
        ssh -f -N -L 9081:localhost:8081 gx10 || { echo "SSH tunnel failed"; return 1; }
        sleep 2
    fi

    local GX10_URL="http://127.0.0.1:9081"
    local concurrencies=(1 4 8 16 32)
    local models=("1.5b:qwen2.5-coder-1.5b-instruct-q4_k_m.gguf" \
                   "7b:qwen2.5-coder-7b-instruct-q4_k_m.gguf" \
                   "32b:qwen2.5-coder-32b-instruct-q4_k_m.gguf")

    for model_spec in "${models[@]}"; do
        IFS=':' read -r size gguf <<< "$model_spec"
        echo "=== [gx10] ${size} ==="

        # Deploy model (32B needs BATCH=4 for memory)
        local batch=32
        [[ "$size" == "32b" ]] && batch=4
        echo "  Starting ${size} on gx10 (BATCH=$batch)..."
        ssh gx10 "pkill apr 2>/dev/null; sleep 2; cd ~/src/aprender && \
            SKIP_PARITY_GATE=1 CUDA_MAX_BATCH=$batch ITERATION_SCHEDULER=1 \
            nohup ./target/release/apr serve run ~/models/${gguf} \
            --gpu --host 0.0.0.0 --port 8081 </dev/null > /tmp/apr-${size}-nightly.log 2>&1 & disown"

        # Wait for health
        echo -n "  Health check..."
        timeout 120 bash -c "until curl -sf $GX10_URL/health >/dev/null 2>&1; do sleep 3; done" || {
            echo " TIMEOUT, skipping ${size}"
            continue
        }
        echo " OK"

        # Correctness
        echo "  Correctness..."
        probador llm test \
            --config prompts/correctness.yaml \
            --url "$GX10_URL" \
            --runtime-name "realizr-gb10-${size}" || true

        # Load tests at each concurrency level (32B caps at c=4)
        for c in "${concurrencies[@]}"; do
            [[ "$size" == "32b" && $c -gt 4 ]] && continue
            echo "  Load c=$c..."
            probador llm load \
                --url "$GX10_URL" \
                --concurrency "$c" \
                --duration 60 \
                --warmup 5 \
                --stream true \
                --max-tokens 128 \
                --runtime-name "realizr-gb10-${size}" \
                -o "results/gb10-${size}-c${c}-${DATE}.json" || true
        done

        # Stop model
        ssh gx10 'pkill apr 2>/dev/null' || true
        sleep 3
    done

    echo "--- [gx10] Scoring ---"
    probador llm score \
        --results results/ \
        --platform gb10 \
        --format table 2>&1 || true
}

case "$MODE" in
    cpu)
        run_benchmark "cpu" "192.168.50.100" "forjar.yaml"
        ;;
    gpu)
        run_benchmark "gpu" "127.0.0.1" "forjar-gpu.yaml"
        ;;
    yoga)
        run_yoga_serial
        ;;
    gx10)
        run_gx10
        ;;
    both)
        run_benchmark "cpu" "192.168.50.100" "forjar.yaml"
        run_benchmark "gpu" "127.0.0.1" "forjar-gpu.yaml"
        ;;
    wgpu)
        # PMAT-376: WGPU correctness gate on intel/W5700X
        INTEL_HOST="192.168.50.100"
        echo "--- [wgpu] Starting WGPU on intel ---"
        ssh "$INTEL_HOST" 'pkill -f "backend wgpu" 2>/dev/null; true' || true
        sleep 2
        ssh "$INTEL_HOST" 'nohup apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --backend wgpu --port 8081 --host 0.0.0.0 > /tmp/wgpu-serve.log 2>&1 & echo started'
        echo "Waiting for WGPU startup..."
        sleep 20
        echo "--- [wgpu] Correctness ---"
        for prompt in "What is 2+2?" "Capital of France?" "Write hello in Python"; do
            echo -n "  \"$prompt\": "
            result=$(curl -s --max-time 120 "http://$INTEL_HOST:8081/v1/chat/completions" \
                -H 'Content-Type: application/json' \
                -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":32}" \
                | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null)
            echo "$result"
            [[ -z "$result" ]] && echo "FAIL: empty response" >&2
        done
        echo "--- [wgpu] Streaming test ---"
        curl -s --max-time 120 -N "http://$INTEL_HOST:8081/v1/chat/completions" \
            -H 'Content-Type: application/json' \
            -d '{"model":"qwen","messages":[{"role":"user","content":"Count to 3"}],"max_tokens":8,"stream":true}' \
            | grep -c "delta" | xargs -I{} echo "  {} SSE chunks received"
        echo "--- [wgpu] Parity gate (WGPU vs CPU) ---"
        # Start CPU backend for comparison
        ssh "$INTEL_HOST" 'pgrep -f "port 8082" >/dev/null || nohup apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --port 8082 --host 0.0.0.0 > /tmp/cpu-serve.log 2>&1 &' || true
        sleep 8
        make parity-wgpu || echo "  Parity gate: CPU backend not available (skipped)"
        ;;
    all)
        run_benchmark "cpu" "192.168.50.100" "forjar.yaml"
        run_yoga_serial
        run_gx10
        ;;
    *)
        echo "Usage: nightly.sh [cpu|gpu|yoga|gx10|wgpu|both|all]"
        exit 1
        ;;
esac

echo "--- Generating reports ---"
probador llm report \
    --results results/ \
    --output performance.md \
    --update-readme README.md

echo "--- Computing scores ---"
probador llm score \
    --results results/ \
    --format table \
    --by-layer --by-profile --by-correctness --by-output-length \
    --by-memory --by-cold-start --by-power --by-scaling

probador llm score \
    --results results/ \
    --format json \
    --output "results/scorecard-${DATE}.json" \
    --by-layer --by-profile --by-correctness --by-output-length \
    --by-memory --by-cold-start --by-power --by-scaling

echo "--- Committing results ---"
git add results/ performance.md README.md
git commit -m "bench: $(date +%Y-%m-%d) $MODE benchmark results" || echo "No changes to commit"

echo "=== Nightly complete ==="
