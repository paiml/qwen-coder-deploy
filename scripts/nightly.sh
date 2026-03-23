#!/bin/bash
set -euo pipefail

# Usage: nightly.sh [cpu|gpu|yoga|both|all]
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
    both)
        run_benchmark "cpu" "192.168.50.100" "forjar.yaml"
        run_benchmark "gpu" "127.0.0.1" "forjar-gpu.yaml"
        ;;
    all)
        run_benchmark "cpu" "192.168.50.100" "forjar.yaml"
        run_yoga_serial
        ;;
    *)
        echo "Usage: nightly.sh [cpu|gpu|yoga|both|all]"
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
