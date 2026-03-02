#!/bin/bash
set -euo pipefail

# Usage: nightly.sh [cpu|gpu|both]
MODE="${1:-gpu}"
DATE=$(date +%Y%m%d)

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

case "$MODE" in
    cpu)
        run_benchmark "cpu" "192.168.50.100" "forjar.yaml"
        ;;
    gpu)
        run_benchmark "gpu" "127.0.0.1" "forjar-gpu.yaml"
        ;;
    both)
        run_benchmark "cpu" "192.168.50.100" "forjar.yaml"
        run_benchmark "gpu" "127.0.0.1" "forjar-gpu.yaml"
        ;;
    *)
        echo "Usage: nightly.sh [cpu|gpu|both]"
        exit 1
        ;;
esac

echo "--- Generating reports ---"
probador llm report \
    --results results/ \
    --output performance.md \
    --update-readme README.md

echo "--- Committing results ---"
git add results/ performance.md README.md
git commit -m "bench: $(date +%Y-%m-%d) $MODE benchmark results" || echo "No changes to commit"

echo "=== Nightly complete ==="
