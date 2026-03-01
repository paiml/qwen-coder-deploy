#!/bin/bash
set -euo pipefail

INTEL_HOST="192.168.50.100"
DATE=$(date +%Y%m%d)

echo "=== Nightly Benchmark: $(date) ==="

# 1. Deploy/update via forjar
echo "--- Deploying ---"
forjar apply

# 2. Wait for services to be healthy
echo "--- Health checks ---"
for port in 8081 8082 8083; do
    echo -n "  Waiting for :$port..."
    timeout 60 bash -c "until curl -sf http://$INTEL_HOST:$port/health >/dev/null 2>&1; do sleep 2; done"
    echo " OK"
done

# 3. Run correctness tests against all 3 runtimes
echo "--- Correctness tests ---"
for runtime in realizar:8081 ollama:8082 llamacpp:8083; do
    name=${runtime%%:*}
    port=${runtime##*:}
    echo "  Testing $name..."
    probador llm test \
        --config prompts/correctness.yaml \
        --url "http://$INTEL_HOST:$port" \
        --runtime-name "$name" \
        --output "results/${name}-correctness-${DATE}.json"
done

# 4. Run load tests against all 3 runtimes
echo "--- Load tests ---"
for runtime in realizar:8081 ollama:8082 llamacpp:8083; do
    name=${runtime%%:*}
    port=${runtime##*:}
    echo "  Load testing $name..."
    probador llm load \
        --url "http://$INTEL_HOST:$port" \
        --concurrency 4 \
        --duration 60s \
        --runtime-name "$name" \
        --output "results/${name}-load-${DATE}.json"
done

# 5. Generate reports
echo "--- Generating reports ---"
probador llm report \
    --results results/ \
    --output performance.md \
    --update-readme README.md

# 6. Commit results
echo "--- Committing results ---"
git add results/ performance.md README.md
git commit -m "nightly: benchmark results $(date +%Y-%m-%d)" || echo "No changes to commit"
git push origin master

echo "=== Nightly complete ==="
