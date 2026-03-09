# qwen-coder-deploy

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture" width="720"/>
</p>

Deploy and benchmark Qwen2.5-Coder-1.5B-Instruct across realizar, ollama, and llama.cpp. All infrastructure managed via [forjar](https://github.com/paiml/forjar).

## Quick Start

```bash
# GPU deployment (localhost, RTX 4090)
make deploy-gpu        # Build + start all 3 runtimes via forjar
make health-gpu        # Health check all endpoints
make test-gpu          # Correctness tests (6 prompts x 3 runtimes)
make load-gpu          # Load tests (60s, concurrency=4)
make report            # Generate performance.md + update README

# CPU deployment (intel host)
make deploy            # Deploy via forjar to 192.168.50.100
make test              # Correctness tests
make load              # Load tests

# Teardown
make teardown-gpu      # Stop GPU processes
make teardown          # Stop CPU services
```

## Runtimes

| Runtime | Port | Model Format | GPU |
|---------|------|-------------|-----|
| realizar (Sovereign AI Stack) | 8081 | GGUF Q4_K_M | CUDA (fused Q4K kernels) |
| ollama | 8082 | GGUF Q4_K_M | CUDA (auto-detected) |
| llama.cpp | 8083 | GGUF Q4_K_M | CUDA (full offload, -ngl 99) |

<!-- PERFORMANCE_START -->
## Performance Results

### Decode Parity — RTX 4060 Laptop (yoga, c=1, isolated, streaming, Mar 8 2026)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) | µs/layer |
|---------|-------------|--------------|---------------|-------------|----------|
| ollama | **150.7** | 319.0 | 72.1 | **6.6** | 236.9 |
| llama.cpp | 143.5 | **2,188.3** | **10.5** | 7.0 | 248.9 |
| **realizr** | 140.0 | 448.5 | 51.3 | 7.1 | 255.2 |

**Decode: 3-way parity.** realizr 0.98x of llama.cpp. Prefill gap: 4.9x (HGEMM FP16 vs fused Q4K GEMM).

### Concurrent Batching — RTX 4060 Laptop (yoga, c=4, isolated, streaming, Mar 8 2026)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| llama.cpp | **291.7** | 74.7 | **23.7** | **13.4** |
| **realizr** | 177.7 | 50.1 | 149.5 | 19.3 |
| ollama | 143.5 | **145.6** | 677.6 | 6.9 |

**c=4 gap: 0.61x llama.cpp.** Root cause: batch scheduler serialization ([realizr GH-141](https://github.com/paiml/realizar/issues/141)).

### Parity Scorecard (realizr vs llama.cpp)

| Metric | c=1 | Status | c=4 | Status |
|--------|-----|--------|-----|--------|
| Decode tok/s | 0.98x | PASS | 0.61x agg | FAIL |
| TTFT P50 | 4.9x | FAIL | 6.3x | FAIL |
| ITL P50 | 1.01x | PASS | 1.44x | PASS |

### Cross-Platform Decode (c=1, isolated)

| Platform | realizr | llama.cpp | Gap |
|----------|---------|-----------|-----|
| RTX 4060 Laptop (24 SMs) | 140.0 | 143.5 | **0.98x** |
| RTX 4090 (128 SMs) | 411.7 | 436.9 | 1.06x |
| Jetson Orin (8 SMs) | **36.3** | 33.1 | **0.91x** |

<!-- PERFORMANCE_END -->

## Infrastructure

| File | Purpose |
|------|---------|
| `forjar-gpu.yaml` | GPU deployment (localhost, RTX 4090) |
| `forjar.yaml` | CPU deployment (intel host, SSH) |
| `prompts/correctness.yaml` | 6-prompt correctness test suite |
| `scripts/nightly.sh` | Automated benchmark pipeline |

## Correctness

All 3 runtimes pass 6/6 correctness tests (math, code gen, explanation, JSON, SQL).

## Testing

Correctness tests verify basic capabilities (math, code generation, explanation).
Load tests measure throughput, latency percentiles, and tokens/sec via `probador llm load`.

All results stored in `results/` and aggregated in `performance.md`.
