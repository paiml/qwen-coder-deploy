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

### Decode Parity — RTX 4060 Laptop (yoga, c=1, isolated, streaming, Mar 9 2026)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|-------------|--------------|---------------|-------------|
| **realizr** | **138.2** | 2,085.7 | 48.9 | **7.2** |
| llama.cpp | 142.5 | **8,280.4** | **12.3** | 7.0 |

**Decode: realizr 0.97x llama.cpp (parity).** ITL at parity. TTFT 4.0x gap (HGEMM FP16 bandwidth vs fused Q4K).

### Concurrent Batching — RTX 4060 Laptop (yoga, c=4, isolated, streaming, Mar 9 2026)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| llama.cpp | **296.4** | **74.5** | **24.0** | **13.4** |
| **realizr** | 193.8 | 51.2 | 128.6 | 19.5 |

**c=4 gap: 0.65x llama.cpp.** Root cause: HGEMM FP16 reads 3.5x more data than Q4K. PMAT-051 v2 + PMAT-061 improved from 0.49x.

### Parity Scorecard (realizr vs llama.cpp)

| Metric | c=1 | Status | c=4 | Status |
|--------|-----|--------|-----|--------|
| Decode tok/s | 0.97x | **PASS** | 0.65x agg | FAIL |
| TTFT P50 | 4.0x | FAIL (target ≤2x) | 5.4x | FAIL |
| ITL P50 | 1.03x | **PASS** | 1.46x | **PASS** |

### Cross-Platform Decode (c=1, isolated)

| Platform | realizr | llama.cpp | Gap |
|----------|---------|-----------|-----|
| RTX 4060 Laptop (24 SMs) | **138.2** | 142.5 | **0.97x** |
| RTX 4090 (128 SMs) | 411.7 | 436.9 | 1.06x |
| Jetson Orin (8 SMs) | **36.3** | 33.1 | **0.91x** |

### Key Optimizations (cumulative)

| Optimization | Impact |
|-------------|--------|
| PMAT-052: Zero-copy c=1 prefill attention | TTFT 78.8→48.9ms (1.6x) |
| PMAT-051 v2: Multi-prompt zero-copy attn | c=4 TTFT 256→129ms (2.0x) |
| PMAT-061: HGEMM batched decode | c=4 decode +19.6% |
| PMAT-059: Disable prefill graph | c=1 TTFT 561→79ms (7.1x) |
| PMAT-044: Continuous batching scheduler | c=4 from serial to 194 tok/s |
| GH-182: Fused Q4K GEMM (validated) | 16x slower — needs tiling rewrite |

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
