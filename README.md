# qwen-coder-deploy

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture" width="720"/>
</p>

Deploy and benchmark Qwen2.5-Coder-1.5B-Instruct across realizar, ollama, llama.cpp, and vLLM. All infrastructure managed via [forjar](https://github.com/paiml/forjar).

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
| vLLM | 8084 | AWQ INT4 | CUDA (PagedAttention, continuous batching) |

<!-- PERFORMANCE_START -->
## Performance — RTX 4060 Laptop (2026-03-10, PMAT-062)

### Decode Speed (c=1, isolated, streaming, 60s)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|-------------|--------------|---------------|-------------|
| **vLLM** | **159.7** | **7,849** | **13.0** | **6.3** |
| ollama | 145.4 | 1,424 | 71.6 | 6.9 |
| llama.cpp | 142.9 | 8,409 | 12.1 | 7.0 |
| realizr | 138.6 | 2,198 | 46.4 | 7.2 |

### Aggregate Throughput (c=4, isolated, streaming, 60s)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| **vLLM** | **604.7** | **154.5** | 24.8 | **6.5** |
| llama.cpp | 296.5 | 74.4 | **22.7** | 13.4 |
| realizr | 197.5 | 52.2 | 128.5 | 19.2 |
| ollama | 143.8 | 144.6 | 2,678 | 6.9 |

See [performance.md](performance.md) for full history across RTX 4090, Jetson Orin, and CPU.
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
