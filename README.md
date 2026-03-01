# qwen-coder-deploy

Deploy and benchmark Qwen2.5-Coder-1.5B-Instruct across realizar, ollama, and llama.cpp.

## Quick Start

```bash
# Deploy all runtimes to intel
make deploy

# Run correctness tests
make test

# Run load tests and generate reports
make load
make report
```

## Runtimes

| Runtime | Port | Model Format | Status |
|---------|------|-------------|--------|
| realizar | 8081 | SafeTensors/APR | Active |
| ollama | 8082 | GGUF Q4_K_M | Active |
| llama.cpp | 8083 | GGUF Q4_K_M | Active |

<!-- PERFORMANCE_START -->
<!-- PERFORMANCE_END -->

## Testing

Correctness tests verify basic capabilities (math, code generation, explanation).
Load tests measure throughput, latency percentiles, and tokens/sec.

All results stored in `results/` and aggregated in `performance.md`.
