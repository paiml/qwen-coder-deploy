# qwen-coder-deploy

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture" width="720"/>
</p>

Deploy and benchmark Qwen2.5-Coder-1.5B-Instruct across four inference runtimes. Infrastructure via [forjar](https://github.com/paiml/forjar). Scoring via [probador](https://github.com/paiml/probador).

## Quick Start

```bash
make bench-yoga-prod         # All 4 runtimes, production methodology
make score-prod              # Production scorecards
make teardown-yoga           # Stop all services
```

## Runtimes

| Runtime | Port | Format | Quantization |
|---------|------|--------|-------------|
| [realizar](https://github.com/paiml/realizar) | 8081 | GGUF | Q4_K_M (DP4A + FP8) |
| ollama | 8082 | GGUF | Q4_K_M |
| llama.cpp | 8083 | GGUF | Q4_K_M |
| vLLM | 8084 | AWQ | INT4 (CUTLASS) |

## Performance

RTX 4060 Laptop, 1900MHz locked, production methodology (medium prompt, uniform output, streaming, 60s).

### Throughput (tok/s aggregate, Mar 21)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 149 | 160 | 154 | 163 |
| 4 | 325 | 351 | 598 | 635 |
| 8 | **525** | 419 | 1,142 | -- |
| 16 | **931** | 912 | 2,037 | -- |
| 32 | 1,600 | **1,949** | 2,998 | -- |

### Quality Scores

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 94 A | 93 A | 98 A+ | 77 B |
| 8 | **76 B** | 66 C+ | 97 A+ | 57 C |
| 32 | **75 B** | 63 C+ | 87 A- | 57 C |
| 128 | **66 C+** | -- | 64 C+ | -- |

realizr overtakes llama.cpp at c=8 (+25% aggregate) and beats vLLM on quality at c=128.

### Asymptotes

| Runtime | Peak | Architecture |
|---------|------|-------------|
| vLLM | 3,163 tok/s | PagedAttention + continuous batching + CUTLASS |
| realizr | 1,511 tok/s | Iteration scheduler + BATCH=32 |
| llama.cpp | 923 tok/s | Fixed 16 slots |
| ollama | 157 tok/s | Serial |

### Cross-Platform (c=1 decode tok/s)

| Platform | realizr | llama.cpp | vLLM |
|----------|---------|-----------|------|
| RTX 4060L (24 SMs) | 149 | 159 | 154 |
| RTX 4090 (128 SMs) | 412 | 437 | -- |
| Jetson Orin (8 SMs) | 25 | -- | -- |

## Key Results

- **Tensor graph dispatch (PMAT-291)**: +8.5% at c=4. Pure Rust graph executor + Q8 activation cache (PMAT-294)
- **GPU kernels within 8% of vLLM** (7.4ms vs 6.8ms per step). 16 kernel fusion approaches falsified — 2-kernel Q8+DP4A pattern is optimal
- **Iteration scheduler**: +71% throughput, 0% errors, production-stable (10-min sustained, 6,843 requests)
- **Prompt-sensitivity**: realizr -24-26% long penalty (plateau), vLLM -9% peak then reverses, llama.cpp invariant
- **Quality crossover**: realizr beats vLLM at c=128 on combined score (decode + ITL advantage)

Full analysis: [gpu-performance-spec.md](docs/specifications/gpu-performance-spec.md) (v5.34.0, 295 PMAT items) | [performance.md](performance.md)

## Infrastructure

| File | Purpose |
|------|---------|
| `forjar-yoga-*.yaml` | Yoga deployment configs (RTX 4060L) |
| `forjar-gpu.yaml` | 4090 profiling deployment |
| `forjar.yaml` | CPU deployment (intel host) |
| `prompts/correctness.yaml` | 6-prompt correctness suite |
| `scripts/nightly.sh` | Automated benchmark pipeline |
| `docs/specifications/gpu-performance-spec.md` | Performance spec v5.34.0 |
| `docs/specifications/scoring.yaml` | Scoring contract v2.0.0 |

## Testing

```bash
make test          # Correctness (6/6 all runtimes)
make load          # Load tests
make score-gate    # CI quality gate (fail if any runtime below C)
```

Results in `results/`, aggregated in `performance.md`.
