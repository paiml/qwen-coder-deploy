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

### Throughput (tok/s)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 147 | 158 | 152 | 149 |
| 4 | 291 | 352 | 587 | 157 |
| 8 | 495 | 417 | 1,114 | 157 |
| 16 | 869 | 894 | 1,983 | 156 |
| 32 | **1,469** | 923 | 2,899 | 153 |
| 64 | **1,485** | -- | 3,146 | -- |
| 128 | **1,511** | -- | 3,163 | -- |

### Quality Scores

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 94 A | 93 A | 98 A+ | 77 B |
| 8 | **76 B** | 66 C+ | 97 A+ | 57 C |
| 32 | **75 B** | 63 C+ | 87 A- | 57 C |
| 128 | **66 C+** | -- | 64 C+ | -- |

realizr overtakes llama.cpp at c=8 and beats vLLM on quality at c=128.

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

- **GPU kernels within 8% of vLLM** (7.4ms vs 6.8ms per step). The 2x throughput gap is CPU kernel dispatch overhead
- **Iteration scheduler**: +71% throughput, 0% errors, production-stable (10-min sustained, 6,843 requests)
- **Prompt-sensitivity**: realizr -24-26% long penalty (plateau), vLLM -9% peak then reverses, llama.cpp invariant
- **Quality crossover**: realizr beats vLLM at c=128 on combined score (decode + ITL advantage)

Full analysis: [gpu-performance-spec.md](docs/specifications/gpu-performance-spec.md) (v5.28.0, 290 PMAT items) | [performance.md](performance.md)

## Infrastructure

| File | Purpose |
|------|---------|
| `forjar-yoga-*.yaml` | Yoga deployment configs (RTX 4060L) |
| `forjar-gpu.yaml` | 4090 profiling deployment |
| `forjar.yaml` | CPU deployment (intel host) |
| `prompts/correctness.yaml` | 6-prompt correctness suite |
| `scripts/nightly.sh` | Automated benchmark pipeline |
| `docs/specifications/gpu-performance-spec.md` | Performance spec v5.28.0 |
| `docs/specifications/scoring.yaml` | Scoring contract v2.0.0 |

## Testing

```bash
make test          # Correctness (6/6 all runtimes)
make load          # Load tests
make score-gate    # CI quality gate (fail if any runtime below C)
```

Results in `results/`, aggregated in `performance.md`.
