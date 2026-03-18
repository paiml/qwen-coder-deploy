# qwen-coder-deploy

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture" width="720"/>
</p>

Deploy and benchmark Qwen2.5-Coder-1.5B-Instruct across realizar, ollama, llama.cpp, and vLLM. All infrastructure managed via [forjar](https://github.com/paiml/forjar). Quantitative scoring via [probador](https://github.com/paiml/probador).

## Quick Start

```bash
# Yoga deployment (PRIMARY — RTX 4060 Laptop, isolated benchmarks)
make bench-yoga-serial       # All 4 runtimes, c=1 and c=4 (short prompt, isolated)
make bench-yoga-prod         # All 4 runtimes, production methodology (medium prompt, c=1-128)
make bench-yoga-prod-realizr # realizr only (c=1-128)
make bench-yoga-prod-vllm    # vLLM only (c=1-128)
make score                   # Generate scorecards (table format)
make score-prod              # Production methodology scorecards only
make teardown-yoga           # Stop all services

# 4090 deployment (deep profiling only — QLoRA training runs full-time)
make deploy-gpu              # Build + start via forjar
make nsys-gpu                # nsys kernel timeline
make ncu-gpu                 # ncu per-kernel roofline

# CPU deployment (intel host)
make deploy                  # Deploy via forjar to 192.168.50.100
make test                    # Correctness tests
make load                    # Load tests
```

## Runtimes

| Runtime | Port | Model Format | GPU |
|---------|------|-------------|-----|
| realizar (Sovereign AI Stack) | 8081 | GGUF Q4_K_M | CUDA (DP4A INT8 + FP8 prefill) |
| ollama | 8082 | GGUF Q4_K_M | CUDA (auto-detected) |
| llama.cpp | 8083 | GGUF Q4_K_M | CUDA (full offload, -ngl 99) |
| vLLM | 8084 | AWQ INT4 | CUDA (PagedAttention, CUTLASS GEMM) |

<!-- PERFORMANCE_START -->
## Performance — RTX 4060 Laptop (2026-03-17, PMAT-177/228, locked 1900MHz)

### Production Methodology (medium prompt ~102 tok, uniform:16,256 output, streaming, 60s)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 147.2 | 158.1 | 152.4 | 151.8 |
| 4 | 217.6 | 354.4 | 587.4 | 160.1 |
| 8 | 351.7 | 420.1 | 1,115.2 | 159.4 |
| 16 | 571.3 | 896.6 | 1,982.9 | 161.0 |
| 32 | 867.3 | 943.2 | 2,757.6 | 159.0 |
| 64 | 887.4 | — | 3,036.1 | — |
| 128 | 857.1 | — | 3,049.4 | — |

realizr uses CUDA_MAX_BATCH=16 (PMAT-223 workaround). Asymptote ~880 tok/s (hetero) / 1,010 (fixed:128). BATCH=32 bug at c≥20 medium (PMAT-221).

### Scorecards (probador llm score, PMAT-229 — definitive combined scoring)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 94 A | 97 A+ | 97 A+ | 78 B |
| 4 | 58 C | 73 B | 97 A+ | 58 C |
| 8 | 64 C+ | 65 C+ | 96 A+ | 58 C |
| 16 | 70 B | 72 B | 94 A | 58 C |
| 32 | 66 C+ | 51 C | 86 A- | 57 C |
| 64 | 68 C+ | — | 73 B | — |
| 128 | **67 C+** | — | 63 C+ | — |

### Asymptotes (PMAT-192/195/197)

| Runtime | Asymptote | Architecture |
|---------|-----------|-------------|
| vLLM | **3,050** tok/s | PagedAttention, continuous batching, CUTLASS GEMM |
| realizr | 880 tok/s (BATCH=16) | Batch-and-step, queue+batch=16 (workaround) |
| llama.cpp | 943 tok/s | Fixed 16 slots, ncols-templated GEMV |
| ollama | 160 tok/s | Serial FIFO |

realizr at BATCH=32: 1,500 tok/s but has quality bug at c≥20 (PMAT-221). BATCH=16 asymptote: 880 (hetero) / 1,010 (fixed:128).

Quality crossover: realizr **beats** vLLM at c=128 (67 C+ vs 63 C+) even at BATCH=16.

### Cross-Platform Decode (c=1, isolated, streaming)

| Platform | vLLM | realizr | llama.cpp | ollama |
|----------|------|---------|-----------|--------|
| **RTX 4060L** (24 SMs, 1900MHz) | **168.3** | 154.8 | 160.7 | 163.5 |
| RTX 4090 (128 SMs) | — | 411.7 | 436.9 | — |
| Jetson Orin (8 SMs, MAXN_SUPER) | — | **40.8** | 36.1 | — |

### Key Findings (PMAT-209→217, nsys/ncu profiling)

- **Three-level kernel architecture**: realizr (44+ kernels, CUDA graph M=1) → llama.cpp (35 kernels, ncols-templated GEMV) → vLLM (15 kernels, CUTLASS GEMM M=batch)
- **vLLM GEMM is batch-invariant**: 2,139→2,199µs (+2.8%) from c=1 to c=16 — throughput scales linearly with batch size
- **realizr CPU blocked 82.4%** in cuStreamSynchronize at c=4 (PMAT-217). M=1 graph invalid for M>1 → 771 kernel launches/step
- **Prompt length hurts realizr most**: −13-16% long penalty (c≥4) vs llama.cpp −2-9%, vLLM −0-10%. TTFT gap: 24× at c=16 (673ms vs 28ms). FP8 2-step prefill is sole cause (PMAT-227)
- **⚠️ Quality bug**: realizr batched prefill corrupts KV at long c≥9, medium c≥20 at BATCH=32 (PMAT-221/222). **Workaround: CUDA_MAX_BATCH=16** (PMAT-223)
- **Heterogeneity penalty**: 37-43% throughput loss from uniform:16,256 vs fixed:128 output. Paged KV (PMAT-052) is highest-ROI fix

See [performance.md](performance.md) for full history. See [gpu-performance-spec.md](docs/specifications/gpu-performance-spec.md) for detailed analysis.
<!-- PERFORMANCE_END -->

## Infrastructure

| File | Purpose |
|------|---------|
| `forjar-yoga-realizr.yaml` | Yoga realizr deployment (RTX 4060L) |
| `forjar-yoga-llamacpp.yaml` | Yoga llama.cpp deployment |
| `forjar-yoga-ollama.yaml` | Yoga ollama deployment |
| `forjar-yoga-vllm.yaml` | Yoga vLLM deployment |
| `forjar-yoga-teardown.yaml` | Stop all Yoga services |
| `forjar-gpu.yaml` | 4090 deployment (deep profiling) |
| `forjar.yaml` | CPU deployment (intel host, SSH) |
| `prompts/correctness.yaml` | 6-prompt correctness test suite |
| `scripts/nightly.sh` | Automated benchmark pipeline |
| `docs/specifications/gpu-performance-spec.md` | Performance specification (v4.5.0) |
| `docs/specifications/scoring.yaml` | Scoring contract v2.0.0 |

## Correctness

All 4 runtimes pass 6/6 correctness tests (math, code gen, explanation, JSON, SQL).

## Testing

Correctness tests verify basic capabilities (math, code generation, explanation).
Load tests measure throughput, latency percentiles, and tokens/sec via `probador llm load`.
Scoring via `probador llm score` with absolute thresholds (decode, TTFT, ITL, tail, errors).

All results stored in `results/` and aggregated in `performance.md`.
