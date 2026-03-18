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
## Performance — RTX 4060 Laptop (2026-03-18, PMAT-257, locked 1900MHz)

### Production Methodology (medium prompt ~102 tok, uniform:16,256 output, streaming, 60s)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 147.2 | 158.1 | 152.4 | 151.8 |
| 4 | **291.3** | 354.4 | 587.4 | 160.1 |
| 8 | **494.8** | 420.1 | 1,115.2 | 159.4 |
| 16 | **884.8** | 896.6 | 1,982.9 | 161.0 |
| 32 | 873.7 | 943.2 | 2,757.6 | 159.0 |
| 64 | 882.0 | — | 3,036.1 | — |
| 128 | 885.6 | — | 3,049.4 | — |

realizr uses CUDA_MAX_BATCH=16 + ITERATION_SCHEDULER=1 (PMAT-257). Iteration scheduler: +34-55% at c=4-16, TTFT −48-83%. Asymptote ~885 tok/s (hetero).

### Scorecards (probador llm score, PMAT-257 — iteration scheduler)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 95 A+ | 97 A+ | 97 A+ | 78 B |
| 4 | 70 B | 73 B | 97 A+ | 58 C |
| 8 | 75 B | 65 C+ | 96 A+ | 58 C |
| 16 | 78 B | 72 B | 94 A | 58 C |
| 32 | 67 C+ | 51 C | 86 A- | 57 C |
| 64 | 68 C+ | — | 73 B | — |
| 128 | **68 C+** | — | 63 C+ | — |

realizr overtakes llama.cpp at c=8 (75 vs 65) and holds through c=32 (67 vs 51). Quality crossover: realizr **beats** vLLM at c=128 (68 C+ vs 63 C+).

### Asymptotes (PMAT-192/195/197/257)

| Runtime | Asymptote | Architecture |
|---------|-----------|-------------|
| vLLM | **3,050** tok/s | PagedAttention, continuous batching, CUTLASS GEMM |
| realizr | 885 tok/s (iter sched) | Iteration scheduler, BATCH=16 |
| llama.cpp | 943 tok/s | Fixed 16 slots, ncols-templated GEMV |
| ollama | 160 tok/s | Serial FIFO |

Iteration scheduler hits asymptote at c=16 (was c=32 with batch-and-step). TTFT: 48ms at c=16 (was 279ms).

### Cross-Platform Decode (c=1, isolated, streaming)

| Platform | vLLM | realizr | llama.cpp | ollama |
|----------|------|---------|-----------|--------|
| **RTX 4060L** (24 SMs, 1900MHz) | **168.3** | 154.8 | 160.7 | 163.5 |
| RTX 4090 (128 SMs) | — | 411.7 | 436.9 | — |
| Jetson Orin (8 SMs, MAXN_SUPER) | — | **40.8** | 36.1 | — |

### Key Findings (PMAT-209→257)

- **Iteration scheduler (PMAT-257)**: +34-55% aggregate, −48-83% TTFT at c=4-16 — zero code changes, just env var
- **Three-level kernel architecture**: realizr (44+ kernels, CUDA graph M=1) → llama.cpp (35 kernels, ncols-templated GEMV) → vLLM (15 kernels, CUTLASS GEMM M=batch)
- **vLLM GEMM is batch-invariant**: 2,139→2,199µs (+2.8%) from c=1 to c=16 — throughput scales linearly with batch size
- **Decode/ITL crossover at c=64** (PMAT-255): realizr wins per-request decode 1.14-2.35× and ITL 0.43-0.87× at c=64-128
- **Heterogeneity penalty** (PMAT-254): 31-42% loss from uniform output. Paged KV (PMAT-052) recovers 1.72× at c=16
- **Phase 1 readiness** (PMAT-256): Paged KV ready, scheduler is blocker (~1,000-1,400 LOC total)
- **⚠️ Quality bug**: BATCH=32 corrupts KV at c≥20 medium (PMAT-221). **Workaround: CUDA_MAX_BATCH=16** (PMAT-223)

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
| `docs/specifications/gpu-performance-spec.md` | Performance specification (v5.1.0) |
| `docs/specifications/scoring.yaml` | Scoring contract v2.0.0 |

## Correctness

All 4 runtimes pass 6/6 correctness tests (math, code gen, explanation, JSON, SQL).

## Testing

Correctness tests verify basic capabilities (math, code generation, explanation).
Load tests measure throughput, latency percentiles, and tokens/sec via `probador llm load`.
Scoring via `probador llm score` with absolute thresholds (decode, TTFT, ITL, tail, errors).

All results stored in `results/` and aggregated in `performance.md`.
