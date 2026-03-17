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
## Performance — RTX 4060 Laptop (2026-03-17, PMAT-224, locked 1900MHz)

### Production Methodology (medium prompt ~102 tok, uniform:16,256 output, streaming, 60s)

| c | realizr | llama.cpp | vLLM | r/llama | r/vLLM |
|---|---------|-----------|------|---------|--------|
| 1 | 147.3 | 158.8 | 152.2 | 0.93× | 0.97× |
| 4 | 315.6 | 367.7 | 583.9 | 0.86× | 0.54× |
| 8 | **559.6** | 429.4 | 1,119.8 | **1.30×** | 0.50× |
| 16 | 1,008.1 | 1,120.1 | 2,043.8 | 0.90× | 0.49× |

All runtimes re-measured PMAT-224 (realizr Mar 16 binary +46-72%, llama.cpp rebuilt latest HEAD +25% at c=16). **realizr beats llama.cpp at c=8** (1.30×).

### CUDA_MAX_BATCH=16 Workaround (PMAT-223, correct at all c)

| c | BATCH=16 (tok/s) | BATCH=32 (tok/s) | Status |
|---|-------------------|-------------------|--------|
| 1 | 147.2 | 147.3 | ✅ identical |
| 16 | 1,009.5 | 1,008.1 | ✅ identical |
| 32 | 1,009.9 | ⚠️ BUG | BATCH=16 fixes |
| 128 | 1,010.3 | ⚠️ BUG | BATCH=16 fixes |

### Asymptotes

| Runtime | Asymptote | Architecture |
|---------|-----------|-------------|
| vLLM | **3,050** tok/s | PagedAttention, continuous batching, CUTLASS GEMM |
| llama.cpp | ~1,120 tok/s (latest HEAD) | Fixed 16 slots, ncols-templated GEMV |
| realizr (BATCH=16) | 1,010 tok/s | Batch-and-step, bug-free at all c |
| realizr (BATCH=32) | 1,169+ tok/s (c=19 max safe) | Batch-and-step, bug above c=19 medium |
| ollama | 160 tok/s | Serial FIFO |

### Cross-Platform Decode (c=1, isolated, streaming)

| Platform | vLLM | realizr | llama.cpp | ollama |
|----------|------|---------|-----------|--------|
| **RTX 4060L** (24 SMs, 1900MHz) | **168.3** | 154.8 | 160.7 | 163.5 |
| RTX 4090 (128 SMs) | — | 411.7 | 436.9 | — |
| Jetson Orin (8 SMs, MAXN_SUPER) | — | **40.8** | 36.1 | — |

### Key Findings (PMAT-209→217, nsys/ncu profiling)

- **Three-level kernel architecture**: realizr (44+ kernels, CUDA graph M=1) → llama.cpp (35 kernels, ncols-templated GEMV) → vLLM (15 kernels, CUTLASS GEMM M=batch)
- **vLLM GEMM is batch-invariant**: 2,139→2,199µs (+2.8%) from c=1 to c=16 — throughput scales linearly with batch size
- **Mar 16 binary: +46-72% scheduling improvement** (PMAT-224). c=4: 216→316, c=8: 355→560, c=16: 587→1008 tok/s. realizr now beats llama.cpp at c≥8
- **Prompt length hurts**: realizr/vLLM gap widens from 0.30× (medium c=16) to 0.24× (long c=16). TTFT gap: 14.4× at c=8 with long prompts (PMAT-220)
- **⚠️ Quality bug**: realizr batched prefill corrupts KV at long c≥9, medium c≥20 at BATCH=32 (PMAT-221). **Workaround: CUDA_MAX_BATCH=16** eliminates bug, −33% asymptote (1500→1010 tok/s) (PMAT-223)

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
| `docs/specifications/gpu-performance-spec.md` | Performance specification (v3.75.0) |
| `docs/specifications/scoring.yaml` | Scoring contract v2.0.0 |

## Correctness

All 4 runtimes pass 6/6 correctness tests (math, code gen, explanation, JSON, SQL).

## Testing

Correctness tests verify basic capabilities (math, code generation, explanation).
Load tests measure throughput, latency percentiles, and tokens/sec via `probador llm load`.
Scoring via `probador llm score` with absolute thresholds (decode, TTFT, ITL, tail, errors).

All results stored in `results/` and aggregated in `performance.md`.
