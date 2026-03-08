# LLM Inference Performance

## GPU (RTX 4090, Qwen2.5-Coder-1.5B Q4_K_M)

### Competition Benchmarks (2026-03-04, c=4, 60s, 5s warmup)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Tok/s | Decode tok/s | Error Rate |
|------|---------|-------------|-----|----------|----------|----------|-------|-------------|------------|
| 2026-03-04 | llama.cpp | 4 | 7.4 | 537.8 | 565.4 | 588.7 | 948.2 | 238.0 | 0% |
| 2026-03-04 | ollama | 4 | 4.4 | 899.5 | 938.7 | 947.2 | 568.9 | 142.3 | 0% |
| 2026-03-04 | realizar-safetensors | 4 | 1.4 | 2,643.7 | 4,428.2 | 4,469.5 | 167.1 | 43.3 | 0% |
| 2026-03-04 | realizar-gguf | 4 | 1.3 | 3,259.5 | 4,078.2 | 4,162.8 | 150.7 | 39.1 | 0% |
| 2026-03-04 | realizar-apr | 4 | 1.4 | 2,728.3 | 3,560.9 | 4,057.3 | 143.3 | 39.9 | 0% |

### Previous (2026-03-03, c=4, 60s, 3 runs)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Tok/s | Error Rate |
|------|---------|-------------|-----|----------|----------|----------|-------|------------|
| 2026-03-03 | llama.cpp | 4 | 7.92 | 504 | 521 | 528 | 1,013.6 | 0% |
| 2026-03-03 | ollama | 4 | 4.75 | 839 | 872 | 887 | 607.9 | 0% |
| 2026-03-03 | realizar-safetensors | 4 | 0.75 | 5,274 | 7,480 | 9,013 | 96.5 | 0% |
| 2026-03-03 | realizar-gguf | 4 | 0.20 | 18,989 | 24,229 | 24,258 | 25.8 | 0% |
| 2026-03-03 | realizar-apr | 4 | 0.00 | N/A | N/A | N/A | 0.0 | 100% |

### Historical (2026-03-02)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Requests |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|----------|
| 2026-03-02 | realizar-gpu | 4 | 10.2 | 392.6 | 599.6 | 705.2 | 392.6 | 10.2 | 609 |
| 2026-03-02 | ollama-gpu | 4 | 120.3 | 30.8 | 48.8 | 72.0 | 30.8 | 240.5 | 7216 |
| 2026-03-02 | llamacpp-gpu | 4 | 328.2 | 11.4 | 15.6 | 18.5 | 11.4 | 656.4 | 19692 |

## CPU (Intel EPYC, 192.168.50.100, Qwen2.5-Coder-1.5B Q4_K_M)

### Competition Benchmarks (2026-03-03, c=4, 60s, 3 runs)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Error Rate |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|------------|
| 2026-03-03 | llama.cpp | 4 | 1.71 | 2,340 | 2,381 | 2,389 | 2,340 | 218.5 | 0% |
| 2026-03-03 | ollama | 4 | 1.17 | 3,356 | 3,782 | 3,817 | 3,356 | 149.5 | 0% |
| 2026-03-03 | realizar-safetensors | 4 | 0.22 | 18,110 | 18,293 | 18,317 | 18,110 | 28.3 | 0% |
| 2026-03-03 | realizar-gguf | 4 | 0.18 | 20,007 | 30,699 | 31,408 | 20,007 | 23.0 | 0% |
| 2026-03-03 | realizar-apr | 4 | 0.07 | 53,263 | 54,537 | 54,537 | 53,263 | 9.5 | 0% |

### Historical (2026-03-01)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Requests |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|----------|
| 2026-03-01 | realizar-apr | 4 | 0.4 | 12807.2 | 12950.4 | 12963.4 | 12807.2 | 6.9 | 13 |
| 2026-03-01 | realizar-gguf | 4 | 1.5 | 2510.7 | 3839.4 | 3876.5 | 2510.6 | 1.5 | 45 |

## Isolated Streaming (c=1, 60s, 5s warmup, stream=true) — PMAT-040, 2026-03-08

### RTX 4090 (128 SMs, sm_89)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) | Latency P50 (ms) |
|---------|-------------|--------------|---------------|-------------|-----------------|
| realizr | **411.7** | 1,734 | 58.8 | 2.4 | 368.1 |
| llama.cpp | 436.9 | 17,620 | 5.8 | 2.3 | — |

**Decode gap: 1.06x (near parity).** Improvement: 1.55x (266→412 tok/s) via Flash Decode chunk\_size 128→32.
Prefill gap: 10.2x (HGEMM FP16 reads vs llama.cpp fused Q4K GEMM).

### Jetson Orin Nano Super (8 SMs, sm_87, locked clocks)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) | Latency P50 (ms) |
|---------|-------------|--------------|---------------|-------------|-----------------|
| realizr | **36.3** | 449.3 | 227.0 | 27.6 | 3,734.6 |
| llama.cpp | 33.1 | 2,492.3 | 40.9 | 30.2 | 3,878.3 |

**Decode: realizr 10% FASTER than llama.cpp (0.91x).** Improvement: 10.7% (32.7→36.3 tok/s).
Prefill gap: 5.5x (HGEMM FP16 reads vs fused Q4K GEMM, isolated mode only).

### Config: GpuProfile auto-detect (no env vars), fused\_gate\_up=true, FLASH\_DECODE\_CHUNK\_SIZE=32

### Optimization History (GH-131/173/174/176, PMAT-033→040)

| Step | Jetson Decode | 4090 Decode | Date |
|------|--------------|-------------|------|
| Baseline (MWV DP4A) | 16.7 | 128.4 | Mar 5 |
| +locked clocks | 21.4 | — | Mar 6 |
| +HW DP4A Q4K (GH-176) | 27.8 | 162.8 | Mar 6 |
| +grid 16 blocks/SM | 33.7 | — | Mar 7 |
| +HGEMM prefill + graph | 32.7 | 266.3 | Mar 7 |
| +Flash Decode chunk=32 | **36.3** | **411.7** | Mar 8 |

### Bandwidth Utilization (corrected: 67 GB/s peak for Orin Nano Super)

| Runtime | BW (GB/s) | % of Peak |
|---------|----------|-----------|
| realizr Q4K GEMV | 20.5 | 30.6% |
| realizr total decode | 18.2 | 27.1% |
| llama.cpp total decode | 27.4 | 40.9% |

Tracking: [GH-131](https://github.com/paiml/realizar/issues/131)

## GPU Profiling — BrickProfiler (2026-03-06, C-GDP-001 Contract)

### Corrected Brick Breakdown (RTX 4090, Immediate Sync, CUDA\_GRAPH\_DISABLE=1)

After fixing 18 hardcoded values in the cbtop pipeline ([aprender#426](https://github.com/paiml/aprender/pull/426)),
BrickProfiler now reports real per-kernel GPU timing:

| Brick | Per-Call (µs) | Per-Decoded-Token (µs) | % of Decode |
|-------|--------------|----------------------|-------------|
| AttentionScore | 67.5 | 1,891 | 17.7% |
| GateProjection | 53.2 | 1,489 | 13.9% |
| RmsNorm | 25.2 | 1,434 | 13.4% |
| DownProjection | 42.1 | 1,178 | 11.0% |
| QkvProjection | 35.1 | 982 | 9.2% |
| Activation | 30.6 | 856 | 8.0% |
| Residual2 | 24.2 | 678 | 6.3% |
| LmHead | 594.2 | 594 | 5.6% |
| OutputProjection | 21.0 | 587 | 5.5% |
| RopeEmbedding | 19.6 | 549 | 5.1% |
| Residual1 | 15.9 | 446 | 4.2% |

Contract: `gpu-decode-profiling-v1` v2.0.0 — 15 falsification tests, all PASS.

Fixes:
- [realizar#137](https://github.com/paiml/realizar/pull/137): Force eager decode when profiler active (CUDA graphs hide brick timing)
- [aprender#426](https://github.com/paiml/aprender/pull/426): 18 hardcoded BrickScore values replaced with real profiler data

### Serial Baseline (2026-03-06, c=1, isolated, CUDA\_GRAPH\_DISABLE=1)

| Runtime | Decode tok/s |
|---------|-------------|
| realizar | 162.7 |
| llama.cpp | 262.7 |
| **Gap** | **1.61x** |

**Note:** This was before Flash Decode chunk\_size=32 (PMAT-040). Current graph-mode decode: 411.7 tok/s (1.06x gap).

## GPU Profiling — Nsight Systems (2026-03-04)

### CUDA Kernel Time Distribution

| Kernel | Time (%) | Instances | Avg (µs) | Med (µs) |
|--------|----------|-----------|----------|----------|
| mwv_q4k_gemv | 46.0% | 53,592 | 9.9 | 4.5 |
| q6k_gemv_warp_reduce | 31.9% | 9,251 | 39.7 | 39.2 |
| multi_warp_attention_indirect | 9.3% | 8,932 | 12.0 | 11.2 |
| rmsnorm_vectorized | 5.3% | 18,183 | 3.3 | 3.1 |
| residual_add | 3.8% | 44,660 | 1.0 | 1.0 |
| rope_neox_indirect | 1.7% | 17,864 | 1.1 | 1.0 |
| kv_cache_scatter_indirect | 1.3% | 17,864 | 0.9 | 0.9 |
| fused_swiglu | 0.7% | 8,932 | 0.9 | 0.9 |

Source: `results/nsys-apr-gpu-kernels-20260304.txt`

### Per-Operation Telemetry (2026-03-02)

| Operation | Time (µs) | % of Decode | Bottleneck |
|-----------|-----------|-------------|------------|
| AttentionScore | 88,390 | 76.0% | MEMORY |
| RmsNorm | 17,118 | 14.7% | MEMORY |
| QkvProjection | 2,755 | 2.4% | MEMORY |
| GateProjection | 1,838 | 1.6% | MEMORY |
| RopeEmbedding | 1,637 | 1.4% | COMPUTE |
| OutputProjection | 965 | 0.8% | MEMORY |
| DownProjection | 938 | 0.8% | MEMORY |

**Kernel launch overhead:** 128,484µs (52.5% of decode time)
**Memory efficiency:** 8.4% (Grade D)
**Decode throughput (profile run):** 130.7 tok/s

Source: `results/profile-gpu-20260302.txt`
