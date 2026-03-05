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

## Jetson Orin (sm_87, Qwen2.5-Coder-1.5B Q4_K_M)

### Load Test Results (2026-03-06, c=1, 60s, 5s warmup)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) | Tok/s |
|---------|-------------|--------------|---------------|-------------|-------|
| realizr | 16.7 | 25.8 | 3,956 | 60 | 11.1 |
| llama.cpp | 32.3 | 2,092 | 49 | 31 | 32.1 |
| ollama | 12.5 | 227 | 449 | 80 | 10.1 |

**Gap vs llama.cpp:** Decode 1.93x, Prefill 81x (GEMV vs cuBLAS GEMM), TTFT 81x

### Config: `DP4A_Q4K=1 DP4A_Q6K=1 MWV_Q6K=1 MWV_WARPS=3`

### Optimization Sweeps

**DP4A impact:**
- No DP4A: 12.6 tok/s (baseline)
- +DP4A_Q4K: 14.5 (+15%)
- +DP4A_Q4K +DP4A_Q6K: **16.7** (+33%)

**Warp count (MWV_WARPS):**
- 2 warps: 14.5, **3 warps: 16.7**, 4 warps: 15.2

### Decode Timing (GRAPH-TIMING, per token)
- h2d async: 40µs, graph launch: 18µs, argmax+sync: **57,300µs** (99.8%)
- Effective bandwidth: 14.8 GB/s (7.2% of 204 GB/s peak)
- Tracking: [GH-131](https://github.com/paiml/realizar/issues/131)

## GPU Profiling (2026-03-04, Nsight Systems)

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
