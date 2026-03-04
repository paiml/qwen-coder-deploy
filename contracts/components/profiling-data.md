# Profiling Data: GPU Decoder Performance

**Parent Spec:** [REALIZAR-GPU-PERF-001](../gpu-performance-spec.md)
**Status:** Reference Document
**Date:** 2026-03-04

---

## 1. GEMV Profiling (Pre-Fix)

### Memory Access Pattern Analysis

| Metric | Non-Coalesced | Target (Coalesced) |
|--------|--------------|-------------------|
| Memory bandwidth utilization | 1.4% | >90% |
| GEMV latency (1×4096×4096) | 4.41ms | <0.05ms |
| Stride between thread reads | 16KB | 4 bytes |
| Transactions per warp | ~128 | 1 |

### Root Cause Visualization

```
Non-coalesced (current):
  Warp 0: A[0,0] A[32,0] A[64,0] ...    ← 16KB stride
  Warp 1: A[0,1] A[32,1] A[64,1] ...    ← Same stride

Coalesced (target):
  Block 0: A[row, 0] A[row, 1] ... A[row, 255]  ← 4-byte stride (COALESCENT)
```

---

## 2. Nsight Systems Kernel Profile (2026-03-04)

**Source:** `results/nsys-apr-gpu-kernels-20260304.txt` (Qwen2.5-Coder-1.5B Q4_K_M, RTX 4090)

### CUDA GPU Kernel Summary

| Kernel | Time (%) | Total Time (ms) | Instances | Avg (µs) | Med (µs) | Min (µs) | Max (ms) |
|--------|----------|-----------------|-----------|----------|----------|----------|----------|
| `mwv_q4k_gemv` | **46.0%** | 528.7 | 53,592 | 9.9 | 4.5 | 2.3 | 1.3 |
| `q6k_gemv_warp_reduce` | **31.9%** | 367.4 | 9,251 | 39.7 | 39.2 | 6.4 | 1.8 |
| `multi_warp_attention_indirect` | **9.3%** | 107.1 | 8,932 | 12.0 | 11.2 | 1.8 | 1.3 |
| `rmsnorm_vectorized` | 5.3% | 60.8 | 18,183 | 3.3 | 3.1 | 3.0 | 1.3 |
| `residual_add` | 3.8% | 43.3 | 44,660 | 1.0 | 1.0 | 0.8 | 1.2 |
| `rope_neox_indirect` | 1.7% | 19.6 | 17,864 | 1.1 | 1.0 | 1.0 | 1.3 |
| `kv_cache_scatter_indirect` | 1.3% | 15.3 | 17,864 | 0.9 | 0.9 | 0.8 | 0.006 |
| `fused_swiglu` | 0.7% | 8.1 | 8,932 | 0.9 | 0.9 | 0.9 | 0.006 |

**Key Findings:**
- GEMV kernels dominate: **77.9%** of GPU time (`mwv_q4k_gemv` 46% + `q6k_gemv_warp_reduce` 31.9%)
- Attention is only 9.3% of kernel time (not 76% — profile telemetry counts host-side overhead)
- `fused_swiglu` is fast (0.7%) — GPU fusion working correctly
- `residual_add` has highest instance count (44,660) at <1µs each — minimal overhead

---

## 2a. Per-Operation Telemetry (2026-03-02)

**Source:** `results/profile-gpu-20260302.txt` (Qwen2.5-Coder-1.5B Q4_K_M, RTX 4090)

Host-side profiling captures end-to-end operation time including kernel launch overhead.

### Operation Hotspots

| # | Operation | Time (µs) | % of Decode | Bottleneck | Bandwidth |
|---|-----------|-----------|-------------|------------|-----------|
| 1 | AttentionScore | 88,390 | **76.0%** | MEMORY | 0.2 GB/s (eff 0%) |
| 2 | RmsNorm | 17,118 | **14.7%** | MEMORY | 1.2 GB/s (eff 0%) |
| 3 | QkvProjection | 2,755 | 2.4% | MEMORY | — |
| 4 | GateProjection | 1,838 | 1.6% | MEMORY | — |
| 5 | RopeEmbedding | 1,637 | 1.4% | COMPUTE | 4.2 GB/s |
| 6 | OutputProjection | 965 | 0.8% | MEMORY | — |
| 7 | DownProjection | 938 | 0.8% | MEMORY | — |
| 8 | Residual1 | 863 | 0.7% | MEMORY | — |
| 9 | Activation | 837 | 0.7% | COMPUTE | — |
| 10 | Residual2 | 830 | 0.7% | MEMORY | — |
| 11 | LmHead | 155 | 0.1% | MEMORY | — |

### Category Summary

| Category | % of Decode |
|----------|-------------|
| Attention | **80.6%** |
| Norm | **14.7%** |
| FFN | 3.2% |
| Other | 1.5% |

### Kernel Launch Overhead (F-PROFILE-009)

| Metric | Value |
|--------|-------|
| Overhead | **128,484µs** |
| % of decode time | **52.5%** |
| Status | **WARNING: >20% — kernel fusion needed** |

This 52.5% overhead explains the gap between Nsight kernel time and host-side telemetry. The actual CUDA kernels are fast, but launch overhead dominates. See root cause §6 and PMAT-015.

### Roofline Analysis

| Metric | Value |
|--------|-------|
| Hardware | NVIDIA RTX 4090 (82,580 GFLOPS, 1,008 GB/s) |
| Achieved compute | 337.3 GFLOPS (0.4% efficiency) |
| Achieved bandwidth | **84.3 GB/s (8.4% efficiency)** |
| Arithmetic intensity | 4.0 FLOP/byte (threshold: 82.0) |
| Classification | **MEMORY BOUND** |
| Performance grade | **D** — significant optimization needed |

### Generation Performance (Profile Run)

| Metric | Value |
|--------|-------|
| Decode throughput | **130.7 tok/s** |
| Prefill throughput | 82.7 tok/s |
| P50 latency | 245.6ms |
| P95 latency | 264.2ms |
| Model | Qwen2 (28 layers, hidden=1,536) |

---

## 3. PCIe Transfer Elimination (Fixes 2-4)

| Fix | Transfers Eliminated | Per Layer | Per Model (28L) |
|-----|---------------------|-----------|-----------------|
| Fix 2: SwiGLU fusion | 3 | GPU→CPU→GPU | 84 |
| Fix 3: GELU fusion | 2 | GPU→CPU→GPU | 56 |
| Fix 4: RMSNorm+Residual | 4+ | GPU→CPU→GPU | 112+ |
| **Total eliminated** | | | **252+** |

---

## 4. Memory Bandwidth Analysis

### RTX 4090 Roofline

```
Peak Compute: 82.6 TFLOPS (FP32)
Peak Bandwidth: 1008 GB/s
Arithmetic Intensity threshold: 82 FLOP/byte

GEMV (M=1): ~2 FLOP/byte → MEMORY BOUND
GEMM (M>64): ~128 FLOP/byte → COMPUTE BOUND
```

### Q4K Dequantization Bandwidth

| Operation | Bytes Read | FLOPS | AI (FLOP/byte) | Bound |
|-----------|-----------|-------|-----------------|-------|
| Q4K GEMV (1×4096) | 2.3MB | 8.2K | 0.004 | Memory |
| Q4K GEMM (64×4096) | 2.3MB | 524K | 0.23 | Memory |
| Q4K GEMM (256×4096) | 2.3MB | 2.1M | 0.91 | Transitional |

---

## 5. KV Cache Performance

### ContiguousKVCache vs Vec<Vec> (PARITY-005)

| Metric | Vec<Vec> | ContiguousKVCache | Speedup |
|--------|----------|-------------------|---------|
| 1000 iterations | 1.09s | 65µs | 16,640x |
| Cache-line alignment | None | 64-byte | ∞ |
| Heap allocations | O(layers×heads) | O(1) | O(layers×heads) |

### Q8 KV Cache Memory (QWEN-007)

| Precision | Memory per KV pair | Reduction |
|-----------|-------------------|-----------|
| FP32 | 4 bytes | 1x |
| Q8 (with scales) | 1.125 bytes | 3.56x |
| Q4 (theoretical) | 0.5625 bytes | 7.1x |

---

## 6. Batch Scheduling Profiling

### spawn_blocking Impact (Fix 3)

Before: Sync GPU inference blocks tokio executor threads
After: `tokio::task::spawn_blocking` isolates GPU work

| Metric | Before | After |
|--------|--------|-------|
| Concurrent requests | 1 (blocked) | Bounded by channel |
| Executor thread starvation | Yes | No |
| Request latency variance | High (queue head) | Low (isolated) |

### Queue-Based Dispatch (Fix 5)

| Configuration | Value |
|---------------|-------|
| Channel type | Bounded MPSC |
| Queue depth | 32 slots |
| Backpressure | Channel full → reject |
| Dispatch strategy | FIFO with priority |

---

## 7. Warp Count Sweep

### Optimal Warp Configuration (GEMV)

| Warps/Block | Threads | Occupancy | Throughput | Notes |
|-------------|---------|-----------|------------|-------|
| 2 | 64 | 25% | Baseline | Underutilized |
| 4 | 128 | 50% | +15% | Good for memory-bound |
| 8 | 256 | 75% | +18% | **Selected** |
| 16 | 512 | 100% | +19% | Diminishing returns |
| 32 | 1024 | 100% | +17% | Register pressure |

Per [Volkov10]: For memory-bound GEMV, increasing occupancy beyond 50% provides <20% improvement. 256 threads selected as optimal balance.

---

## 8. Batch Size Throughput Scaling

### Internal Microbenchmarks (Feb 2026)

| Batch Size (M) | Throughput (tok/s) | Ollama Ratio (internal) | Bound |
|----------------|-------------------|------------------------|-------|
| 1 | ~180 | 0.75x | Memory |
| 8 | 740.5 | 2.54x | Memory |
| 16 | 583.6 | 2.01x | Transitional |
| 64 | TBD | TBD | Compute |

### Competition Load Test (Mar 2026, c=4, 60s)

| Runtime | GPU (tok/s) | CPU (tok/s) | GPU/CPU |
|---------|-------------|-------------|---------|
| llama.cpp | 1,013.6 | 218.5 | 4.64x |
| ollama | 607.9 | 149.5 | 4.07x |
| realizar (safetensors) | 96.5 | 28.3 | 3.41x |
| realizar (GGUF) | 25.8 | 23.0 | 1.12x |
| realizar (APR native) | 0.0 (broken) | 9.5 | N/A |

*Note: Discrepancy between internal 740.5 tok/s (M=8) and competition 96.5 tok/s is due to batched microbenchmark vs concurrent load test conditions. See baselines.md §5a.*

---

## External Profiling Data

For additional GPU profiling data including warp count sweeps and decode timing breakdowns, see:
- `batuta/book/src/appendix/benchmarks.md` — Profiling data appendix
- `qwen-coder-deploy/performance.md` — Snapshot tables

---

## References

- [Volkov10] Volkov, V. (2010). "Better Performance at Lower Occupancy." GTC 2010.
- [Williams09] Williams, S., et al. (2009). "Roofline Model." CACM 52(4).
- [Jain91] Jain, R. (1991). "The Art of Computer Systems Performance Analysis." Wiley.
