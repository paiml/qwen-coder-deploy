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

## 2. GPU Kernel Profiling (Post-Fix)

### APR Q4 Forward Pass (Qwen2.5-Coder-0.5B)

**After Fixes 1-4 (GPU-resident):**

| Operation | Time (ms) | PCIe Transfers | Notes |
|-----------|-----------|----------------|-------|
| Embedding lookup | 0.01 | 0 | GPU-resident |
| RMSNorm (per layer) | 0.005 | 0 | GPU ptr-based |
| Q4K Attention GEMV | 0.12 | 0 | Fused dequant |
| SwiGLU FFN | 0.08 | 0 | Fused kernel |
| Residual add | 0.003 | 0 | GPU in-place |
| Output norm | 0.005 | 0 | GPU ptr-based |
| LM head | 0.05 | 0 | GPU matmul |
| **Total per token** | **~1.35** | **0** | M=8: 740.5 tok/s |

### PCIe Transfer Elimination (Fixes 2-4)

| Fix | Transfers Eliminated | Per Layer | Per Model (28L) |
|-----|---------------------|-----------|-----------------|
| Fix 2: SwiGLU fusion | 3 | GPU→CPU→GPU | 84 |
| Fix 3: GELU fusion | 2 | GPU→CPU→GPU | 56 |
| Fix 4: RMSNorm+Residual | 4+ | GPU→CPU→GPU | 112+ |
| **Total eliminated** | | | **252+** |

---

## 3. Memory Bandwidth Analysis

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

## 4. KV Cache Performance

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

## 5. Batch Scheduling Profiling

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

## 6. Warp Count Sweep

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

## 7. Batch Size Throughput Scaling

| Batch Size (M) | Throughput (tok/s) | Ollama Ratio | Bound |
|----------------|-------------------|--------------|-------|
| 1 | ~180 | 0.75x | Memory |
| 8 | 740.5 | 2.54x | Memory |
| 16 | 583.6 | 2.01x | Transitional |
| 64 | TBD | TBD | Compute |

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
