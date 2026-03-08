# Component: GGUF Prefill (TTFT)

**Parent:** [perf-parity-spec.md](../perf-parity-spec.md)
**Status:** Active — 9.4x gap on yoga, 5.6x on Jetson
**Test target:** ssh yoga, forjar setup/teardown, one runtime at a time

---

## Goal

Reduce TTFT (time to first token) gap with llama.cpp from 5.6x to < 2x.

Prefill processes the entire prompt in one pass. Unlike decode (memory-bound
GEMV for M=1), prefill is a GEMM (M=prompt_len) which is compute-bound
and benefits from tensor cores.

---

## Current Approach: Cached FP16 + HGEMM (PMAT-031)

```
Model load: Q4K weights -> dequant FP32 -> convert FP16 -> cache on GPU
Prefill:    FP16 cached weights x FP16 activations -> cuBLAS HGEMM (tensor cores)
```

- Eliminates per-request dequant cost
- Uses tensor cores (FP16 multiply, FP32 accumulate)
- Requires ~2.5GB FP16 weight cache (fits in 8GB VRAM for 1.5B model)

### Prefill Path Comparison (Jetson Orin, c=1)

| Path | Prefill tok/s | TTFT (ms) | Notes |
|------|--------------|-----------|-------|
| HGEMM (FP16 cached + tensor cores) | 447 | 228 | Default |
| SGEMM (per-request FP32 dequant) | 152 | 671 | HGEMM_PREFILL=0 |
| Batched GEMV (no cuBLAS) | 30 | 3399 | CUBLAS_PREFILL=0 |
| llama.cpp (fused Q4K GEMM) | 2489 | 41 | Reads Q4K directly |

---

## Gap Analysis

### Why 5.6x slower than llama.cpp?

Two factors multiply:

1. **Data size**: FP16 = 2 bytes/elem vs Q4K = 0.5625 bytes/elem = 3.56x more bandwidth
2. **BW utilization**: 20% (HGEMM) vs 32% (llama.cpp) = 1.57x

3.56 x 1.57 = 5.6x

### Why not fused Q4K GEMM?

llama.cpp's approach: `load_tiles_q4_K()` -> cooperative Q4K decode in shared memory
-> `vec_dot_q4_K_q8_1_dp4a()`. Reads Q4K directly without dequant to FP16.

Implementing this requires:
- Tiled GEMM in hand-written PTX with shared memory management
- Cooperative Q4K super-block decoding across thread block
- Q8_1 activation quantization for DP4A dot product
- Multi-week implementation effort

This is the single biggest remaining optimization opportunity.

---

## Yoga Baselines (2026-03-08, c=1, 60s, streaming, locked 1500MHz)

| Runtime | Prefill tok/s | TTFT P50 (ms) | Gap vs llama.cpp |
|---------|--------------|---------------|-----------------|
| realizr | 891.7 | 114.4 | 9.4x slower |
| llama.cpp | 8399.9 | 12.1 | baseline |
| ollama | 1427.1 | 71.5 | 5.9x slower |

Yoga gap (9.4x) is wider than Jetson (5.6x) — higher bandwidth amplifies
the FP16 vs Q4K data size penalty.

---

## Pass Criteria

- realizr TTFT P50 <= 2x llama.cpp TTFT P50
- No OOM with HGEMM enabled (FP16 cache must fit alongside KV caches)
- HGEMM_PREFILL=0 fallback works correctly when VRAM is tight
