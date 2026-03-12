# Component: GGUF Prefill (TTFT)

**Parent:** [perf-parity-spec.md](../perf-parity-spec.md)
**Status:** Active — 1.33x gap on yoga (1900MHz), 1.4x on Jetson (MAXN_SUPER)
**Test target:** ssh yoga, forjar setup/teardown, one runtime at a time

---

## Goal

Reduce TTFT (time to first token) gap with llama.cpp to < 2x. **ACHIEVED on both platforms.**

- Yoga (1900MHz): 1.33x (13.4ms vs 10.1ms) — FP8 prefill via cuBLASLt
- Jetson (MAXN_SUPER): 1.4x (47.8ms vs 34.0ms) — HGEMM on-demand FP16

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

### Prefill Path Comparison (Jetson Orin, c=1, MAXN_SUPER 1020MHz)

| Path | Prefill tok/s | TTFT (ms) | Notes |
|------|--------------|-----------|-------|
| HGEMM (FP16 on-demand cache) | **481** | **47.8** | Default (warmup OOMs on 8GB) |
| HGEMM_PREFILL=0 (DP4A prefill) | ~150 | ~281 | Fallback |
| llama.cpp (fused Q4K GEMM) | **676** | **34.0** | Reads Q4K directly |
| **Gap** | **0.71x** | **1.4x** | |

---

## Gap Analysis

### Current gap: 1.33x on yoga, 1.4x on Jetson

**Yoga (1900MHz, FP8 prefill):**
- FP8 = 1 byte/elem vs Q4K = 0.5625 bytes/elem = 1.78x more bandwidth
- Remaining 0.75x from absmax+convert pipeline overhead
- Total: 1.78 × 0.75 ≈ 1.33x

**Jetson (MAXN_SUPER, HGEMM FP16 on-demand):**
- FP16 = 2 bytes/elem vs Q4K = 0.5625 = 3.56x more bandwidth
- But much better utilization than before (47.8ms vs 228ms old)
- On-demand caching works despite warmup OOM

### Fused Q4K GEMM — tested, architecturally limited on sm_89

llama.cpp's approach: `load_tiles_q4_K()` -> cooperative Q4K decode in shared memory
-> `vec_dot_q4_K_q8_1_dp4a()`. Reads Q4K directly without dequant to FP16.

Multiple approaches tested (PMAT-064/065/045/066): all slower than cuBLAS on tensor-core
GPUs because scalar dequant cannot use tensor cores. FP8 via cuBLASLt is the winning
approach for sm_89+. Fused Q4K GEMM is viable only on Jetson (BW-constrained, 67 GB/s).

---

## Yoga Baselines (2026-03-12, c=1, 60s, streaming, locked 1900MHz)

| Runtime | Prefill tok/s | TTFT P50 (ms) | Gap vs llama.cpp |
|---------|--------------|---------------|-----------------|
| llama.cpp | 2,280 | **10.1** | baseline |
| realizr | 1,718 | 13.4 | **1.33x (PASS < 2x)** |
| vLLM | 2,016 | 11.4 | 1.13x |
| ollama | 326 | 70.5 | 6.98x |

TTFT gap narrowed from 9.4x (Mar 8, 1500MHz, HGEMM FP16) to 1.33x
(Mar 12, 1900MHz, FP8 prefill + cuBLASLt caching + non-blocking drain).

---

## Pass Criteria

- realizr TTFT P50 <= 2x llama.cpp TTFT P50
- No OOM with HGEMM enabled (FP16 cache must fit alongside KV caches)
- HGEMM_PREFILL=0 fallback works correctly when VRAM is tight
