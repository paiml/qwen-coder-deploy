# GPU Decoder Throughput Performance Specification

**Document ID:** REALIZAR-GPU-PERF-001
**Version:** 2.3.0
**Status:** ACTIVE
**Date:** 2026-03-04
**Methodology:** Toyota Way (14 Principles) + Popperian Falsification + Peer-Reviewed Citations
**Target:** >=2x Ollama parity on Jetson Orin for decoder-only transformer inference
**Supersedes:** SPEC-QWEN-PERF-001, REALIZAR-QWEN-PERF-001, Decoder Throughput Spec v1.3.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Scope](#2-architecture-scope)
3. [Performance Baseline](#3-performance-baseline)
4. [Completed Fixes](#4-completed-fixes-production)
5. [Root Cause Analysis](#5-root-cause-analysis)
6. [Optimization Roadmap](#6-optimization-roadmap)
7. [Kernel Specs Summary](#7-kernel-specs-summary)
8. [Benchmarking Methodology](#8-benchmarking-methodology)
9. [Profiling Data](#9-profiling-data)
10. [Falsification Tests](#10-falsification-tests)
11. [PMAT Compliance](#11-pmat-compliance)
12. [External Contracts](#12-external-contracts)
13. [Academic References](#13-academic-references)
14. [Revision History](#14-revision-history)

---

## 1. Executive Summary

This specification consolidates all GPU decoder throughput optimization work for the realizar inference engine. It covers autoregressive decode for LLaMA, Mistral, Phi, and Qwen model families — approximately 80-85% of HuggingFace inference workloads.

**Scope:**
- M=1 GEMV kernel optimization for decode phase
- GPU↔CPU transfer elimination in forward pass
- Async runtime integration for serving
- Quantized attention and KV cache optimization

**Key Result (Internal):** From 0.9 tok/s (GPU) to 740.5 tok/s at M=8 — a **823x improvement** in internal microbenchmarks.

**Competition Reality (Mar 4, 2026 — 4090):** Under standardized load testing (c=4, 60s), realizar achieves **151.4 tok/s** (GGUF) vs llama.cpp **931.5 tok/s** and ollama **561.6 tok/s** — a **3.7x gap** to Ollama. Decode bottleneck at ~40 tok/s (c=4), ~270 tok/s raw per-token (DECODE_TIMING). Root cause: Q6K GEMV kernel uses 1-warp (32 threads) vs Q4K's MWV 4-warp — Q6K is 4x slower per call, consuming 31.9% of GPU time (GH #118).

**Next step:** Benchmarks moving to dedicated Jetson Orin; 4090 freed for full-time QLoRA training.

**Methodology:**
- Toyota Way: Jidoka (stop-on-error), Kaizen (iterative improvement), Genchi Genbutsu (direct measurement)
- Popperian Falsification: Every claim has defined falsification conditions
- Peer-reviewed citations: 35+ references from ICLR, ICML, NeurIPS, SOSP, PPoPP

---

## 2. Architecture Scope

### Supported Model Families

| Family | Models | Key Characteristics |
|--------|--------|---------------------|
| LLaMA | 2-7B, 2-13B, 3-8B, 3-70B | GQA, SwiGLU, RoPE |
| Mistral | 7B, Nemo, Mixtral-8x7B | Sliding window attention |
| Phi | 2, 3-mini, 3-medium | LayerNorm + GELU, partial attention |
| Qwen | 7B, 14B, Qwen2-7B, 2-72B | Aggressive GQA (6:1-8:1), large RoPE theta (1M) |

### Decode Path

```
Token → Embedding → [RMSNorm → Attention → Residual → RMSNorm → FFN → Residual] × L → LM Head → Logits
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                     All GPU-resident after Fixes 1-6 (zero PCIe transfers)
```

### Scope Boundaries

**IN:** M=1 GEMV, memory coalescing, GPU transfer elimination, async serving, quantized KV cache
**OUT:** Prefill-phase GEMM [Patel24], multi-GPU distribution, training

### Deployment Topology

```
4090 Host (noah-Lambda-Vector)           Jetson Orin (jetson)
├── QLoRA training (full-time)           ├── apr-gguf     :8081  (CUDA)
├── Deep profiling (occasional):         ├── ollama       :8082  (CUDA)
│   nsys-gpu, ncu-gpu, profile-gpu       ├── llama.cpp    :8083  (CUDA)
│   apr profile, apr bench               ├── apr-apr      :8084  (CUDA)
└── Builds: apr, llama.cpp, trueno       ├── apr-safetens :8085  (CUDA)
                                         └── probador load tests (continuous)
```

For architecture details (Qwen2 parameters, GQA ratios), see [baselines.md](./components/baselines.md#2-model-reference-qwen2qwen25-architecture).

---

## 3. Performance Baseline

### Internal Microbenchmarks (Feb 2026)

| Metric | Before (Dec 2025) | After (Feb 2026) | Improvement |
|--------|-------------------|-------------------|-------------|
| GGUF CPU throughput | 3.0 tok/s | 12.5-17.3 tok/s | 4-6x |
| APR GPU throughput (M=8) | 0.9 tok/s | 740.5 tok/s | 823x |
| Ollama ratio (internal) | 0.004x | 2.54x (M=8) | 635x |
| PCIe transfers per token | 252+ | 0 | ∞ |

### Competition Benchmarks (Mar 2026)

Standardized load test: `probador llm load` (60s, c=4). Model: Qwen2.5-Coder-1.5B Q4_K_M.

**RTX 4090 (Mar 4 2026 — final 4090 baselines before Jetson migration):**

| Runtime | Tokens/s | Decode tok/s | Latency P50 (ms) |
|---------|----------|-------------|-------------------|
| llama.cpp | **931.5** | **233.5** | 548 |
| ollama | **561.6** | **139.9** | 915 |
| realizar (GGUF) | 151.4 | 40.8 | 2,530 |

**Gap to parity (4090):** realizar decode (40.8 tok/s) is **3.4x slower** than ollama decode (139.9 tok/s) and **5.7x slower** than llama.cpp (233.5 tok/s) at c=4. Raw per-token decode is ~270 tok/s (DECODE_TIMING), suggesting concurrency lock contention and prefill overhead dominate under load.

**Gap to parity (Jetson Orin, isolated):** realizr (7.8 tok/s) is **4.1x slower** than llama.cpp (31.9 tok/s) and **3.0x slower** than ollama (23.4 tok/s) at c=1. ITL 128ms vs 31ms (4x gap). Zero throughput scaling at c=4 (RwLock contention). Batch mode OOM on 7.4 GB unified memory.

**Jetson Orin (Mar 4 2026 — serial isolated benchmarks, one runtime at a time):**

| Runtime | c=1 tok/s | c=1 decode | c=1 ITL (ms) | c=4 tok/s | c=4 decode | c=4 P50 (ms) |
|---------|-----------|------------|-------------|-----------|------------|------------|
| llama.cpp | **31.9** | **31.9** | **31.4** | **66.2** | **16.5** | 1,934 |
| ollama | 23.4 | 23.3 | 42.8 | 32.6 | 8.2 | 3,907 |
| realizr (GGUF, GPU) | 7.8 | 7.8 | 128.3 | 7.8 | 1.9 | 16,428 |

**Methodology:** Serial isolated benchmarks — each runtime tested alone with all others stopped (`forjar-jetson-{realizr,ollama,llamacpp}.yaml`). Critical for Jetson's 7.4 GB unified memory where concurrent servers cause memory contention (ollama jumped from 11.3 → 23.4 tok/s when isolated, 2.1x improvement).

**Key findings:**
- realizr GPU (7.8 tok/s) is **4.1x slower** than llama.cpp (31.9) and **3x slower** than ollama (23.4) at c=1
- **ITL gap is 4x:** realizr 128ms/token vs llama.cpp 31ms/token — decode kernel is the bottleneck
- Concurrency scaling: llama.cpp 2.1x, ollama 1.4x, realizr 0x (flat, RwLock contention)
- Native CUDA build on Jetson (45 min, no cross-compile) was 7.8x faster than CPU-only (1.0 tok/s)
- Batch mode (`--batch`) OOM-killed on 7.4 GB unified memory

### Hardware Reference

| Host | GPU | Memory BW | VRAM | Role |
|------|-----|-----------|------|------|
| noah-Lambda-Vector (4090) | RTX 4090 | 1,008 GB/s | 24 GB GDDR6X | QLoRA training (full-time) + deep profiling (nsys/ncu, occasional) |
| jetson | Orin (nvgpu) | 102 GB/s | 7.4 GB unified | Continuous load testing + CI benchmarks (dedicated) |

**Architecture split (v2.3.0):** Load testing moves permanently to Jetson Orin, freeing the 4090 for full-time QLoRA fine-tuning. The 4090 is only used for inference during occasional deep GPU profiling (nsys/ncu). All `probador llm load` benchmarks target Jetson-hosted services.

For complete baseline tables, threshold registry, and measurement protocol, see [baselines.md](./components/baselines.md).

---

## 4. Completed Fixes (Production)

### Fix 1: CPU Q4K Routing Resolution

**Problem:** Q4K dequantization was routing to CPU even when GPU was available.
**Fix:** Corrected backend dispatch to use GPU path for Q4K operations.
**Impact:** CPU throughput 3.0 → 12.5-17.3 tok/s.

### Fix 2: SwiGLU GPU Fusion (QWEN-003)

**Problem:** SwiGLU activation in FFN performed 3 PCIe round-trips per layer.
**Fix:** Wired `fused_swiglu_gpu` kernel (PAR-023) into `gpu/adapters/apr_q4.rs`.
**Impact:** Eliminates 84 PCIe transfers for 28-layer models.
**Citation:** [Shazeer20] GLU Variants Improve Transformer.

### Fix 3: Async spawn_blocking (Runtime)

**Problem:** Synchronous GPU inference blocked tokio executor threads.
**Fix:** `tokio::task::spawn_blocking` for GPU inference isolation.
**Impact:** Enables concurrent request handling without runtime starvation.

### Fix 4: GPU RMSNorm + Residual (QWEN-013) + GELU Fusion (QWEN-011)

**Problem:** RMSNorm, residual connections, and GELU activation used CPU round-trips.
**Fix:** GPU-resident `rmsnorm_gpu_ptr()`, `residual_add_gpu()`, in-place `gelu_gpu()`.
**Impact:** M=8: 740.5 tok/s (2.54x Ollama), M=16: 583.6 tok/s (2.01x Ollama).

### Fix 5: Queue-Based Dispatch

**Problem:** No backpressure mechanism for concurrent GPU requests.
**Fix:** Bounded MPSC channel (32 slots) with FIFO dispatch.
**Impact:** Predictable latency under load, graceful degradation.

### Fix 6: Continuous Batching

**Problem:** Single-request inference underutilizes GPU compute.
**Fix:** Configurable batch intervals with dynamic request grouping.
**Impact:** Higher throughput at batch sizes > 1.

For kernel implementation details and code samples, see [kernel-specifications.md](./components/kernel-specifications.md).

---

## 5. Root Cause Analysis

### Primary Root Cause

> The GEMV kernel's thread-to-data mapping caused non-coalesced global memory reads, reducing effective memory bandwidth by 68x.

### 5 Whys Summary

1. Decode throughput 190x slower → 192 GEMVs at 4.41ms each
2. GEMV slow → 1.4% bandwidth utilization
3. Low bandwidth → strided access defeating coalescing [McKee24]
4. Strided access → column-per-warp thread assignment
5. Column-per-warp → initial implementation prioritized simplicity

### Secondary Root Causes

- **GPU↔CPU transfers:** 252+ PCIe round-trips per token for activation functions (fixed)
- **Async blocking:** Sync GPU inference starved tokio executor (fixed)
- **APR format corruption:** Force-validated tensor mappings caused inefficient broadcasting (fixed)

### New Root Causes (Mar 2026)

- **Q6K GEMV 1-warp bottleneck (GH #118):** Q6K GEMV uses 32-thread kernel (1 warp, 33% occupancy) while Q4K uses MWV 4-warp (128 threads). Q6K is 4x slower per call (39.7µs vs 9.9µs), consuming 31.9% of GPU time despite only 29 tensors. Affects: `output.weight`, `ffn_down.weight` (28L), `attn_v.weight` (28L).
- **Kernel launch overhead:** 52.5% of decode time from ~180 kernel launches/token (PMAT-015/017)
- **Concurrency lock contention:** RwLock serialization at c=4 drops decode from ~270 tok/s (raw) to 40.8 tok/s (probador). Single-request path is 153 tok/s due to prefill + HTTP overhead.

### Jetson Orin Root Cause Analysis (Mar 5, 2026)

**Five-Whys: realizr 104ms/token vs llama.cpp 31ms/token on Orin**

1. Why 3.4x slower? → GEMV kernels consume 94% of GPU time (nsys profiling)
2. Why GEMV so slow? → MWV Q4K avg 342µs on Orin vs 10µs on 4090 (34x, not 10x expected from BW ratio)
3. Why 34x not 10x? → 3.4x kernel efficiency gap — MWV tuned for 128-SM 4090, suboptimal on 8-SM Orin
4. Why suboptimal? → L2 cache (2 MB vs 72 MB), warp count (3 warps default), PTX sm_70 JIT'd to sm_87
5. Why not tuned for Orin? → Historical focus on 4090; Orin has 16x fewer cores and 10x less bandwidth

**nsys Kernel Profiling (Ground Truth — NOT brick profiler):**

| Kernel | Time % | Instances | Avg (µs) | Phase |
|--------|--------|-----------|----------|-------|
| batched_q4k_gemv_warp_reduce | 42% | 4,032 | 488 | Prefill |
| mwv_q4k_gemv | 19% | 2,688 | 342 | Decode |
| q6k_gemv_warp_reduce | 17% | 1,808 | 442 | Decode |
| batched_q6k_gemv_warp_reduce | 16% | 336 | 2,297 | Prefill |
| multi_warp_attention | 0.3% | 2,688 | 6 | Decode |

**Decode per-layer: ~3 Q4K (342µs) + ~2 Q6K (442µs) ≈ 1.9ms/layer × 28 = 53ms estimated.**

**CORRECTION:** Earlier brick profiling (`apr profile --granular`) claimed AttentionScore was 92% of time. nsys shows this was an artifact of CPU-side synchronization overhead. Actual attention is 0.3% of GPU time.

**Warp Sweep Results (Mar 5, 2026 — Jetson Orin):**

| MWV_WARPS | tok/s | Avg (ms) | Notes |
|-----------|-------|----------|-------|
| 1 | 8.9 | 4080 | Too few warps |
| 2 | 9.9 | 3729 | Tied best (FP16) |
| 3 (default) | 9.9 | 3707 | Tied best (FP16) |
| 4 | 9.7 | 3786 | Slight regression |
| 6 | 9.3 | 3946 | Occupancy contention |
| 8 | 8.8 | 4152 | Worst — too many warps for 8-SM Orin |

| Variant | tok/s | Avg (ms) | Notes |
|---------|-------|----------|-------|
| Default MWV | 9.9 | 3707 | FP16 baseline |
| WIDE_Q4K | 6.8 | 5247 | 8 warps too wide |
| VECTORIZED_Q4K | 9.3 | 3942 | 1 warp, u32 loads |
| **DP4A_Q4K** | **11.2** | **3306** | **INT8 dot-product — best** |
| WIDE_Q4K_DISABLE | 6.5 | 5495 | Legacy tiled — slowest |

**Winner: DP4A_Q4K (+13% over default MWV).** Orin sm_87 has native DP4A INT8 acceleration.

**nsys with DP4A (Mar 5, 2026):**

| Kernel | Time % | Instances | Avg (µs) | Phase |
|--------|--------|-----------|----------|-------|
| batched_q4k_gemv_warp_reduce | 44% | 4,032 | 488 | Prefill |
| q6k_gemv_warp_reduce | 18% | 1,808 | 442 | Decode |
| batched_q6k_gemv_warp_reduce | 17% | 336 | 2,302 | Prefill |
| mwv_dp4a_q4k_gemv | 16% | 2,688 | **268** | Decode |
| multi_warp_attention | 0.4% | 2,688 | 6 | Decode |

**DP4A reduced Q4K decode GEMV from 342µs to 268µs (22% faster).** Q6K at 442µs is now 1.65x slower than Q4K DP4A — Q6K is the new primary decode bottleneck.

**Confirmed facts:**
- Weights ARE on GPU (0 MB preload is reporting bug — weights uploaded in constructor, second call sees them cached)
- CUDA graph IS active (graph replay path confirmed via GRAPH-TIMING)
- GPU utilization 99% during decode (tegrastats GR3D_FREQ)
- `trace: false` hardcoded in API handlers (fixed: now respects X-Trace-Level)
- GEMV is 94% of GPU time, attention 0.3% (nsys on Orin, Mar 5 2026)
- DP4A Q4K is optimal kernel variant for Orin (11.2 vs 9.9 tok/s, +13%)
- Default MWV 2-3 warps is optimal for Orin (vs 4090 default of 3-4 warps)
- Q6K decode GEMV (442µs) is now the dominant decode bottleneck with DP4A enabled

For full analysis including the "impossible observation" (CPU outperforming GPU), see [root-cause-analysis.md](./components/root-cause-analysis.md).

---

## 6. Optimization Roadmap

### Tier Summary

| Tier | Speedup Range | Items | Status |
|------|---------------|-------|--------|
| T0: Completed | Shipped | 6 fixes | ✅ Production |
| T0a: Regressions | P0 | APR native GPU fix | ❌ Broken |
| T1: Critical | 2-5x | SageAttention, EAGLE, CUDA graphs | Planned |
| T2: High Impact | 1.5-2x | Marlin, DCA, KV quant, MInference | Mixed |
| T3: Incremental | 1.1-1.5x | 3-way fusion, tile tuning | ✅ Mostly done |

### Priority Matrix

| ID | PMAT | Optimization | Speedup | Status |
|----|------|--------------|---------|--------|
| GH #118 | PMAT-019 | **Q6K MWV GEMV kernel** | **2-4x Q6K** | **Planned (31.9% GPU time)** |
| QWEN-015 | PMAT-018 | APR native GPU fix | N/A | ✅ Fixed |
| QWEN-014 | PMAT-017 | **Kernel launch overhead** | **2-5x** | **Planned (52.5% overhead)** |
| QWEN-003 | PMAT-002 | SwiGLU GPU fusion | 1.5-2x | ✅ DONE |
| QWEN-011 | PMAT-003 | GELU GPU fusion | 1.2x | ✅ DONE |
| QWEN-013 | PMAT-004 | GPU RMSNorm+Residual | 1.3x | ✅ DONE |
| QWEN-007 | PMAT-005 | KV cache quantization | 4x memory | ✅ DONE |
| QWEN-010 | PMAT-007 | RTX 4090 tile tuning | 1.1x | ✅ DONE |
| QWEN-009 | PMAT-006 | 3-way kernel fusion | 1.2x | ✅ Kernel done |
| QWEN-001 | PMAT-008 | SageAttention INT8 | 2-3x | Planned |
| QWEN-004 | PMAT-009 | EAGLE speculative | 2-3x | Planned |
| QWEN-005 | PMAT-010 | Marlin-style GPTQ | 2.6x | Planned |
| QWEN-006 | PMAT-011 | DCA long context | N/A | Planned |
| QWEN-008 | PMAT-012 | MInference sparse | 3-6x prefill | Planned |

For full tier descriptions with acceptance criteria and citations, see [optimization-tiers.md](./components/optimization-tiers.md).

---

## 7. Kernel Specs Summary

### Production Kernels

| Kernel | Purpose | Transfers Eliminated |
|--------|---------|---------------------|
| `fused_swiglu_gpu` | SwiGLU activation (FFN) | 3/layer |
| `gelu_gpu` | GELU activation (standard FFN) | 2/layer |
| `rmsnorm_gpu_ptr` | RMSNorm with cached gamma | 2/layer |
| `residual_add_gpu` | Residual connection | 1/layer |
| `Q8Dequant` | KV cache dequantization | N/A (memory) |

### Planned Kernels

| Kernel | Purpose | Expected Speedup |
|--------|---------|-----------------|
| `CoalescedGemvKernel` | M=1 GEMV with coalesced access | 68x bandwidth |
| `FusedRmsNormGateUpSwigluQ4K` | 3-way FFN fusion | 1.2x per FFN |
| SageAttention INT8 | Quantized Q@K^T | 2.1x vs FA2 |

For implementation details, PTX generation, and memory savings analysis, see [kernel-specifications.md](./components/kernel-specifications.md).

---

## 8. Benchmarking Methodology

### Protocol (Per Hoefler & Belli SC'15)

1. **CV-based stopping:** Auto-stop at CV < 0.05
2. **Warmup discard:** Separate warmup from measurement
3. **Outlier detection:** MAD-based (k=1.4826)
4. **Percentiles:** p50, p95, p99 latencies
5. **Environment metadata:** Full reproducibility

### Infrastructure

**Load Testing (Jetson Orin — dedicated):**
- All `probador llm load` benchmarks run on Jetson Orin (aarch64, CUDA 12.6, 7.4 GB unified)
- **Serial (isolated) mode required:** Jetson's 7.4 GB unified memory is shared between CPU and GPU. Running multiple servers simultaneously causes memory contention and invalidates results.
- Per-runtime forjar configs: `forjar-jetson-realizr.yaml`, `forjar-jetson-ollama.yaml`, `forjar-jetson-llamacpp.yaml` — each stops ALL other servers before starting the target runtime
- `make bench-jetson-serial` runs all 3 runtimes in isolation (c=1 + c=4 per runtime)
- `make bench-jetson-realizr` / `bench-jetson-ollama` / `bench-jetson-llamacpp` for individual runtimes
- Teardown between tests: `forjar-jetson-teardown.yaml` stops all inference processes
- Parallel deploy (`forjar-jetson.yaml`) available for smoke tests but NOT for benchmarking

**Deep Profiling (4090 — occasional):**
- nsys/ncu kernel profiling requires 4090 SM count and PCIe topology
- `make nsys-gpu / ncu-gpu / profile-gpu` targets remain 4090-only
- Run only when diagnosing kernel-level bottlenecks

Implemented in PARITY-007 through PARITY-010:
- `CVStoppingBenchmark`, `WarmupBenchmark`, `EnvironmentMetadata`
- `VersionedBenchmarkResult` with schema versioning
- `detect_outliers()` with MAD scale factor

### External Contracts

Authoritative benchmark methodology and competition baselines are maintained externally:
- **Methodology:** [benchmarking-v2.md](./benchmarking-v2.md)
- **Baselines:** [inference-showdown-v1.yaml](./inference-showdown-v1.yaml)

For detailed baseline tables and threshold registry, see [baselines.md](./components/baselines.md#7-benchmarking-standards).

---

## 9. Profiling Data

### Nsight Systems Kernel Profile (2026-03-04)

| Kernel | Time (%) | Instances | Avg (µs) |
|--------|----------|-----------|----------|
| `mwv_q4k_gemv` | **46.0%** | 53,592 | 9.9 |
| `q6k_gemv_warp_reduce` | **31.9%** | 9,251 | 39.7 |
| `multi_warp_attention_indirect` | 9.3% | 8,932 | 12.0 |
| `rmsnorm_vectorized` | 5.3% | 18,183 | 3.3 |
| `residual_add` | 3.8% | 44,660 | 1.0 |
| `fused_swiglu` | 0.7% | 8,932 | 0.9 |

**Dominant bottleneck:** GEMV kernels consume **77.9%** of GPU time.

### Host-Side Profiling (2026-03-02)

| Metric | Value |
|--------|-------|
| Kernel launch overhead | **52.5%** of decode time (128,484µs) |
| Memory efficiency | **8.4%** (84.3 / 1,008 GB/s) |
| Decode throughput (M=1) | 130.7 tok/s |
| Performance grade | **D** |
| Roofline classification | **MEMORY BOUND** (4.0 FLOP/byte, threshold 82.0) |

### Key Numbers

| Metric | Value | Source |
|--------|-------|--------|
| PCIe transfers eliminated | 252+/token | Fixes 2-4 |
| ContiguousKV speedup | 16,640x | PARITY-005 |
| Q8 KV memory reduction | 3.56x | QWEN-007 |
| Optimal warp config (4090) | 256 threads (8 warps) | Warp sweep |
| Optimal warp config (Orin) | 96 threads (3 warps) | Warp sweep Mar 5 |
| Optimal kernel (Orin) | DP4A Q4K (+13%) | Variant sweep Mar 5 |
| Q4K decode GEMV (Orin, DP4A) | 268µs | nsys Mar 5 |
| Q6K decode GEMV (Orin) | 442µs (1.65x slower) | nsys Mar 5 |

### Roofline Position

```
GEMV (M=1):  ~2 FLOP/byte → MEMORY BOUND
GEMM (M>64): ~128 FLOP/byte → COMPUTE BOUND
```

For full profiling tables, PCIe transfer analysis, warp count sweep, and batch scaling data, see [profiling-data.md](./components/profiling-data.md).

External profiling appendix: `batuta/book/src/appendix/benchmarks.md`.

---

## 10. Falsification Tests

### Hypothesis Summary

| ID | Claim | Prediction | Status |
|----|-------|------------|--------|
| H1 | Coalesced access → >90% BW | gld_efficiency > 0.90 | Pending |
| H2 | Coalesced GEMV → <0.05ms | mean_latency < 0.05ms | Pending |
| H3 | End-to-end >200 tok/s | throughput > 200 tok/s | ✅ EXCEEDED (740.5) |
| H4 | float4 loads → 2x bandwidth | vectorized/scalar > 2.0 | Pending |
| H5 | Occupancy >50% ≈ diminishing | ratio(1024/256) < 1.2 | Pending |
| H-APR1 | Fix mapping → >50 tok/s | After fix: >50 | ✅ EXCEEDED (740.5) |
| H-APR3 | GQA fix → linear speedup | >50% improvement | FALSIFIED (already correct) |

### Verification Matrix

| Section | Tests | Passing |
|---------|-------|---------|
| A: GQA Fix | 3 | ✅ 3/3 |
| B: SwiGLU Fusion | 3 | ✅ 3/3 |
| C: Attention Quant | 3 | Pending |
| D: Launch Overhead | 3 | Pending |
| E: APR GPU Regression | 3 | Pending |

For full hypothesis definitions, F-tests, pre-flight controls, and QA checklist, see [falsification-tests.md](./components/falsification-tests.md).

---

## 11. PMAT Compliance

### Quality Gate Thresholds

| Metric | Threshold | Command |
|--------|-----------|---------|
| TDG Score | >= 93.0 (A) | `pmat analyze tdg` |
| Cognitive Complexity | <= 25 | `pmat analyze complexity` |
| SATD | 0 critical | `pmat analyze satd` |
| Test Coverage | >= 80% | `make coverage` |
| Clippy Warnings | 0 | `make lint` |

### Roadmap

**Source of truth:** [roadmap.yaml](../docs/roadmaps/roadmap.yaml) (`pmat work list`)

### Work Tickets Summary

| PMAT ID | QWEN Ticket | Title | Status |
|---------|-------------|-------|--------|
| PMAT-001 | QWEN-002 | GQA Broadcasting Fix | ✅ Completed |
| PMAT-002 | QWEN-003 | SwiGLU GPU Fusion | ✅ Completed |
| PMAT-003 | QWEN-011 | GELU GPU Fusion | ✅ Completed |
| PMAT-004 | QWEN-013 | GPU RMSNorm+Residual | ✅ Completed |
| PMAT-005 | QWEN-007 | KV Cache Quantization | ✅ Completed |
| PMAT-006 | QWEN-009 | 3-Way FFN Fusion | ✅ Completed |
| PMAT-007 | QWEN-010 | RTX 4090 Tile Tuning | ✅ Completed |
| PMAT-008 | QWEN-001 | SageAttention INT8 | Planned |
| PMAT-009 | QWEN-004 | EAGLE Speculative | Planned |
| PMAT-010 | QWEN-005 | Marlin-Style Kernel | Planned |
| PMAT-011 | QWEN-006 | DCA Long Context | Planned |
| PMAT-012 | QWEN-008 | MInference Sparse | Planned |
| PMAT-013 | — | Nsight Profiling Integration | ✅ Completed |
| PMAT-014 | — | Competition Baseline Update | ✅ Completed |
| PMAT-015 | — | Kernel Launch Overhead RCA | ✅ Completed |
| PMAT-016 | — | APR Native GPU Regression RCA | ✅ Completed |
| PMAT-017 | QWEN-014 | CUDA Graphs / Fusion (launch overhead) | Planned |
| PMAT-018 | QWEN-015 | APR Native GPU Fix | ✅ Fixed (--skip-contract) |
| PMAT-019 | GH #118 | **Q6K MWV GEMV kernel** (31.9% GPU time) | **Planned — 2-4x Q6K speedup** |
| PMAT-020 | — | Jetson Orin load test migration | **In Progress** |
| PMAT-021 | GH #121 | **DP4A Q4K default on Orin sm_87** | **Validated (+13%)** |
| PMAT-022 | GH #118 | **Q6K MWV GEMV kernel (442µs → target ~200µs)** | **Next — decode bottleneck** |

For full ticket YAML definitions and pre-commit protocol, see [pmat-work-tickets.md](./components/pmat-work-tickets.md).

---

## 12. External Contracts

The following external documents are authoritative for their respective domains and are not duplicated here:

| Document | Location | Purpose |
|----------|----------|---------|
| PMAT Roadmap | [roadmap.yaml](../docs/roadmaps/roadmap.yaml) | Work ticket tracking |
| Benchmarking Methodology v2 | [benchmarking-v2.md](./benchmarking-v2.md) | Benchmark protocol |
| Inference Showdown v1 | [inference-showdown-v1.yaml](./inference-showdown-v1.yaml) | Competition baselines |
| Performance Snapshots | [performance.md](../performance.md) | Measured throughput tables |
| Profiling Appendix | `batuta/book/src/appendix/benchmarks.md` | GPU decode profiling data |

---

## 13. Academic References

### Architecture & Models

1. [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) — Bai et al., 2024
2. [Qwen2.5-1M Technical Report](https://qwenlm.github.io/blog/qwen2.5-1m/) — Alibaba, 2025
3. [LLaMA](https://arxiv.org/abs/2302.13971) — Touvron et al., 2023
4. [Mistral 7B](https://arxiv.org/abs/2310.06825) — Jiang et al., 2023
5. [Phi-3 Technical Report](https://arxiv.org/abs/2404.14219) — Abdin et al., 2024
6. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al., 2017

### Attention & Memory

7. [FlashAttention (NeurIPS 2022)](https://arxiv.org/abs/2205.14135) — Dao et al.
8. [PagedAttention (SOSP 2023)](https://arxiv.org/abs/2309.06180) — Kwon et al.
9. [GQA](https://arxiv.org/abs/2305.13245) — Ainslie et al., 2023
10. [SageAttention (ICLR 2025)](https://arxiv.org/abs/2410.02367)
11. [SageAttention2 (ICML 2025)](https://arxiv.org/abs/2411.10958)
12. [SageAttention3 (NeurIPS 2025)](https://arxiv.org/abs/2505.11594)
13. [KIVI KV Quantization](https://arxiv.org/abs/2402.02750) — Liu et al., 2024
14. [MInference (Microsoft, 2024)](https://arxiv.org/abs/2407.02490)

### Speculative Decoding

15. [EAGLE (ICML 2024)](https://arxiv.org/abs/2401.15077) — Li et al.
16. [EAGLE-2 (EMNLP 2024)](https://arxiv.org/abs/2406.16858)
17. [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/abs/2503.01840)

### GPU Performance & Kernels

18. [MARLIN (PPoPP 2025)](https://arxiv.org/abs/2408.11743) — Frantar et al.
19. [GLU Variants](https://arxiv.org/abs/2002.05202) — Shazeer, 2020
20. [Roofline Model](https://doi.org/10.1145/1498765.1498785) — Williams et al., 2009
21. [GPU Microarchitecture Benchmarking](https://arxiv.org/abs/1804.06826) — Jia et al., 2018
22. [GPU Memory Hierarchy](https://doi.org/10.1109/TPDS.2016.2549523) — Mei & Chu, 2017
23. [Better Performance at Lower Occupancy](https://www.nvidia.com/content/GTC-2010/pdfs/2238_GTC2010.pdf) — Volkov, 2010
24. [CUDA C++ Programming Guide v12.3](https://docs.nvidia.com/cuda/) — NVIDIA, 2023
25. [Auto-tuning GEMV on GPUs (PPoPP 2015)](https://doi.org/10.1145/2688500.2688513) — Li et al.
26. [KBLAS Optimized BLAS (ACM TOMS)](https://doi.org/10.1145/2818311) — Abdelfattah et al., 2016
27. [GPU Atomic Performance Modeling](https://vulkan.org/user/pages/09.events/vulkanised-2024) — McKee, 2024
28. [Modeling NVIDIA Ampere Performance (MEMSYS '23)](https://doi.org/10.1145/3631461.3631546) — Abdelkhalik et al., 2023

### Inference Systems

29. [DeepSpeed Inference (SC22)](https://arxiv.org/abs/2207.00032) — Aminabadi et al.
30. [FlexGen (ICML 2023)](https://arxiv.org/abs/2303.06865) — Sheng et al.
31. [Splitwise (ISCA '24)](https://arxiv.org/abs/2311.18677) — Patel et al.
32. [Sarathi-Serve (OSDI '24)](https://arxiv.org/abs/2403.02310) — Agrawal et al.
33. [SpecInfer (ASPLOS 2024)](https://arxiv.org/abs/2305.09781) — Miao et al.
34. [ScaleLLM (ACL 2024)](https://arxiv.org/abs/2407.00588) — Chen et al.
35. [CPU Computations for LLM Inference (Euro-Par 2024)](https://doi.org/10.1007/978-3-031-69577-3_15) — Park & Egger

### Methodology

36. [The Logic of Scientific Discovery](https://www.routledge.com/9780415278447) — Popper, 1959
37. [Scientific Benchmarking (SC15)](https://doi.org/10.1145/2807591.2807644) — Hoefler & Belli
38. [Statistically Rigorous Java Evaluation (OOPSLA 2007)](https://doi.org/10.1145/1297027.1297033) — Georges et al.
39. [The Art of Computer Systems Performance Analysis](https://www.wiley.com/en-us/9780471503361) — Jain, 1991
40. [The Toyota Way](https://www.mhprofessional.com/9780071392310-usa-the-toyota-way) — Liker, 2004

---

## 14. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 2.3.0 | 2026-03-04 | **Jetson Orin migration:** Load testing moves to dedicated Jetson Orin (aarch64, CUDA 12.6, 7.4 GB), freeing 4090 for full-time QLoRA. New forjar-jetson.yaml + Makefile targets. Q6K GEMV bottleneck identified (GH #118): 31.9% GPU time, 4x slower than Q4K MWV. Updated baselines: APR 40.8 tok/s decode (c=4) vs llama.cpp 233.5. PMAT-019 (Q6K MWV), PMAT-020 (Jetson migration) added. |
| 2.2.0 | 2026-03-04 | v3 benchmarks: gap narrowed from 6.3x to 3.4x. APR native regression fixed (PMAT-018, --skip-contract). All formats 143-167 tok/s. GitHub issues #1-#5 filed. forjar hardened (continue_independent, SafeTensors timeout). |
| 2.1.0 | 2026-03-04 | Competition baselines (v3/20260303), Nsight profiling integration, kernel launch overhead RCA (52.5%), APR native GPU regression (100% errors), PMAT-013 through PMAT-018 added. |
| 2.0.0 | 2026-03-04 | Consolidated from 3 specs (SPEC-QWEN-PERF-001, REALIZAR-QWEN-PERF-001, Decoder Throughput v1.3.0). Added component sub-specs. pmat work roadmap with 12 tickets. |
| 1.3.0 | 2025-12-29 | Decoder Throughput Spec: Popperian review, updated baselines (predecessor) |
| 1.0.0 | 2026-02-02 | SPEC-QWEN-PERF-001: Initial Qwen optimization spec (predecessor) |
| 1.4.0 | 2026-02-01 | REALIZAR-QWEN-PERF-001: Showcase throughput improvement (predecessor) |

---

**Signed:**
*Realizar Performance Engineering*
*Date: 2026-03-04*
