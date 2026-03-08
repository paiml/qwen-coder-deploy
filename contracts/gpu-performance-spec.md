# GPU Decoder Throughput Performance Specification

**Document ID:** REALIZAR-GPU-PERF-001
**Version:** 2.14.0
**Status:** ACTIVE
**Date:** 2026-03-08
**Methodology:** Toyota Way (14 Principles) + Popperian Falsification + Peer-Reviewed Citations
**Target:** >=2x Ollama parity on Jetson Orin for decoder-only transformer inference
**Supersedes:** SPEC-QWEN-PERF-001, REALIZAR-QWEN-PERF-001, Decoder Throughput Spec v1.3.0

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Scope](#2-architecture-scope)
3. [Performance Baseline](#3-performance-baseline)
4. [Completed Fixes](#4-completed-fixes-production)
5. [Root Cause Analysis](#5-root-cause-analysis) (includes: [Why Tooling Alone Doesn't Close Gaps](#why-tooling-alone-doesnt-close-performance-gaps), [First Principles: System Component Anatomy](#first-principles-system-component-anatomy))
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

**Gap to parity (Jetson Orin, isolated, non-streaming v1):** realizr (7.8 tok/s) is **4.1x slower** than llama.cpp (31.9 tok/s) and **3.0x slower** than ollama (23.4 tok/s) at c=1. Zero throughput scaling at c=4 (RwLock contention). Batch mode OOM on 7.4 GB unified memory.

**CORRECTION (Mar 5 2026 — SSE streaming metrics):** Previous non-streaming measurement blended prefill + decode into a single tokens/sec figure. With SSE streaming (probador `--stream true`), we can now separate TTFT (prefill) from ITL/decode. **The decode gap is 1.9x, not 4.1x.** The dominant bottleneck is prefill (148x slower), not decode.

**Jetson Orin (Mar 5 2026 — SSE streaming, serial prefill, c=1):**

| Runtime | Decode tok/s | ITL P50 (ms) | TTFT P50 (ms) | Prefill tok/s | E2E tok/s |
|---------|-------------|-------------|--------------|--------------|-----------|
| llama.cpp | **32.2** | **31.0** | **48.6** | **2099.8** | **32.1** |
| ollama | 33.1 | 30.2 | 428.6 | 238.0 | 30.0 |
| realizr (serial prefill) | 17.0 | 58.9 | 7192 | 14.2 | 8.7 |

**FIX (Mar 5 2026 — PMAT-023 batched prefill):** Root cause: `generate_gpu_resident_streaming` in generate_1.rs defaulted `BATCHED_PREFILL` to `false`, processing each prompt token through the full transformer stack sequentially. For a ~20-token prompt: 20 × (7 GEMV × 28 layers) = 3,920 kernel launches instead of 196. Setting `BATCHED_PREFILL=1` (now the default) gives **9x TTFT improvement**.

**Jetson Orin (Mar 5 2026 — SSE streaming, BATCHED prefill, c=1):**

| Runtime | Decode tok/s | ITL P50 (ms) | TTFT P50 (ms) | Prefill tok/s | E2E tok/s |
|---------|-------------|-------------|--------------|--------------|-----------|
| llama.cpp | **32.3** | **31.0** | **43.9** | **524.3** | **31.9** |
| realizr (batched prefill) | 17.7 | 56.6 | 815.7 | 28.2 | 12.5 |
| **Gap** | **1.8x** | **1.8x** | **18.6x** | **18.6x** | **2.6x** |

**Jetson Orin Nano Super (Mar 6 2026 — GH-176 HW DP4A, locked clocks, isolated, c=1, 60s streaming):**

| Runtime | Decode tok/s | ITL P50 (ms) | TTFT P50 (ms) | Prefill tok/s |
|---------|-------------|-------------|--------------|--------------|
| llama.cpp | **33.1** | **30.2** | **41** | **2478** |
| realizr (GH-176, HW DP4A) | 27.8 | 36.0 | 1045 | 97.8 |
| **Gap** | **1.19x** | **1.19x** | **25x** | **25x** |
| *realizr (GH-174, MWV DP4A)* | *21.4* | *46.8* | *3542* | *28.8* |

**GH-173/174/175/176 optimization history (Mar 6, locked clocks):**

| Optimization | Decode tok/s | Delta |
|---|---|---|
| Baseline (DP4A, 3 warps) | 16.7 | - |
| +GH-173 parallel byte-masked scale | 19.8 | +18.6% |
| +locked clocks (`jetson_clocks`) | 21.4 | +28.1% total |
| +GH-174 grid-stride LM head | 21.4 | no change |
| +GH-175 prefetch | 21.6 | +0.9% (noise) |
| +GH-176 `.maxnreg 255` | 21.4 | no impact (kernel uses only 34 regs) |
| **+GH-176 half-warp DP4A Q4K** | **27.8** | **+66.5% total** |

**GH-176 half-warp DP4A Q4K GEMV (trueno #175):** 16 threads per super-block (vs 32 in MWV), matching llama.cpp's QI4_K=32/VDR=2 architecture. 112 inner-loop instructions / 16 values = 7.0 insn/value (vs MWV 12.4, 1.77x fewer). All threads load scales directly (L1 coalesced, no shfl broadcast). Integer `mul.lo.s32(scale, dot)` avoids 2 cvt per sub-block. Env var: `HW_DP4A_Q4K=1`.

**PMAT-024 cuBLAS GEMM for prefill (Mar 6 2026):** Implemented dequant Q4K→FP32 + cuBLAS SGEMM for all Q4K weight projections during prefill (M >= 4). Q6K weights (attn_v, ffn_down, LM head) still use batched GEMV.

**Jetson Orin Nano Super (Mar 6 2026 — PMAT-024 cuBLAS prefill, MWV DP4A, locked clocks, c=1, 60s streaming):**

| Runtime | Decode tok/s | ITL P50 (ms) | TTFT P50 (ms) | Prefill tok/s |
|---------|-------------|-------------|--------------|--------------|
| llama.cpp | **33.1** | **30.2** | **41** | **2478** |
| realizr (PMAT-024, MWV DP4A) | 21.4 | 46.7 | 1816 | 56.2 |
| **Gap (pre-HW-DP4A)** | **1.55x** | **1.55x** | **44x** | **44x** |

**PMAT-024 impact:** Prefill throughput **1.95x improvement** (28.8→56.2 tok/s), TTFT **1.95x improvement** (3542→1816 ms). Decode unchanged (expected — cuBLAS only activates for M >= 4). Subsequently narrowed further by GH-176 HW DP4A to 1.19x decode, 25x prefill (see table above).

**PMAT-026 Q6K cuBLAS GEMM for prefill (Mar 6 2026):** Extended cuBLAS prefill to Q6K weights (attn_v, ffn_down). Implemented `Q6KDequantKernel` in trueno-gpu that dequantizes Q6K super-blocks (210 bytes → 256 FP32) on GPU, followed by cuBLAS SGEMM. Bug fix: `selp_f32` operand ordering was `(f32, f32, pred)` instead of `(pred, f32, f32)`, causing CUDA_ERROR_INVALID_PTX (error 218). Now all 7 projections per layer use cuBLAS during prefill (M >= 4).

**PMAT-026 impact:** Combined Q4K+Q6K cuBLAS gives same 56.3 tok/s prefill (Q6K was already a small fraction of total prefill time — attn_v and ffn_down are n=1536 or n=8960, much smaller than the Q4K gate/up n=8960 that dominated). The main remaining gap is cuBLAS SGEMM overhead vs llama.cpp's fused quantized GEMM.

**Weight quantization types (Qwen2.5-Coder-1.5B Q4_K_M):**

| Weight | Quant Type | cuBLAS? | Notes |
|--------|-----------|---------|-------|
| attn_q, attn_k, attn_output | Q4_K | Yes | 3/7 projections |
| ffn_gate, ffn_up | Q4_K | Yes | 2/7 projections |
| attn_v | Q6_K | Yes (PMAT-026) | Q6K dequant + cuBLAS SGEMM |
| ffn_down | Q6_K | Yes (PMAT-026) | Q6K dequant + cuBLAS SGEMM |
| output (LM head) | Q6_K | No | Not in batched_attn_ffn_phase path |

**HARDWARE CORRECTION (v2.6.0):** Device is Jetson Orin Nano Super Dev Kit (NOT AGX/NX). Peak memory BW is **67 GB/s** (LPDDR5), not 102 or 204 GB/s. This changes BW utilization calculations: realizr ~20.5 GB/s = 30.6% of 67 GB/s; llama.cpp ~27.4 GB/s = 40.9% of 67 GB/s.

**Jetson Orin (Mar 4 2026 — non-streaming v1, serial isolated benchmarks):**

| Runtime | c=1 tok/s | c=1 decode | c=1 ITL (ms) | c=4 tok/s | c=4 decode | c=4 P50 (ms) |
|---------|-----------|------------|-------------|-----------|------------|------------|
| llama.cpp | **31.9** | **31.9** | **31.4** | **66.2** | **16.5** | 1,934 |
| ollama | 23.4 | 23.3 | 42.8 | 32.6 | 8.2 | 3,907 |
| realizr (GGUF, GPU) | 7.8 | 7.8 | 128.3 | 7.8 | 1.9 | 16,428 |

**Methodology:** Serial isolated benchmarks — each runtime tested alone with all others stopped (`forjar-jetson-{realizr,ollama,llamacpp}.yaml`). Critical for Jetson's 7.4 GB unified memory where concurrent servers cause memory contention (ollama jumped from 11.3 → 23.4 tok/s when isolated, 2.1x improvement). Mar 5 results use SSE streaming (`probador --stream true`) for real per-token TTFT/ITL separation. "Short" prompt profile (~20 tokens).

**Key findings (updated Mar 6 2026):**
- **DECODE gap is 1.19x** (27.8 vs 33.1 tok/s) — GH-176 half-warp DP4A Q4K closed from 1.93x
- **PREFILL gap is 25x** (1045 vs 41ms TTFT) — cuBLAS GEMM + HW DP4A
- **Real decode breakdown** (BrickProfiler, Immediate sync): LmHead 25.7%, FFN Down 25.4%, FFN Gate 23.1%
- **LmHead is the #1 bottleneck**: single Q6K GEMV (n=151936) at 10,948µs per call — as expensive as all 28 FFN layers combined
- Concurrency scaling: llama.cpp 2.1x, ollama 1.4x, realizr 0x (flat, RwLock contention)
- Native CUDA build on Jetson (45 min, no cross-compile) was 7.8x faster than CPU-only (1.0 tok/s)
- Batch mode (`--batch`) OOM-killed on 7.4 GB unified memory

### Hardware Reference

| Host | GPU | Memory BW | VRAM | Role |
|------|-----|-----------|------|------|
| noah-Lambda-Vector (4090) | RTX 4090 | 1,008 GB/s | 24 GB GDDR6X | QLoRA training (full-time) + deep profiling (nsys/ncu, occasional) |
| jetson | Orin Nano Super (nvgpu, sm_87, 8 SMs) | **67 GB/s** | 8 GB LPDDR5 unified | Continuous load testing + CI benchmarks (dedicated) |

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

### Jetson Orin Root Cause Analysis (Updated Mar 5, 2026)

**CRITICAL CORRECTION:** SSE streaming reveals the real bottleneck is **prefill**, not decode. Previous non-streaming measurement (7.8 vs 31.9 tok/s = "4.1x gap") blended both phases. After PMAT-023 (batched prefill default), prefill gap is 18.6x (was 148x with serial prefill).

**Five-Whys: realizr 1816ms prefill vs llama.cpp 41ms on Orin (44x gap, post-PMAT-024)**

1. Why 44x slower prefill? → Q6K weights (attn_v, ffn_down, LM head) still use batched GEMV; Q4K fixed via cuBLAS
2. Why only Q4K fixed? → PMAT-024 dequant kernel (`Q4KDequantKernel`) only handles Q4K super-blocks (144 bytes)
3. Why not Q6K dequant too? → Q6K super-blocks (210 bytes, misaligned) need a separate dequant kernel
4. Why is LM head dominant? → output.weight is Q6K with n=151936 — single largest GEMV call per prefill token
5. Why critical on Orin? → Orin's 67 GB/s BW + 8 SMs make repeated weight reads per M=8 tile unacceptably slow

**Resolution:** PMAT-026 ✅ DONE — Q6K dequant kernel implemented. Bug fix: `selp_f32` operand ordering was swapped (f32 in predicate slot, predicate in false_val slot → CUDA_ERROR_INVALID_PTX). Combined Q4K+Q6K cuBLAS gives 56.3 tok/s prefill (1.95x from baseline). Remaining 44x gap: cuBLAS dequant+SGEMM fundamentally slower than llama.cpp's fused quantized GEMM that reads compressed weights directly.

**Five-Whys: Serial prefill default in streaming path (148x → 18.6x, FIXED)**

1. Why 148x slower? → generate_1.rs processed each prompt token as individual forward pass (serial)
2. Why serial? → `BATCHED_PREFILL` env var defaulted to `false` in generate_gpu_resident_streaming
3. Why false? → Historical: batched prefill was opt-in for profiling, not production default
4. Why mismatch? → generate_2.rs already defaulted to `true`; generate_1.rs was not updated
5. Why now? → Jetson SSE streaming benchmarks exposed the prefill bottleneck that 4090 masked

**Five-Whys: realizr 36.0ms/token decode vs llama.cpp 30.2ms/token (1.19x gap) — UPDATED Mar 6 post GH-176 HW DP4A + real BrickProfiler**

1. Why 1.19x slower decode? → BrickProfiler (Immediate sync) shows **LmHead 25.7%**, **FFN Down 25.4%**, **FFN Gate 23.1%** of decode time. These three bricks account for 74.2%.
2. Why is LmHead #1? → Single Q6K GEMV with n=151,936, k=1536. 10,948µs per call — massive output dimension, called once per token (not per layer).
3. Why are FFN gate/down expensive? → 28 layers × ~370µs each. Q4K GEMV with k=8960 (gate) and n=1536, k=8960 (down). Two separate GEMVs per layer.
4. Why can't fused gate/up help? → PAR-077 attempt was 3x SLOWER. Gate and up share input but fused kernel doubles register pressure.
5. Why not just use cuBLAS for M=1? → cuBLAS GEMV for quantized formats requires dequant + SGEMV, which is slower than fused quantized GEMV for small M.

**Resolution**: GH-176 half-warp DP4A Q4K: 16.7→27.8 tok/s (+66.5%). Gap 1.93x→1.19x. Remaining 1.19x gap dominated by LmHead Q6K GEMV (25.7% of decode). Next: optimize LmHead path (Q6K n=151936).

**Five-Whys: RTX 4090 decode gap — 2.70x → 1.71x (80/20 fix: HW DP4A Q4K)**

1. Why 2.70x slower per layer on 4090? → Forjar config used `DP4A_Q4K=1` (MWV, 32 threads/super-block, 12.4 insn/value) instead of `HW_DP4A_Q4K=1` (half-warp, 16 threads, 7.0 insn/value)
2. Why does thread count matter? → MWV uses shfl-broadcast to distribute scales across 32 threads — high instruction overhead. HW DP4A has each thread load scales directly (L1 coalesced), eliminating broadcast instructions.
3. Why 1.77x fewer instructions? → HW DP4A: 112 inner-loop instructions / 16 values = 7.0 insn/value. MWV: 396 / 32 = 12.4. Half the threads, each doing proportionally less work with no broadcast overhead.
4. Why wasn't HW DP4A tested on 4090? → GH-176 was benchmarked only on Jetson; the 4090 serial pipeline (`bench-gpu-serial`) was created in the same session but used the pre-GH-176 env vars.
5. **ROOT CAUSE:** Config drift — the 4090 forjar config wasn't updated to match the Jetson-proven optimization.

**80/20 Result (verified Mar 6, probador --num-layers 28, isolated serial c=1, 60s):**

| Config | Decode tok/s | µs/layer | Gap to llama.cpp |
|--------|-------------|----------|-----------------|
| llama.cpp | 278.1 | 128.4 | baseline |
| **realizr HW DP4A** | **162.8** | **219.3** | **1.71x** |
| *realizr MWV DP4A (before)* | *106.8* | *334.3* | *2.70x* |

**+52% decode throughput from one env var change.** Gap reduced from 2.70x to 1.71x.

**Remaining 1.71x gap**: Now dominated by Q6K GEMV (LM head + attn_v + ffn_down), attention kernels, and norm/residual overhead. Q4K instruction efficiency is close to parity — further gains require optimizing the Q6K path (PMAT-029).

**Cross-platform insight**: Same kernel, both platforms improved. Jetson: 21.4→27.8 (+30%), 4090: 106.8→162.8 (+52%). 4090 benefits MORE because its 128 SMs amplify instruction-count savings (more warps running the more efficient kernel simultaneously).

**Measurement methodology**: Per-layer decode time (`µs/layer = TPOT_ms * 1000 / num_layers`) is derived from wall-clock ITL, NOT per-brick sync. This makes it:
- **Overhead-free**: no profiler sync artifacts
- **Runtime-agnostic**: same metric for realizr, llama.cpp, ollama
- **Comparable**: probador `--num-layers 28` outputs the same metric for any OpenAI-compat endpoint

**nsys Kernel Profiling (Ground Truth — NOT brick profiler):**

*Post GH-173 (SKIP_CUDA_GRAPH=1, Mar 6 2026):*

| Kernel | Time % | Instances | Avg (µs) | Med (µs) | Phase |
|--------|--------|-----------|----------|----------|-------|
| mwv_dp4a_q4k_gemv | 47% | 11,424 | 179 | 69 | Decode |
| dp4a_q6k_gemv | 28% | 2,280 | 544 | 26 | Decode (bimodal: layers 26µs, LM head ~17ms) |
| batched_q4k_gemv_warp_reduce | 13% | 504 | 1,125 | 393 | Prefill |
| batched_q6k_gemv_warp_reduce | 2% | 28 | 4,526 | 4,497 | Prefill |
| flash_decoding_chunk | 2% | 1,904 | 64 | 63 | Decode |
| q8_quantize | 1% | 13,704 | 5 | 4 | Both |

**Decode per-token budget (~68 tokens):**

| Component | Time (ms) | % of decode |
|-----------|-----------|-------------|
| Q4K GEMVs | 30.0 | 60% |
| Q6K LM head | 17.4 | 35% |
| Q6K layers | 0.7 | 1.5% |
| Flash attention | 1.8 | 3.6% |
| **Total** | **~50** | |

**CRITICAL: LM head Q6K (n=151936) launches 151,936 blocks — fixed by GH-174 grid-stride (pending benchmark).**

**CORRECTION (v2.6.0):** Earlier brick profiling (`apr profile --granular`) claimed AttentionScore was 92% of time. nsys showed this was an artifact of CPU-side synchronization overhead. Actual attention is 0.3% of GPU time.

**CORRECTION (v2.9.0):** BrickProfiler with `Deferred` sync mode (the default) measures only CPU-side kernel launch latency (~26µs for QkvProjection), not actual GPU execution time (~89µs). Fix: `executor_mut().set_profiler_sync_mode(trueno::SyncMode::Immediate)` — GH-176.

### BrickProfiler Decode Breakdown (Mar 6 2026 — GH-176 HW DP4A, Immediate sync, Jetson Orin)

**Ground truth per-brick GPU timing** via CUDA `stream.synchronize()` after each brick. Throughput with sync overhead: 18.7 tok/s (vs 27.8 tok/s without sync — 33% overhead from per-brick sync is expected and acceptable for profiling).

| Brick | Per-call avg (µs) | % of decode | Per-token (µs) | Calls/token |
|-------|-------------------|------------|---------------|------------|
| **LmHead** | **10,948** | **25.7%** | **35.3** | **1** |
| **DownProjection** | **386** | **25.4%** | **34.9** | **28** |
| **GateProjection** | **351** | **23.1%** | **31.7** | **28** |
| AttentionScore | 120 | 7.9% | 10.8 | 28 |
| QkvProjection | 89 | 5.9% | 8.1 | 28 |
| OutputProjection | 53 | 3.5% | 4.8 | 28 |
| RmsNorm | 23 | 3.0% | 4.1 | 56 |
| RopeEmbedding | 24 | 1.6% | 2.2 | 28 |
| Activation | 22 | 1.4% | 1.9 | 28 |
| Residual2 | 20 | 1.3% | 1.8 | 28 |
| Residual1 | 18 | 1.2% | 1.7 | 28 |
| **Total** | | **100%** | **137.3** | |

**Key finding:** LmHead + FFN (gate+down) account for **74.2%** of decode time. Attention is only **17.3%**. The dominant bottleneck for closing the remaining 1.19x gap is the LmHead Q6K GEMV (n=151,936, single call per token).

**Methodology:** `apr cbtop --model-path <model> --headless --iterations 5 --warmup 2 --skip-contract`. JSON output includes `"grade": "R"` (Real) for all bricks. Config: `HW_DP4A_Q4K=1 DP4A_Q6K=1 MWV_Q6K=1 MWV_WARPS=3`. Clocks locked via `sudo jetson_clocks`.

### BrickProfiler Decode Breakdown (Mar 6 2026 — RTX 4090, Immediate sync)

**Ground truth per-brick GPU timing** on RTX 4090 (128 SMs, 1008 GB/s GDDR6X). Throughput with sync overhead: 78.9 tok/s. Production throughput (CUDA graphs): 161.5-195.8 tok/s. llama.cpp on same hardware: 253.1 tok/s. **Decode gap: 1.29x (optimal) to 1.57x (fresh).**

| Brick | Per-call avg (µs) | % of decode | Per-token (µs) | Calls/token |
|-------|-------------------|------------|---------------|------------|
| AttentionScore | 67.9 | 17.3% | 1,902 | 28 |
| **RmsNorm** | **27.6** | **14.4%** | **1,575** | **57** |
| **DownProjection** | 50.3 | 12.8% | 1,408 | 28 |
| **GateProjection** | 48.8 | 12.4% | 1,366 | 28 |
| QkvProjection | 34.8 | 8.9% | 974 | 28 |
| Activation | 30.3 | 7.7% | 849 | 28 |
| Residual2 | 28.4 | 7.2% | 795 | 28 |
| RopeEmbedding | 20.6 | 5.2% | 575 | 28 |
| OutputProjection | 20.6 | 5.2% | 575 | 28 |
| LmHead | 493.7 | 4.5% | 494 | 1 |
| Residual1 | 16.4 | 4.2% | 460 | 28 |
| **Total** | | **100%** | **10,974** | |

**Category breakdown (dramatically different from Jetson):**

| Category | 4090 (µs) | 4090 % | Jetson (µs) | Jetson % |
|----------|-----------|--------|-------------|----------|
| Attention (QKV+Score+Out) | 3,452 | 31.5% | 7,342 | 17.3% |
| FFN (Gate+Down+Act) | 3,623 | 33.0% | 19,768 | 48.5% |
| Norms+Residuals+RoPE | 3,405 | **31.0%** | 3,456 | 8.5% |
| LmHead | 494 | **4.5%** | 10,948 | **25.7%** |

**Key findings:**
1. **LmHead scales with SM count:** 25.7% on Orin (8 SMs) → 4.5% on 4090 (128 SMs). The n=151,936 output is parallelized across 16× more SMs. LmHead optimization (PMAT-028) is Orin-specific — not the bottleneck on 4090.
2. **Norms+Residuals unexpectedly expensive on 4090: 31.0%** vs 8.5% on Jetson. RmsNorm at 27.6µs per call for hidden_dim=1536 (6KB) is **14× roofline** (theoretical: 12KB / 1008 GB/s ≈ 0.012µs). This is dominated by per-brick `stream.synchronize()` overhead (~12µs PCIe round-trip per measurement). In production (CUDA graphs), these elementwise kernels execute back-to-back with zero sync — the 31% cost is a **profiling artifact**, not a production bottleneck.
3. **FFN GEMV percentage drops from 48.5% to 33.0%**: 4090's higher SM count reduces GEMV time proportionally, but sync overhead per measurement remains constant, inflating small-kernel relative share.
4. **Per-brick sync overhead dominates on 4090**: Small kernels (Residual1: 16.4µs, OutputProjection: 20.6µs) include ~12µs of sync overhead — the actual kernel execution may be <5µs. On Jetson (unified memory), sync overhead is ~3-5µs, giving better resolution. **BrickProfiler data on 4090 is most accurate for large kernels (LmHead, Gate, Down, Attention).**

**Corrected production estimate (subtracting ~12µs sync overhead per measurement):**

| Category | Raw µs | Sync overhead | Corrected µs | Corrected % |
|----------|--------|---------------|-------------|-------------|
| Attention | 3,452 | 84×12=1,008 | 2,444 | 32.8% |
| FFN (Gate+Down+Act) | 3,623 | 84×12=1,008 | 2,615 | 35.1% |
| Norms+Res+RoPE | 3,405 | 141×12=1,692 | 1,713 | 23.0% |
| LmHead | 494 | 1×12=12 | 482 | 6.5% |
| **Total** | **10,974** | **2,720** | **7,254** | |

Corrected implied tok/s: 1,000,000 / 7,254 = **137.9 tok/s** — consistent with 161.5 tok/s production (the ~15% remaining gap is CPU overhead, graph launch, and argmax download, which are amortized in production but not measured per-brick).

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
- **Decode gap is 1.19x** (27.8 vs 33.1 tok/s) after GH-176 half-warp DP4A Q4K (Mar 6 2026)
- **4090 decode gap closed 2.70x → 1.71x** by enabling HW DP4A Q4K (219.3 vs 128.4 µs/layer, +52% decode)
- **4090 decode tok/s: 162.8 vs 278.1** (realizr HW DP4A vs llama.cpp, isolated serial c=1, Mar 6)
- **4090 BrickProfiler profile differs from Jetson**: LmHead 4.5% (vs 25.7%), Norms+Res 31.0% (vs 8.5%) — SM scaling + per-brick sync overhead
- **Per-brick sync overhead ~12µs on 4090** (PCIe round-trip) inflates small kernel times; corrected total: ~7.3ms/tok = 137.9 tok/s
- **Prefill gap narrowed from 86x to 25x** via PMAT-024/026 cuBLAS GEMM + HW DP4A
- **cuBLAS GEMM implemented** for Q4K+Q6K prefill (PMAT-024/026): dequant→FP32 + cuBLAS SGEMM
- **`.maxnreg 255` has no effect** — kernel uses only 34 registers, no spill (GH-176, Mar 6 2026)
- **Locked clocks critical**: `sudo jetson_clocks` required for stable, reproducible benchmarks on Orin
- **BrickProfiler MUST use Immediate sync** — Deferred mode only measures CPU launch latency (GH-176, Mar 6 2026)
- **Real decode breakdown**: LmHead 25.7%, FFN Down 25.4%, FFN Gate 23.1%, Attention 17.3% (BrickProfiler Immediate sync)
- **HW DP4A Q4K is optimal kernel** for Orin: 27.8 tok/s vs 21.4 MWV DP4A (+30%), env var `HW_DP4A_Q4K=1`

### Why Tooling Alone Doesn't Close Performance Gaps

Despite extensive investment in profiling (BrickProfiler, nsys, ncu), benchmarking (probador LLM), deployment automation (forjar), and a structured optimization spec (this document), the decode gap persisted at 1.55-1.93x for over a week before GH-176 closed it to 1.19x. This section documents **why** — as a meta-analysis of the optimization process itself.

**Observation:** We had every tool available and still couldn't close the gap. The tools are necessary but not sufficient. Understanding why reveals structural problems in how hand-written GPU kernels diverge from compiler-optimized ones.

**Chain of reasoning:**

1. **Measurement fidelity lag.** Our primary profiler (BrickProfiler) shipped with `Deferred` sync mode as the default. This meant `apr cbtop` — our dedicated profiling tool — reported QkvProjection at 26µs when the real GPU time was 89µs. We built and iterated on optimization hypotheses using CPU-side launch latency, not actual kernel execution time. The nsys data (external tool, harder to integrate) was correct but we only ran it occasionally. **Lesson:** A built-in profiler that gives wrong numbers by default is worse than no profiler — it creates false confidence. The `*` suffix on derived estimates was an honest signal, but the fix was to make real measurement the default path, not a separate mode.

2. **Instruction-level blindness.** Our tools measure time (µs per brick, tok/s) but not *why* a kernel is slow. ncu showed the MWV DP4A Q4K kernel was compute-bound (72% compute, 36% memory), which was surprising for an M=1 GEMV that should be memory-bound. But knowing "compute-bound" doesn't tell you *which instructions* to eliminate. We had to manually count PTX instructions (60+ per super-block) and compare against llama.cpp's CUDA C (~25 instructions) to understand the gap. No automated tool in our stack performs this comparison. **Lesson:** Time-based profiling identifies *where* time is spent. Closing the gap requires *instruction-level* analysis — comparing your kernel's PTX to the competition's SASS, instruction by instruction.

3. **Architectural assumptions baked into the kernel.** The MWV (multi-warp vectorized) Q4K kernel used 32 threads per super-block with warp-shuffle broadcasts for scale extraction. This architecture was designed for the 4090 (128 SMs, high occupancy tolerance). On Orin (8 SMs), the 32-thread design meant each thread processed fewer values, but the scale broadcast overhead (shfl_idx + selp chains) remained constant — a fixed cost amortized over less work. The half-warp DP4A kernel (16 threads/SB) was a fundamentally different architecture, not a parameter tweak. **Lesson:** Profiling tools optimize within an architecture. Crossing the 1.5x→1.2x boundary required changing the kernel's thread-to-data mapping entirely — something no automated tuner would discover from timing data alone.

4. **The "good enough" trap.** At 21.4 tok/s (1.55x gap), GH-173/174/175/176 (.maxnreg, grid-stride, prefetch) all showed no further gains. The optimization appeared to have plateaued — the gap seemed "architectural" (hand-written PTX vs NVCC). But the real issue wasn't that we'd reached a ceiling; it was that we were optimizing the *wrong kernel architecture*. The MWV 32-thread design had 12.4 instructions per value; the half-warp 16-thread design has 7.0. The 1.77x instruction reduction directly translated to a 1.30x throughput improvement (21.4→27.8 tok/s). **Lesson:** When incremental optimizations plateau, the signal isn't "we're done" — it's "we're optimizing the wrong thing." Step back and question the architecture, not the parameters.

5. **Benchmark-profile feedback loop latency.** Each optimization cycle required: write kernel → commit trueno → commit realizar → commit aprender → cross-compile on Intel (9 min) → deploy to Jetson → run benchmark → analyze. This ~20 minute cycle meant we could test ~3 hypotheses per hour. llama.cpp developers iterate locally with `make -j && ./bin/test-backend-ops` in seconds. **Lesson:** The deploy pipeline is optimized for correctness (forjar ensures consistent state), not iteration speed. A local kernel test harness on Jetson would 10x the hypothesis testing rate.

6. **The profiler data we needed existed all along — behind a wrong default.** The BrickProfiler in trueno has `Immediate` sync mode that gives real GPU timing. The realizr executor exposes `set_profiler_sync_mode()`. The `apr cbtop` tool calls `enable_profiling()` but never set the sync mode. This meant the real per-brick breakdown (LmHead 25.7%, FFN 48.5%) was available but invisible. Instead we relied on nsys (external, harder to run) and derived estimates (the "fugazi" `benchmark_bricks()`). Once we fixed the default, the data immediately showed LmHead as #1 target — not Q4K GEMV instruction count, which we'd been optimizing for a week. **Lesson:** The most impactful optimization is often fixing the tooling, not the kernel. One line of code (`set_profiler_sync_mode(Immediate)`) changed our entire understanding of the bottleneck.

**Summary:** Tools measure. Engineers reason. The gap between measurement and action is bridged by:
- **Measurement fidelity** (are we measuring what we think we're measuring?)
- **Instruction-level analysis** (not just where, but why)
- **Architectural willingness** (don't tweak parameters when the design is wrong)
- **Feedback loop speed** (how fast can we test hypotheses?)

The 1.93x→1.19x improvement came not from better tools, but from questioning whether the kernel architecture matched the hardware — then proving it with a new design that required half the threads and 1.77x fewer instructions per value.

### First Principles: System Component Anatomy

Every token generated by realizr traverses six layers. Understanding where time is spent — and why — requires tracing the full path from HTTP request to hardware execution unit. This section documents each layer with real timing data, code paths, and the physical constraints that determine throughput.

**Layer 0: Load Testing (probador → HTTP)**

probador sends `POST /v1/chat/completions` with a ChatML-formatted prompt. The request hits an axum router (`realizar/src/api/router.rs:58`) which dispatches to `openai_chat_completions_handler`. For streaming, `try_cuda_backend()` (`cuda_chat_backend.rs:36-62`) creates a `tokio::sync::mpsc::channel::<Result<u32, String>>(16)` and spawns GPU generation via `tokio::task::spawn_blocking`. The SSE response is returned immediately while tokens flow through the channel.

**Time budget:** HTTP parsing + channel creation + spawn ≈ negligible (<1ms). The 16-slot mpsc channel provides backpressure if the client can't consume tokens fast enough. Each SSE event wraps a `ChatCompletionChunk` JSON payload (~200 bytes). Keep-alive heartbeat every 15s.

**Why this matters:** The `spawn_blocking` boundary is critical. GPU inference cannot be `async` (CUDA calls are blocking), so the tokio runtime must never block on GPU work. A bad implementation (synchronous inference in an async handler) would serialize all concurrent requests even before reaching the GPU.

**Layer 1: Tokenization & Chat Template (CPU)**

The prompt is formatted via ChatML template (`<|im_start|>role\ncontent<|im_end|>\n`) with prompt injection prevention (`sanitize_special_tokens` escapes `<|` to prevent injected control tokens). Then BPE tokenization converts the formatted string to token IDs using merge rules from `tokenizer.json`. A ~20-token prompt produces ~102 BPE tokens after chat template expansion.

EOS token resolution (GH-330): model config EOS (highest priority) → tokenizer lookup for `<|im_end|>` → fallback 0.

**Time budget:** <5ms for a 20-token prompt. Not a bottleneck.

**Layer 2: Generation Loop (`generate_gpu_resident_streaming`, generate_1.rs:27-165)**

The generation loop has two phases: prefill and decode.

*Prefill (lines 80-133):* All prompt tokens except the last are processed to build the KV cache. With `BATCHED_PREFILL=1` (now default, PMAT-023), all tokens go through a single batched forward pass via `executor.prefill_all_layers_gpu()`. Without it, each token requires a separate forward pass — 20× slower for a 20-token prompt.

*Decode (lines 135-162):* The core loop. For each output token:
```
1. forward_gpu_resident_to_token_id(last_token, cache, position)  → u32 token_id
2. if stop_token → break
3. on_token(token_id)  → send via mpsc channel → SSE event
4. position += 1; last_token = token_id
```

Greedy sampling (temperature=0) uses GPU-side argmax (`gpu_argmax` in reduces.rs) — downloads 4 bytes instead of 600KB of logits. Non-greedy uses CPU-side `sample_topk` after downloading full logits.

**Time budget:** Decode loop overhead (CPU side) ≈ <0.5ms/token. The bottleneck is step 1.

**Layer 3: CUDA Graph Capture & Replay (reduces.rs, graphed_capture.rs)**

The first decode token triggers CUDA graph capture: `stream.begin_capture(Global)` → execute full transformer forward → `stream.end_capture()` → `graph.instantiate()`. This records ~280 kernel launches as a single replayable graph.

Subsequent tokens replay the captured graph:
```
1. H2D async: position_buf, seq_len_buf, graph_input_buf  (~20µs, same stream)
2. stream.launch_graph(graph_exec)                          (~10µs API overhead)
3. GPU executes 280 kernels                                 (~36ms at 27.8 tok/s)
4. gpu_argmax: two-pass block reduction                     (vocab=151936 → single u32)
5. D2H: 4 bytes (token ID)                                 (<1µs)
6. cache.advance()                                          (CPU, negligible)
```

**Why CUDA graphs matter:** Without graphs, 280 kernel launches × ~20µs each = 5.6ms overhead per token (16% of decode time). With graphs, one launch × ~10µs = 0.01ms — a 560× reduction in launch overhead.

**Trade-off:** Graph parameters are fixed at capture time. Changing sequence length requires re-capture (handled automatically). The KV cache position is updated via H2D copy (step 1), not baked into the graph.

**Layer 4: Transformer Forward Pass — Per-Token Brick Execution**

Each decode token executes 28 transformer layers plus output norm + LM head. The BrickProfiler (Immediate sync) measures real GPU time per brick type. The ordering within each layer:

```
Per layer (×28):
  RmsNorm (attn)     →  23µs   (normalize hidden state)
  QkvProjection      →  89µs   (3× GEMV: Q, K, V — fused under one BrickId)
  RopeEmbedding      →  24µs   (rotary position encoding)
  AttentionScore     → 120µs   (flash decoding: Q@K^T, softmax, @V)
  OutputProjection   →  53µs   (GEMV: project attention output back)
  Residual1          →  18µs   (add attention output to input)
  RmsNorm (ffn)      →  23µs   (normalize for FFN)
  GateProjection     → 351µs   (GEMV: gate weight, k=1536→n=8960, Q4K)
  Activation         →  22µs   (SwiGLU: gate × silu(up))
  DownProjection     → 386µs   (GEMV: down weight, k=8960→n=1536, Q4K)
  Residual2          →  20µs   (add FFN output to attention output)

After all layers:
  RmsNorm (output)   →  23µs   (final normalization)
  LmHead             → 10,948µs (GEMV: k=1536→n=151,936, Q6K)
```

**Per-token total: ~36ms** (27.8 tok/s). Of this:
- **LmHead: 10.9ms (25.7%)** — one GEMV call, output dimension 151,936 (vocab size)
- **FFN gate+down: 28 × (351+386) = 20.6ms (48.5%)** — 56 GEMV calls total
- **QKV+output+attention: 28 × (89+120+53) = 7.3ms (17.3%)** — 84 GEMV + 28 attention
- **Norms+RoPE+residuals+activation: 28 × (23+23+24+22+18+20) = 3.6ms (8.5%)**

**Key insight:** The LmHead is a single kernel call that costs as much as 28 FFN layers. Its n=151,936 output dimension is 17× larger than the largest per-layer GEMV (n=8,960). This is the #1 optimization target (PMAT-028).

**Layer 5: GEMV Kernel Internals (trueno-gpu PTX)**

trueno generates PTX at runtime via a builder API (`Kernel` trait → `build_ptx()` → PTX text → nvcc → CUBIN → `CudaModule::load()`). This enables hardware-specific kernel generation without shipping pre-compiled binaries for each SM version.

**Half-Warp DP4A Q4K GEMV (HW_DP4A_Q4K=1, the production kernel on Orin):**

Thread organization: 16 threads per super-block (half-warp), 3 warps per block (6 half-warps × 16 threads = 96 threads). Each warp processes 2 independent super-blocks via half-warp ID = `(warp_id << 1) | (lane_id >> 4)`.

Q4K super-block layout (144 bytes → 256 quantized values):
```
Bytes 0-1:    d (f16 scale)
Bytes 2-3:    dmin (f16 minimum)
Bytes 4-15:   scales (12 bytes: 6-bit per sub-block, packed)
Bytes 16-143: qs (128 bytes: 256 × 4-bit quantized values)
```

Per-iteration inner loop (2 iterations: low + high nibbles):
```ptx
// 1. Load 4 quantized bytes (coalesced u32)
ld.global.u32 packed, [qs_base + lane_id*4]
and.b32 nibbles, packed, 0x0F0F0F0F       // extract low/high nibbles

// 2. Load corresponding Q8 activation (already quantized)
ld.global.u32 u8_data, [q8_base + lic*4]

// 3. DP4A: integer dot product of 4 × (u8 × s8) pairs
dp4a.u32.s32 dot, nibbles, u8_data, dot   // dot += Σ(nibble[i] × q8[i])

// 4. Byte sum for min term (reuses same activation data)
dp4a.u32.s32 sum, 0x01010101, u8_data, sum // sum += Σ(q8[i])

// 5. Integer scale multiply (avoids FP conversion per sub-block)
mul.lo.s32 sdot, scale, dot
mul.lo.s32 msum, min, sum

// 6. Single FP conversion + accumulate (once per sub-block pair)
cvt.rn.f32.s32 sdot_f, sdot
cvt.rn.f32.s32 msum_f, msum
fma.rn.f32 acc, q8_d, (d * sdot_f - dmin * msum_f)  // simplified
```

**Instruction density:** 112 instructions for 16 values = **7.0 insn/value** (vs MWV: 99 instructions for 8 values = 12.4 insn/value). The 1.77× reduction comes from:
- 16 threads (not 32) → each thread handles more data, amortizing scale overhead
- Direct lane-to-scale mapping (`ci = lane_id/4` is known) → no shfl broadcast chain
- Integer `mul.lo.s32(scale, dot)` → avoids 2× `cvt.rn.f32.u32` per sub-block

Reduction: warp-synchronous `shfl.down(delta=8,4,2,1)` within each half-warp → shared memory reduction across half-warps → thread 0 stores result.

**Q6K MWV GEMV (LmHead kernel, the #1 bottleneck):**

Q6K super-blocks are 210 bytes → 256 values. Layout: ql (128B, 4-bit low) + qh (64B, 2-bit high) + scales (16B, i8) + d (2B, f16). Each value reconstructed as: `(ql_nibble | (qh_2bits << 4)) - 32`. The 6-bit encoding requires 3 separate loads (ql, qh, scales) vs Q4K's single packed load — inherently more instruction-heavy.

Additional overhead: 210 bytes per super-block is misaligned (210 mod 4 = 2). Odd-indexed super-blocks cause `ld.global.u32` misalignment on sm_87. Fix: `ld_global_u32_unaligned()` via 4× byte loads + shifts (GH-129, trueno `c4d2bea`).

**Layer 6: Hardware Execution (Jetson Orin Nano Super, sm_87)**

- **8 SMs** (streaming multiprocessors), each with 128 CUDA cores = 1024 total
- **Max clock:** 918 MHz (locked via `sudo jetson_clocks` — without this, DVFS throttles to ~600 MHz under thermal pressure)
- **Memory:** 8 GB LPDDR5 unified (CPU+GPU shared), **67 GB/s peak bandwidth**
- **DP4A:** Native INT8 dot product (4 multiplies + 3 adds in 1 cycle) — sm_87 has dedicated DP4A functional units
- **L1 cache:** 128 KB per SM (configurable shared/L1 split)
- **L2 cache:** 1 MB shared across all 8 SMs

**Memory bandwidth is the theoretical ceiling:**
- Qwen2.5-Coder-1.5B Q4_K_M total weights: ~850 MB
- At 67 GB/s: minimum weight read time = 850/67000 = **12.7ms per token**
- At 27.8 tok/s: actual time = 36ms → **BW utilization = 12.7/36 = 35.2%**
- llama.cpp at 33.1 tok/s: 30.2ms → **BW utilization = 12.7/30.2 = 42.1%**
- **Gap in BW utilization: 1.20×** — consistent with measured 1.19× throughput gap

**Why we can't reach 100% BW utilization:** Even a perfect GEMV kernel achieves ~65-70% of peak BW on LPDDR5 (vs ~85% on GDDR6X). The unified memory architecture shares bandwidth between CPU and GPU. Additionally, L2 cache misses, TLB pressure from 850MB working set, and instruction execution overhead (even at 7.0 insn/value) all contribute. llama.cpp's ~42% is close to practical ceiling for this hardware.

**The 6.8ms gap (36ms - 29.2ms theoretical at 42% BW):**
- LmHead overhead: Q6K GEMV at 10.9ms vs theoretical ~4ms (Q6K kernel not yet optimized with half-warp)
- FFN overhead: 20.6ms vs theoretical ~15ms at llama.cpp's BW efficiency
- Instruction overhead: compute-bound kernels waste cycles on dequant arithmetic even when data is ready

**Summary: Where does 36ms go?**

```
Layer                          Time     Source
─────────────────────────────────────────────────────
L6: DRAM→L2→L1 (weight reads)  ~12.7ms  850MB @ 67 GB/s (theoretical minimum)
L5: Dequant + DP4A arithmetic   ~11.3ms  Compute-bound overhead (7.0 insn/val × 850M vals / 8 SMs)
L5: LmHead Q6K excess           ~6.9ms   Q6K not yet half-warp optimized (10.9ms vs ~4ms at Q4K efficiency)
L4: Attention + norms + residual ~3.6ms   Non-GEMV bricks (8.5% of decode)
L3: Graph launch + argmax        ~0.5ms   CUDA overhead
L0-L2: HTTP + tokenize + loop    ~1.0ms   Application overhead
─────────────────────────────────────────────────────
Total                            ~36ms    = 27.8 tok/s
```

The path to parity is clear: eliminate the 6.9ms LmHead Q6K excess (half-warp Q6K, PMAT-028) and reduce FFN dequant overhead further. At 42% BW utilization (matching llama.cpp), the floor is ~30ms = 33 tok/s — exactly llama.cpp's measured throughput.

For full analysis including the "impossible observation" (CPU outperforming GPU), see [root-cause-analysis.md](./components/root-cause-analysis.md).

---

## 6. Optimization Roadmap

### Tier Summary (Updated Mar 6 2026 — post GH-176 HW DP4A + real BrickProfiler)

| Tier | Speedup Range | Items | Status |
|------|---------------|-------|--------|
| T0: Completed | Shipped | 6 fixes + PMAT-023/024/026 + GH-173/176 | ✅ Production |
| **T0e: LmHead Q6K** | **~1.34x decode** | **LmHead GEMV optimization (25.7% of decode)** | **#1 PRIORITY** |
| **T0f: FFN gate/down** | **~1.19x decode** | **Per-layer GEMV optimization (48.5% of decode)** | **#2 PRIORITY** |
| **T0d: Fused Q4K GEMM** | **~25x prefill** | **Fused quantized GEMM (read Q4K directly, no dequant)** | **#3 PRIORITY** |
| T1: Critical | 2-5x | SageAttention, EAGLE, CUDA graphs | Planned |
| T2: High Impact | 1.5-2x | Marlin, DCA, KV quant, MInference | Mixed |
| T3: Incremental | 1.1-1.5x | 3-way fusion | ✅ Mostly done |

### Priority Matrix

| ID | PMAT | Optimization | Speedup | Status |
|----|------|--------------|---------|--------|
| — | PMAT-023 | **Batched prefill default** | **9x TTFT** | **✅ DONE (148x → 18.6x gap)** |
| — | PMAT-024 | **Prefill GEMM kernel (cuBLAS Q4K)** | **1.95x prefill** | **✅ DONE (86x→44x gap, Q4K only)** |
| — | PMAT-026 | **Prefill GEMM for Q6K (cuBLAS dequant)** | **1.95x prefill** | **✅ DONE (86x→44x gap with Q4K+Q6K)** |
| GH #118 | PMAT-019 | **Q6K MWV GEMV kernel** | **2-4x Q6K** | **Planned (31.9% GPU time)** |
| — | PMAT-022 | **Q6K MWV as default** | **1.3x decode** | ✅ Code done (env var MWV_Q6K=1) |
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
| **Decode gap (streaming)** | **1.19x** (27.8 vs 33.1 tok/s) | probador --stream Mar 6, GH-176 HW DP4A |
| **Prefill gap** | **25x** (1045 vs 41ms TTFT) | probador --stream Mar 6, GH-176 + cuBLAS |
| **Prefill throughput** | 97.8 tok/s (vs 2478 llama.cpp) | probador --stream Mar 6, GH-176 |
| **BW utilization (realizr)** | 35.2% of 67 GB/s | HW DP4A, Mar 6 |
| **BW utilization (llama.cpp)** | 40.9% of 67 GB/s | calculated |
| **LmHead % of decode** | 25.7% (10,948µs per call) | BrickProfiler Immediate sync, Jetson, Mar 6 |
| **FFN (gate+down) % of decode** | 48.5% | BrickProfiler Immediate sync, Jetson, Mar 6 |
| **4090 decode (PMAT-040)** | **1.06x** (411.7 vs 436.9 tok/s, avg_tok=128) | isolated c=1, 60s stream, Mar 8 |
| **Jetson decode (PMAT-040)** | **0.91x** (36.2 vs 33.0 tok/s) — **realizr 10% FASTER** | isolated c=1, 60s stream, Mar 8 |
| *4090 decode gap (pre-040)* | *1.64x short, 1.92x long* | *GpuProfile auto, serial c=1, Mar 6-7* |
| **4090 TTFT P50** | 58.8 vs 5.8 ms (10.1x) | prefill-dominated, HGEMM prefill |
| **4090 prefill tok/s** | 1734 vs 17620 (10.2x) | FP16 reads vs Q4K fused GEMM |
| **Jetson prefill tok/s** | 447.7 vs 2488.9 (5.6x) | HGEMM prefill, isolated |
| *4090 decode gap (before)* | *2.70x per layer (334.3 vs 123.8 µs/layer)* | *MWV DP4A, same methodology* |
| **4090 LmHead % of decode** | 4.5% (493.7µs per call) | BrickProfiler Immediate sync, 4090, Mar 6 |
| **4090 Norms+Res % of decode** | 31.0% (raw), 23.0% (sync-corrected) | BrickProfiler 4090, Mar 6 |
| **4090 FFN % of decode** | 33.0% (raw), 35.1% (sync-corrected) | BrickProfiler 4090, Mar 6 |

### Nsight Compute Per-Kernel Profile (2026-03-06, Jetson Orin)

**CRITICAL FINDING: Q4K GEMV is COMPUTE-BOUND, not memory-bound.**

ncu profiling (basic set, 8 replay passes per kernel, CUDA_GRAPH=0):

| Kernel | Grid | Block | Regs | Theo Occ | Achieved Occ | Mem BW % | Compute % |
|--------|------|-------|------|----------|-------------|----------|-----------|
| `mwv_dp4a_q4k_gemv` | 256 | 96 | 34 | 100% | 80% | 27% | 53% |
| `mwv_dp4a_q4k_gemv` | 1,536 | 96 | 34 | 100% | 93% | 36% | **72%** |
| `mwv_dp4a_q4k_gemv` | 8,960 | 96 | 34 | 100% | 95% | 39% | **75%** |
| `dp4a_q6k_gemv` | 256 | 96 | 40 | 100% | 81% | 45% | 59% |
| `batched_q4k_gemv` | 256 | 32 | 40 | 33% | 31% | 19% | 52% |

**Analysis:**
- Occupancy is NOT the bottleneck (80-95% achieved, 100% theoretical for MWV)
- **Compute throughput (72-75%) >> Memory throughput (36-39%)**: excessive dequantization arithmetic
- Scale extraction alone: ~57 instructions to decode all 16 values, then ~14 selp to select 4 = **71 instructions**
- llama.cpp uses parallel byte masks (0x3F3F3F3F) for ~8 instruction scale extraction
- BFE regression explained: kernel was already compute-bound, higher-latency BFE made it worse
- **Fix (GH-173)**: Parallel byte-masked extraction reduces scale handling from 79 → 35 instructions (56%)

### Roofline Position

```
GEMV (M=1):  ~2 FLOP/byte → SHOULD BE MEMORY BOUND
                              ACTUALLY COMPUTE BOUND due to dequant overhead (ncu Mar 6)
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
| PMAT-022 | GH #118 | Q6K MWV GEMV default (was 442µs single-warp) | ✅ Done (MWV default, Refs #118) |
| PMAT-023 | — | **Batched prefill default (DONE)** | ✅ DONE (148x → 86x gap) |
| PMAT-024 | — | **Prefill GEMM kernel (cuBLAS, 3542ms → target <100ms)** | **NEW — #1 PRIORITY (86x gap)** |
| PMAT-025 | GH-176 | `.maxnreg 255` PTX directive (no impact — 34 regs) | ✅ Done (no perf change) |
| PMAT-026 | GH-176 | **Half-warp DP4A Q4K GEMV** (16 thr/SB, 7.0 insn/val) | **✅ DONE (+66.5%, 1.19x gap)** |
| PMAT-027 | GH-176 | **BrickProfiler Immediate sync** (real GPU timing in cbtop) | **✅ DONE (cbtop JSON grade "R")** |
| PMAT-028 | — | **LmHead Q6K GEMV optimization** (25.7% of decode on Orin, n=151936) | **NEW — #1 PRIORITY (Orin)** |
| PMAT-029 | — | **Vectorized byte-mask scale extraction** (Q4K compute→memory bound) | **NEW — #1 PRIORITY (4090)** |

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
| 2.14.0 | 2026-03-08 | **DECODE NEAR-PARITY ACHIEVED.** PMAT-038: Fixed CUDA graph capture for HW DP4A + fused gate+up+SwiGLU. PMAT-039: BFE byte extraction (108→103 insn/SB, no measurable impact — memory-bound). PMAT-040: Flash Decode chunk_size 128→32 — THE BREAKTHROUGH. With chunk_size=128, sequences <128 tokens got 1 chunk = zero split-K parallelism. Reducing to 32 gives 2-4 chunks, enabling actual SM utilization. **4090**: 266→412 tok/s (1.55x improvement), decode gap 1.64x→**1.06x** (near parity). **Jetson**: 32.7→36.2 tok/s (+10.7%), now **10% FASTER** than llama.cpp (33.0 tok/s). Remaining gaps: prefill 5.6-10x (FP16 reads vs Q4K, requires fused Q4K tiled GEMM), TTFT 10.1x (prefill-dominated). |
| 2.13.0 | 2026-03-06 | **Kaizen: Flash Decoding sync removal + attention scaling analysis.** Removed unnecessary `stream.synchronize()` between Flash Decoding chunk and reduce kernels — CUDA stream semantics guarantee ordering within a stream. Sync was based on misunderstanding of CUDA API. Also added GpuProfile auto-detection (compute_capability-based kernel selection, replacing 9 env vars in forjar configs). New finding: 4090 decode gap is **sequence-length dependent** — 1.64x at avg_tok=32 but 1.92x at avg_tok=128. Root cause: realizr attention adds 39µs/layer per 4x more KV entries while llama.cpp adds only 2µs/layer (FlashAttention-2 scales better). Next target: attention kernel scaling with KV cache length. |
| 2.12.0 | 2026-03-06 | **80/20 fix: HW DP4A Q4K on 4090.** Five-Whys root cause: forjar config used MWV DP4A (`DP4A_Q4K=1`) instead of half-warp DP4A (`HW_DP4A_Q4K=1`). Single env var change: decode 106.8→162.8 tok/s (+52%), gap 2.70x→1.71x per layer (219.3 vs 128.4 µs/layer). Remaining 1.71x gap: Q6K GEMV (LmHead + attn_v + ffn_down), attention, norms. Updated forjar-gpu-realizr.yaml + forjar-gpu.yaml. |
| 2.11.0 | 2026-03-06 | **4090 per-layer benchmark + Five-Whys.** Added probador `--num-layers` for overhead-free cross-runtime comparison (µs/layer from wall-clock ITL, not per-brick sync). Verified 4090 gap: **2.70x per layer** (334.3 vs 123.8 µs/layer, isolated serial c=1). Prior BrickProfiler estimate (1.29-1.57x) was misleading due to sync overhead artifacts. Added forjar-gpu-realizr.yaml + forjar-gpu-llamacpp.yaml for isolated 4090 benchmarks. BrickProfiler 4090 breakdown: Attention 31.5%, FFN 33.0%, Norms+Res 31.0%, LmHead 4.5%. Five-Whys root cause: trueno PTX instruction overhead (7.0 insn/val vs llama.cpp ~4-5). New tickets: PMAT-029 (byte-mask scale extraction), GH-8. |
| 2.10.0 | 2026-03-06 | **First Principles: System Component Anatomy.** New section tracing every layer of the inference stack from HTTP request (probador→axum→SSE) through tokenization, CUDA graph capture/replay, per-brick transformer execution (28 layers + LmHead), GEMV kernel internals (half-warp DP4A PTX, Q6K MWV), down to Orin hardware (8 SMs, 67 GB/s LPDDR5, DP4A units). Full 36ms per-token breakdown across 6 layers with real BrickProfiler data. Shows path to parity: 6.9ms LmHead Q6K excess + 5.6ms FFN dequant overhead = the 1.19x gap. |
| 2.9.0 | 2026-03-06 | **GH-176: Half-warp DP4A Q4K + real BrickProfiler.** (1) Half-warp DP4A Q4K GEMV: 16 threads/SB (vs 32 MWV), 7.0 insn/val (vs 12.4), matching llama.cpp's QI4_K architecture. Decode: 21.4→27.8 tok/s (+66.5%), gap narrowed 1.55x→1.19x. (2) Fixed `apr cbtop` to output real GPU timing via BrickProfiler `Immediate` sync mode (was measuring CPU-side launch latency only). (3) Real decode breakdown: LmHead 25.7% (n=151936 Q6K), FFN Down 25.4%, FFN Gate 23.1%. Top 3 bricks = 74.2% of decode. New #1 priority: LmHead Q6K GEMV optimization (PMAT-028). |
| 2.8.0 | 2026-03-06 | **PMAT-026: cuBLAS GEMM for prefill (Q6K).** Extended cuBLAS prefill to Q6K weights (attn_v, ffn_down). Q6KDequantKernel in trueno-gpu: dequantizes Q6K super-blocks (210 bytes → 256 FP32) on GPU. Bug fix: `selp_f32` operand ordering swapped (f32 as pred, pred as false_val → CUDA_ERROR_INVALID_PTX error 218). All 7 projections per layer now use cuBLAS during prefill. Combined Q4K+Q6K result: 56.3 tok/s prefill (same as Q4K-only — Q6K is small fraction). Remaining 44x gap: dequant+SGEMM vs llama.cpp's fused quantized GEMM. New priority: decode instruction count reduction (compute-bound 72%). |
| 2.7.0 | 2026-03-06 | **PMAT-024: cuBLAS GEMM for prefill (Q4K).** Implemented dequant Q4K→FP32 + cuBLAS SGEMM for all Q4K weight projections during prefill (M >= 4). TTFT: 3542→1816ms (1.95x). Prefill tok/s: 28.8→56.2 (1.95x). Gap narrowed from 86x to 44x. Remaining gap: Q6K weights (attn_v, ffn_down, LM head = 2/7 + LM head) still use batched GEMV. PTX dispatch bug fixed: Q4KDequant was missing from `kernels_generate_gemm_cuda.rs` runtime dispatch chain. New ticket: PMAT-026 (Q6K cuBLAS dequant). |
| 2.6.0 | 2026-03-06 | **GH-173/174/175/176 optimization sweep + hardware correction.** Decode narrowed from 1.8x to 1.55x (21.4 vs 33.1 tok/s) via GH-173 parallel byte-masked scale + locked clocks. GH-174 grid-stride, GH-175 prefetch, GH-176 `.maxnreg 255` all confirmed no impact — remaining gap is architectural. Hardware corrected: Jetson Orin Nano Super with 67 GB/s peak BW (was 102 GB/s). ncu profiling: kernel is COMPUTE-BOUND (72% compute, 36% memory). BW utilization recalculated: realizr 30.6%, llama.cpp 40.9% of 67 GB/s. PMAT-024 (cuBLAS prefill GEMM) remains #1 priority (86x gap). |
| 2.5.0 | 2026-03-05 | **PMAT-023: Batched prefill default.** Root cause: generate_1.rs (streaming) defaulted BATCHED_PREFILL=false, causing 150 serial forward passes instead of 1 batched. Fix: default to true (matching generate_2.rs). TTFT improved 7.2s→816ms (9x). Remaining gap: 18.6x (batched GEMV M=8 tiles vs cuBLAS GEMM). New priority: PMAT-024 (cuBLAS prefill GEMM). |
| 2.4.0 | 2026-03-05 | **SSE streaming reveals prefill bottleneck:** probador --stream separates TTFT from decode. Decode gap 1.9x (17 vs 32 tok/s), not 4.1x. Prefill 148x slower (7.2s vs 48ms). New #1 priority: GEMM kernel for prefill (PMAT-023). Q6K MWV default committed (PMAT-022). probador overhaul: 6 issues fixed (#25-#30): rate control, SLO/goodput, token batching robustness. |
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
