# GPU Decoder Throughput Performance Specification

**Document ID:** REALIZAR-GPU-PERF-001
**Version:** 2.48.0
**Status:** ACTIVE
**Date:** 2026-03-12
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

**Competition Reality (Mar 12, 2026 — yoga RTX 4060L @ 1900MHz):** Under standardized load testing (c=1, 60s, streaming):
- **Decode parity:** realizr 154.8 tok/s vs llama.cpp 160.7 (**0.96x**), vLLM 168.3, ollama 163.5
- **TTFT parity:** realizr 13.4ms vs llama.cpp 10.1ms (**1.33x**, PASS < 2x target)
- **c=4 aggregate:** realizr 264.6 vs llama.cpp 338.6 (0.78x) vs vLLM 598.0 (0.44x) — FP8 decode breaks DP4A ceiling
- **c=8 aggregate:** realizr 432.3 tok/s (FP8) vs 324.8 (DP4A) — **+33%**, tensor cores beat DP4A compute ceiling
- **Cross-platform:** Jetson Orin realizr **13% FASTER** than llama.cpp on decode (40.8 vs 36.1 tok/s)

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
Yoga (PRIMARY benchmark target)          4090 Host (QLoRA training + deep profiling)
├── realizr    :8081  (GGUF, CUDA)       ├── QLoRA fine-tuning (full-time)
├── ollama     :8082  (GGUF, CUDA)       ├── Deep profiling (occasional):
├── llama.cpp  :8083  (GGUF, CUDA)       │   nsys-gpu, ncu-gpu, profile-gpu
├── vLLM       :8084  (AWQ INT4)         └── Builds: apr, llama.cpp, trueno
└── RTX 4060 Laptop, sm_89, 8GB

Jetson Orin (secondary load testing)
├── realizr    :8081  (GGUF, CUDA)
├── llama.cpp  :8083  (GGUF, CUDA)
└── 8 SMs, sm_87, 8GB unified, MAXN_SUPER 1020MHz
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

Standardized load test: `probador llm load` (60s, streaming, isolated). Model: Qwen2.5-Coder-1.5B Q4_K_M.

**RTX 4060 Laptop — yoga (Mar 12 2026, c=1, isolated, streaming, 1900MHz, PMAT-087):**

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|-------------|--------------|---------------|-------------|
| **vLLM** | **168.3** | 2,016 | 11.4 | **5.9** |
| ollama | 163.5 | 326 | 70.5 | 6.1 |
| llama.cpp | 160.7 | **2,280** | **10.1** | 6.2 |
| realizr | 154.8 | 1,718 | 13.4 | 6.5 |

**DECODE PARITY ACHIEVED (c=1).** realizr 154.8 vs llama.cpp 160.7 = **0.96x** — within measurement noise. All four runtimes within 8% on decode.
TTFT: realizr 13.4ms vs llama.cpp 10.1ms = **1.33x** (PASS < 2x target). FP8 prefill via cuBLASLt.

**RTX 4060 Laptop — yoga (Mar 12 2026, c=4, isolated, streaming, 1900MHz, PMAT-088):**

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| **vLLM** | **598.0** | **162.3** | 23.4 | **6.2** |
| llama.cpp | 338.6 | 86.0 | **17.3** | 11.6 |
| realizr | 210.8 | 72.4 | 41.0 | 13.8 |
| ollama | 158.0 | 160.6 | 616.0 | 6.2 |

**c=4 gap: 1.6x aggregate (realizr vs llama.cpp), 2.8x vs vLLM.** realizr PMAT-088d latest: 257.4 aggregate (+22% via continuous batch recycling). Root causes of remaining gap: (1) DP4A GEMV ceiling at M=4 = 306 tok/s — need W4A16 tensor core GEMM, (2) serial prefill overhead. vLLM dominates via continuous batching + PagedAttention.

**Cross-platform decode summary (c=1, isolated, streaming):**

| Platform | vLLM | realizr | llama.cpp | ollama |
|----------|------|---------|-----------|--------|
| **RTX 4060 Laptop** (24 SMs, 1900MHz) | **168.3** | 154.8 | 160.7 | 163.5 |
| RTX 4090 (128 SMs) | — | 411.7 | 436.9 | — |
| Jetson Orin (8 SMs, MAXN_SUPER 1020MHz) | — | **40.8** | 36.1 | — |

**RTX 4090 (Mar 4 2026 — historical, c=4, non-streaming):**

| Runtime | Tokens/s | Decode tok/s | Latency P50 (ms) |
|---------|----------|-------------|-------------------|
| llama.cpp | **931.5** | **233.5** | 548 |
| ollama | **561.6** | **139.9** | 915 |
| realizar (GGUF) | 151.4 | 40.8 | 2,530 |

**Jetson Orin (MAXN_SUPER 1020MHz, isolated, streaming):** realizr **40.8 tok/s** vs llama.cpp 36.1 = **13% FASTER** on decode. TTFT: 47.8ms vs 34.0ms = **1.4x** (PASS < 2x target). c=4 zero scaling benefit (8 SMs fully saturated at M=1).

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

**Jetson Orin Nano Super (Mar 12 2026 — MAXN_SUPER 1020MHz, PMAT-078/088d, isolated, c=1, 60s streaming):**

| Runtime | Decode tok/s | ITL P50 (ms) | TTFT P50 (ms) | Prefill tok/s |
|---------|-------------|-------------|--------------|--------------|
| **realizr** | **40.8** | **24.5** | 47.8 | 481 |
| llama.cpp | 36.1 | 27.7 | **34.0** | **676** |
| **Gap** | **0.88x (realizr 13% faster)** | **0.88x** | **1.4x** | **1.4x** |

**Key change:** Previous baselines (36.3/33.1 tok/s) were measured at MAXN_SUPER but Jetson later reverted to 15W mode (GPU 612 MHz) after a reboot — not a code regression. MAXN_SUPER (GPU 1020 MHz, `nvpmodel -m 2 && jetson_clocks`) is required for all Jetson benchmarks. Also: trueno#184 PTX JIT target fix (sm_90 clamp instead of sm_70), PMAT-078 Q6K shared memory Q8 cache, PMAT-088d continuous batching. c=4 provides zero benefit (8 SMs saturated at M=1).

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

### Scorecard (Mar 11 2026, v3.1.0 — 9 dimensions, PMAT-087 clock correction)

**Tool:** `probador llm score` with 9 scoring dimensions (contract: `scoring.yaml` v3.0.0).

**RTX 4060 Laptop — yoga (c=1 and c=4, isolated, streaming, locked clocks 1900MHz):**

| Dimension | realizr | llama.cpp | vLLM | ollama | Target |
|-----------|---------|-----------|------|--------|--------|
| **Composite (c=1)** | **98 A+** | 99 A+ | 100 A+ | 78 B | >= 90 A |
| **Layer decode (c=1)** | **97 A+ (230.8)** | 99 A+ (222.3) | 100 A+ (212.2) | 100 A+ (218.4) | >= 90 A |
| **Prompt profile** | 98 A+ | 99 A+ | 100 A+ | 78 B | >= 90 A |
| **Output length** | 97 A+ | 99 A+ | 100 A+ | 99 A+ | >= 90 A |
| **Correctness** | 100 A+ | 92 A | — | — | >= 90 A |
| **Memory** | 95 A+ (36.1 tok/s/GB) | — | — | — | >= 90 A |
| **Cold start (TTFT)** | 99 A+ (13.4ms) | 100 A+ (10.1ms) | 100 A+ (11.4ms) | 53 C (70.5ms) | >= 90 A |
| **Power** | 100 A+ (3.13 tok/s/W) | — | — | — | >= 90 A |
| **Concurrency scaling** | **51 C (34.1%)** | 77 B (52.7%) | 99 A+ (88.9%) | 36 D- (24.2%) | >= 90 A |

**PMAT-087: Clock correction (1500→1900 MHz).**
The benchmark locked SM clocks at 1500 MHz — 26% below the GPU's natural sustained boost
(1890 MHz measured via `nvidia-smi` during unlocked decode). This artificially penalized all
runtimes equally. Corrected to 1900 MHz (matching natural sustained boost).

Results at 1900 MHz vs 1500 MHz:
| Metric | 1500 MHz | 1900 MHz | Δ |
|--------|----------|----------|---|
| realizr c=1 decode | 138.6 tok/s | **154.8 tok/s** | +11.7% |
| realizr c=1 µs/layer | 258.6 | **230.8** | -10.7% |
| realizr c=1 TTFT | 46.4ms | **13.4ms** | -71.1% |
| llama.cpp c=1 decode | 142.9 tok/s | 160.7 tok/s | +12.5% |
| realizr→llama.cpp gap | 3.0% | 3.7% | unchanged |

**Why higher SM clocks help a "memory-BW-bound" decode workload:**
- Q4K GEMV dequantization (DP4A, shared memory ops) is ~15% compute
- At 1500 MHz: BW utilization 45% (compute bottleneck limits pipeline)
- At 1900 MHz: BW utilization 51% (compute freed, BW limit reached sooner)
- Evidence: 33% SM clock increase → 13% speedup (consistent with 15% compute fraction)

**8 of 9 dimensions at A or above. Single remaining gap:**

| Dimension | Score | Need | Root Cause | Fix |
|-----------|-------|------|------------|-----|
| Concurrency scaling | 51 C (34.1% efficiency) | 90 A (≥74% efficiency) | Batched GEMV: attention KV scales linearly with M, per-slot decode degrades 2.55x at M=4 | GH-141: Continuous batching + PagedAttention (architectural change, multi-week) |

**PMAT-086 analysis (Mar 11):** Pre-allocated GPU input/logits buffers + removed
redundant sync. Measured <0.1ms improvement — kernel time dominates at M=4.

**Dimensions at A or above (8/9):**
- Composite c=1: 98 A+ (best non-vLLM composite)
- Layer decode: 97 A+ (230.8 us/layer, 154.8 tok/s)
- Prompt profile: 98 A+ (consistent across short/medium)
- Output length: 97 A+ (consistent ITL across output lengths)
- Correctness: 100 A+ (32/32 pass rate)
- Memory: 95 A+ (36.1 tok/s/GB — GGUF Q4K uses 3.5x less VRAM than vLLM AWQ)
- Cold start: 99 A+ (TTFT 13.4ms, improved 71% from clock correction)
- Power: 100 A+ (3.13 tok/s/W)

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

### Remaining Gaps — Five-Whys Root Cause Analysis (Mar 12 2026, PMAT-088 baseline)

**c=1 decode parity achieved (0.96x) and TTFT within target (1.33x). Two gaps remain: c=4 aggregate vs llama.cpp and vs vLLM.**

#### ~~Gap 1: TTFT/Prefill~~ RESOLVED (PMAT-071→087, Mar 12)

**TTFT: 13.4ms vs 10.1ms = 1.33x (PASS < 2x target).** Closed from 9.4x (Mar 8) via:
- PMAT-053b: FP8 E4M3 per-tensor absmax scaling via cuBLASLt alpha parameter
- PMAT-079→086: Async FP8 pipeline, JIT warmup, descriptor caching, non-blocking drain
- PMAT-087: Clock correction 1500→1900MHz

**Remaining 0.33x (3.3ms):** FP8 reads 1 B/elem vs Q4K 0.56 B/elem (1.78x BW ratio). Further improvement requires direct Q4K tensor-core GEMM — all tested approaches (DP4A, WMMA, L2 dequant) were slower than cuBLAS on sm_89.

#### Gap 2: c=4 aggregate 0.62x vs llama.cpp (210.8 vs 338.6 tok/s)

**Five-Whys (updated for PMAT-088):**

1. Why 0.62x aggregate? → DP4A GEMV is compute-bound at M=4 (2.425ms/token/slot).
2. Why compute-bound? → Q4K dequant requires ~70 instructions per super-block. At M=4, 4× independent DP4A accumulation chains saturate INT32 units.
3. Why not tensor cores? → PMAT-088b FALSIFIED: HGEMM at M=4 reads 3.5x more BW (FP16 2 B/elem vs Q4K 0.5625) — tensor core savings don't compensate. Both approaches tested identically (256 vs 261 aggregate).
4. Why not CUDA graphs? → PMAT-088b FALSIFIED: Graph replay 3ms SLOWER (654 kernels, frozen attention grids, RoPE positions). Eager is optimal.
5. Why does llama.cpp reach 338.6? → Same DP4A kernel architecture but ~34% more efficient per-step (11.3ms vs 15.1ms ITL). Root cause: fewer dequant instructions + no shared memory management overhead.

**Root cause:** Per-step kernel efficiency gap (1.34x). DP4A ceiling at M=4 = 306 tok/s; realizr at 257.4 = 84% of ceiling. llama.cpp exceeds theoretical model due to more efficient Q4K dequant.

**Fix:** PMAT-072/073/074/088 (continuous batching) are DONE. Remaining improvement path:
- **Kernel optimization:** Reduce Q4K dequant instruction count (trueno GEMV kernel, ~20→5 insn for scale extraction)
- **Chunked prefill:** Interleave prefill chunks with decode to reduce TTFT at c=4 (41→~20ms)
- **W4A16 tensor core GEMM:** INT4 storage + FP16 compute (Marlin-style) — breaks DP4A ceiling entirely

#### Gap 3: c=4 aggregate 0.35x vs vLLM (210.8 vs 598.0 tok/s)

**Five-Whys (updated for PMAT-088):**

1. Why 2.8x gap to vLLM? → vLLM c=4 decode barely degrades from c=1 (162 vs 168, -4%). realizr degrades 53% (72.4 vs 154.8).
2. Why does realizr degrade? → DP4A GEMV compute-bound at M>1 (4× accumulation chains). vLLM Marlin kernel maintains M=1 efficiency at M=4 via tensor core HMMA.
3. Why doesn't vLLM degrade? → W4A16 architecture: INT4 weight storage + FP16 tensor core compute. Memory-bound even at M=4 (reads INT4, computes FP16). PagedAttention: <4% memory waste.
4. Why can't realizr do this? → Q4K format requires scalar dequant to Q8_1 before DP4A. No INT4→FP16 in-kernel dequant path.
5. Why no Marlin-style kernel? → Requires offline INT4 weight reshuffling + FP16 activation path + HMMA PTX. Multi-week implementation (PMAT-054B).

**Root cause:** Architectural — DP4A Q4K GEMV becomes compute-bound at M>1, while vLLM's Marlin W4A16 stays memory-bound. The 2.8x gap is primarily kernel architecture, not scheduling (realizr's iteration scheduler + recycling are now functional).

**Fix path:** W4A16 tensor core GEMM kernel (PMAT-054B) is the only approach that breaks the DP4A ceiling. All scheduling improvements (PMAT-072→088) are complete and yielded 257.4 tok/s (84% of DP4A ceiling).

**Trajectory (actual, PMAT-072→088 all DONE):**
```
Pre-CB:         197.5 aggregate tok/s (0.33x vLLM)
+ PMAT-072:     197.5 (lock release — structural prerequisite) ✓
+ PMAT-073:     197.4 (mid-batch joins — no gain when all arrive together) ✓
+ PMAT-074:     216.4 (slot recycling — +10% heterogeneous traffic) ✓
+ PMAT-088a:    232.7 (iteration scheduler — +10.4%) ✓
+ PMAT-088c/d:  257.4 (batch recycling + multi-prompt — 84% of DP4A ceiling) ✓
DP4A ceiling:   306 tok/s (M=4, compute-bound)
vLLM AWQ:       598.0 (W4A16 tensor core GEMM, fundamentally different arch)
```

**All scheduling improvements exhausted.** Reaching vLLM-class aggregate requires W4A16 tensor core GEMM (Marlin-style kernel, PMAT-054B) — the DP4A ceiling is structural.

#### Post-Continuous Batching Analysis: Why c=4 Stalls at 216 tok/s (v2.23.0)

**PMAT-072/073/074 complete. Continuous batching (join, recycle, step-wise decode) delivered only +10% (197→216). The theoretical maximum at M=4 is ~550 tok/s (4 × 138). Three root causes explain the 2.55× shortfall.**

##### Finding 1: Per-slot decode degrades 2.67× at M=4 (ITL 19.2ms vs 7.2ms)

**Five-Whys:**

1. Why is c=4 aggregate only 216 tok/s vs theoretical 550? → Per-slot decode is 52 tok/s (not 138). Aggregate = M × per-slot = 4 × 52 = 208.
2. Why 52 tok/s per slot? → ITL is 19.2ms (vs 7.2ms at M=1 = 2.67× slower per step). Weight reads should be amortized across M slots.
3. Why 2.67× per step? → Three additive factors: (a) no CUDA graphs (+~5ms kernel launch overhead), (b) 4× larger L2 working set (+12% from PMAT-058), (c) batched attention reads 4× KV entries.
4. Why no CUDA graphs? → `BATCHED_GRAPH=1` tested 25% slower (PMAT-056). Disabled by default. The capture overhead occurs per-batch, not amortized across decode steps.
5. Why 25% slower with graphs? → Graph capture forces cuBLAS workspace-free algorithms (same root cause as PMAT-059 prefill). But batched decode uses DP4A GEMV (custom PTX, not cuBLAS) — the 25% overhead needs re-investigation. Hypothesis: workspace buffer address instability between graph captures, not algorithm selection.

**Root cause:** Kernel launch overhead is significant at M=4. The batched decode path fires **654 kernel launches per step** (23 per layer × 28 layers + 10 final) without graph amortization, adding ~2.6ms (17% of 15.1ms ITL). L2 cache spreading (+12%) and batched attention scaling compound the regression. CUDA graph for M>1 was attempted (PMAT-088b) but FALSIFIED (H-CB11): graph replay is 3ms slower than eager due to frozen attention grid dimensions.

**Fix (PMAT-075) — IMPLEMENTED, FALSIFIED:** Infrastructure for stable graph reuse across batches is complete. Three changes: (1) `batched_cleanup` preserves workspace buffers (PAR-200 skip path sets batch_size=1 without reallocation), (2) `batched_cleanup` preserves KV caches and auxiliary pointer buffers (addresses stable for graph reuse), (3) `init_batched_kv_cache_gpu` skips auxiliary buffer reallocation when KV caches are preserved. Latent bug fixed: `init_prefill_workspace` now clears batched decode graphs on reallocation (previously only cleared M=1 decode graph). VRAM: FP16(2944)+KV(896)+Q4K(850)+WS(40) = 4730 MB fits 7.5 GB RTX 4060L.

**Result:** Graphs persist across batches (confirmed: "Reusing batched KV cache" messages, single capture per M value). But **graph replay is 2.8ms SLOWER than eager** (ITL 22.0ms vs 19.2ms at M=4). BATCHED_GRAPH=1 remains disabled by default.

**Falsification outcome:** BATCHED_GRAPH=1 gives ITL 22.0ms > 19.2ms → kernel launch overhead is NOT the dominant source of the M=4 decode regression. The 2.67× per-slot degradation is primarily from batched attention scaling (4× more KV entries per head) and L2 cache working set pressure, not kernel launch overhead. The ~280 launches per step contribute ~0.8ms (4% of ITL) vs the hypothesized ~5ms (28%). Graph overhead (5 synchronous H2D copies + graph dispatch) adds ~2.8ms that outweighs the launch savings. Root cause of graph overhead requires nsys profiling.

**Revised five-whys (Mar 12):** Why is batched decode 2.67× slower per slot? → Batched GEMV compute scales linearly with M (DP4A chains, H-CB10 CONFIRMED). Why does batched graph make it worse? → Attention kernel grid dimensions (proportional to seq_len) are frozen at capture time. Graph captured with dummy seq_lens=1 replays with wrong grid size for seq_lens=128+. Additionally, 654 CUDA graph nodes create management overhead exceeding the launch savings. Both HGEMM crossover (H-CB9) and batched CUDA graph (H-CB11) are FALSIFIED. Next: chunked prefill (Phase 3) and trueno GEMV kernel optimization (Phase 2b).

##### Finding 2: Dead slot compute waste

**Five-Whys:**

1. Why does recycling only help +10%? → Recycling prevents idle SLOTS, but done slots still consume GPU compute until recycled.
2. Why do done slots consume compute? → `batched_decode_step` embeds all M tokens (line 278-288), runs forward pass for full M, then distribute_tokens skips done slots.
3. Why not skip done slots? → Batched GEMV operates on a contiguous `[M × hidden_dim]` buffer. Removing a slot mid-buffer requires compaction or scatter-gather.
4. Why contiguous? → GPU kernels use `grid.y = M` with `slot_idx = blockIdx.y`. Holes in the M dimension would require either mask-based skip or buffer compaction per step.
5. Why not just compact? → Compacting would require: (a) reordering embed_buf, (b) remapping KV cache slot indices, (c) updating all per-slot state arrays. This is O(M × hidden_dim) memcpy per compaction — potentially more expensive than the wasted compute.

**Root cause:** Batched GEMV assumes dense M. Done slots get zero embeddings, producing wasted GEMV output and attention computation. At M=4 with 1 done slot, 25% of compute is wasted.

**Fix (PMAT-076): IMPLEMENTED.** Added `batched_done_mask` field to CudaExecutor, set from `BatchedDecodeState.done` before each forward pass. Attention kernels (batched, flash decode, graph replay) zero `seq_lens[i]` for done slots, triggering early-exit (zero KV iterations). GEMV still runs on dead slots (zero input → near-zero output, discarded). No regression: c=1=138.9, c=4=199.0 tok/s on uniform traffic. Heterogeneous batch (max_tokens 10+128) verified correct.

**Falsification:** Uniform probador traffic shows no ITL change (all slots finish simultaneously → no dead slots). Needs PMAT-077 heterogeneous traffic for measurement. Manual test with 2/4 dead slots showed correct behavior — quantitative impact pending.

##### Finding 3: Uniform traffic defeats recycling

**Five-Whys:**

1. Why does probador c=4 show no recycling benefit? → All 4 slots finish at the same gen_idx (128).
2. Why same gen_idx? → Same prompt + same max_tokens=128 → identical generation length. Temperature=0.7 sampling occasionally produces different EOS timing, but max_tokens is the binding constraint.
3. Why binding? → Probador uses a fixed prompt and fixed max_tokens for all requests. No variance in request characteristics.
4. Why is this a problem? → Slot recycling requires heterogeneous request completion times. When all finish simultaneously, there's nothing to recycle INTO — the batch ends and a new batch starts.
5. Why does this matter? → probador is the only benchmark tool. It cannot demonstrate recycling's production value, making it invisible in performance tracking.

**Root cause:** Benchmark blind spot. probador sends uniform traffic (same prompt, same max_tokens), which is pathological for continuous batching. Production traffic has variance in prompt length, max_tokens, and EOS timing — exactly the scenario recycling optimizes.

**Fix (PMAT-077):** Add heterogeneous traffic mode to probador. `probador llm load --max-tokens-distribution uniform:16,128` sends requests with max_tokens uniformly distributed between 16 and 128. This creates staggered slot completion, enabling recycling measurement. Alternative: use multiple probador prompts from a file (`--prompts-file`) with varying lengths.

**Falsification:** If heterogeneous mode (probador or curl test) shows ≥ 250 tok/s aggregate at c=8, recycling is confirmed valuable for real traffic. (Already validated: 216.4 tok/s with curl heterogeneous test, limited by per-slot ITL.)

**Updated trajectory (Mar 12, PMAT-088d complete):**
```
Continuous batching baseline:   216.4 aggregate tok/s (0.36x vLLM)
+ PMAT-088a (iter scheduler):   232.7 (+10.4%)
+ PMAT-088c (batch recycling):  256.3 (+24% from baseline)
+ PMAT-088d (multi-prompt):     257.4 (+0.4%, TTFT 60.7ms)
DP4A ceiling at M=4:            306 tok/s (84% reached)
vLLM AWQ (tensor core GEMM):   604.7 (fundamentally different arch)
```
**Remaining c=4 gap:** 257.4 vs 306 ceiling = 16% headroom. Sources:
(1) recycle stall (~16ms prefill blocks decode), (2) scheduling latency (~12ms reconnect+wait).
Chunked prefill (PMAT-088e) would help for long prompts but short 125-token prompts
already fit in one chunk. Kernel optimization (reduce compute/token 2.43→1.94ms) is the
remaining lever for aggregate throughput.

### Jetson Orin Root Cause Analysis (Updated Mar 12, 2026)

**Decode parity ACHIEVED (Mar 8, PMAT-040).** realizr 40.8 tok/s vs llama.cpp 36.1 tok/s
— **13% faster** after flash decode chunk_size fix + MAXN_SUPER power mode (1020 MHz GPU).
TTFT: 47.8ms vs 34.0ms (1.4x, PASS < 2x target). Jetson sm_87 cannot use FP8 prefill
(requires sm_89+), so prefill uses cuBLAS HGEMM from on-demand FP16 weight cache.

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

**Five-Whys: realizr 36.0ms/token decode vs llama.cpp 30.2ms/token (1.19x gap) — UPDATED Mar 8 post PMAT-040 flash decode fix**

1. Why was decode 1.19x slower? → BrickProfiler showed **LmHead 25.7%** of decode time (Q6K GEMV n=151,936).
2. Why is LmHead #1? → Single Q6K GEMV with n=151,936, k=1536. 10,948µs per call.
3. Why are FFN gate/down expensive? → 28 layers × ~370µs each. Q4K GEMV with k=8960.

**Resolution (PMAT-040, Mar 8)**: Flash Decode chunk_size 128→32 fix (sequences <128 got 1 chunk = zero split-K parallelism). **realizr now 13% FASTER than llama.cpp on Jetson**: 40.8 vs 36.1 tok/s, 24.5ms vs 27.7ms ITL (MAXN_SUPER 1020MHz, Mar 12). Decode gap CLOSED.

**Remaining gap: Prefill 1.4x** (TTFT 47.8ms vs 34.0ms, MAXN_SUPER 1020MHz). Jetson sm_87 has no FP8 E4M3 (requires sm_89+), so prefill uses cuBLAS HGEMM from on-demand FP16 weight cache — 3.5x more BW than llama.cpp's fused Q4K GEMM. TTFT within 2x target (PASS).

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

The first decode token triggers CUDA graph capture: `stream.begin_capture(Global)` → execute full transformer forward → `stream.end_capture()` → `graph.instantiate()`. For M=1 this records ~280 kernel launches; for batched M>1 this records **654 kernel launches** (23 per layer × 28 layers + 10 final: rmsnorm, 4× fused QKV DP4A, 3× bias, 2× RoPE, 2× KV scatter, attention, output proj, residual, rmsnorm, 3× fused gate+up DP4A, SwiGLU, down proj, residual per layer + output norm + LM head + argmax).

Subsequent tokens replay the captured graph:
```
1. H2D async: position_buf, seq_len_buf, graph_input_buf  (~20µs, same stream)
2. stream.launch_graph(graph_exec)                          (~10µs API overhead)
3. GPU executes 280 kernels                                 (~36ms at 27.8 tok/s)
4. gpu_argmax: two-pass block reduction                     (vocab=151936 → single u32)
5. D2H: 4 bytes (token ID)                                 (<1µs)
6. cache.advance()                                          (CPU, negligible)
```

**Why CUDA graphs matter (M=1):** Without graphs, 280 kernel launches × ~20µs each = 5.6ms overhead per token (16% of decode time). With graphs, one launch × ~10µs = 0.01ms — a 560× reduction in launch overhead. **For batched M>1:** 654 launches × ~4µs = 2.6ms overhead (17% of 15.1ms ITL). However, batched CUDA graph is FALSIFIED (H-CB11) — attention grid dimensions frozen at capture time make the graph produce wrong results for varying seq_lens.

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

## 5b. Continuous Batching Architecture Design (GH-141, PMAT-088)

### Problem Statement

Concurrency scaling is the only dimension below A (score 51, 34.1% efficiency at c=4).
Phases 1-3 (PMAT-072/073/074) implemented step-wise decode, mid-batch joins, and slot
recycling but delivered only +10% (197→216 tok/s). The bottleneck is **not** scheduling
overhead — it is per-slot decode degradation at M>1 caused by batched DP4A GEMV compute
scaling linearly with M (2.425ms per additional token). **PMAT-088b confirmed:** the DP4A
aggregate ceiling is 412 tok/s regardless of concurrency. Achieving 3x c=1 (455 tok/s)
requires W4A16 tensor core GEMM, not scheduling improvements.

vLLM achieves 88.9% efficiency at c=4 (598→604.7 tok/s) via three architectural innovations
that realizr currently lacks: (1) iteration-level scheduling with token budgets, (2) paged
KV cache, and (3) chunked prefill interleaving decode and prefill in the same forward pass.

### Phase 1 Results (PMAT-088a — Iteration Scheduler, Mar 11)

**Measured on yoga RTX 4060 Laptop, isolated serial, 60s, streaming, 1900 MHz locked.**

**Initial measurement (pre-bugfix):**

| Metric | Baseline (batch sched) | PMAT-088 initial | Delta |
|--------|----------------------|----------------------|-------|
| c=1 decode tok/s | 154.8 | 152.8 | -1.3% (noise) |
| c=1 TTFT P50 | 13.4ms | 20.6ms | +54% (waiting queue drain overhead) |
| c=1 ITL P50 | 7.2ms | 6.5ms | -9.7% |
| c=4 aggregate tok/s | 210.8 | 256.9 | +21.9% |
| c=4 decode/slot tok/s | 52.2 | 66.8 | +28.0% |
| c=4 TTFT P50 | 128.5ms | 80.6ms | -37.3% |
| c=4 ITL P50 | 19.2ms | 15.0ms | -21.9% |
| c=4 scaling efficiency | 34.1% | 42.0% | +8pp |

**Post-bugfix (PMAT-088 variable-M buffer fixes, final):**

The initial 256.9 aggregate included inflated counts from error-retry cycles caused by three
classes of buffer length mismatch when M changes between iterations (M=4→M=1→M=3). After
fixing all variable-M bugs (see below), the corrected numbers are:

| Metric | Baseline (batch sched) | PMAT-088 final | Delta |
|--------|----------------------|----------------|-------|
| c=1 decode tok/s | 154.8 | **151.8** | -1.9% (noise) |
| c=1 TTFT P50 | 13.4ms | 20.4ms | +52% (waiting queue drain) |
| c=1 ITL P50 | 6.5ms | 6.6ms | +1.5% (noise) |
| c=4 aggregate tok/s | 210.8 | **232.7** | **+10.4%** |
| c=4 decode/slot tok/s | 52.2 | 66.1 | +26.6% |
| c=4 TTFT P50 | 128.5ms | **81.6ms** | **-36.5%** |
| c=4 ITL P50 | 19.2ms | 15.1ms | -21.4% |
| c=4 scaling efficiency | 34.1% | **38.3%** | +4.2pp |

**Variable-M buffer bugs (PMAT-088 bugfix sweep):**

The iteration scheduler creates batches of varying M (1, 3, 4) unlike the old batch scheduler
which always waited for full M=4. This exposed three classes of buffer length mismatch in the
CUDA executor, all with the same root cause: `copy_from_host()` requires exact length match
but high-water-mark buffers retained the capacity from the largest M ever seen.

1. **Logits buffer** (M×vocab vs vocab): `prepare_capture_buffers` checked `is_none()` not
   size. M=4 decode resizes to 4×151936, then M=1 graph capture fails with "host 151936 vs
   device 607744". Fix: check `b.len() != vocab_size`, also `clear_decode_graph()` on realloc.

2. **Input/hidden buffer** (M×hidden vs previous M×hidden): Grow-only allocation (`<`) kept
   M=4 capacity; M=3 batch gets "host 4608 vs device 6144" (3×1536 vs 4×1536). Fix: exact-
   size reallocation (`!=`), logical size for `hidden_buf2_len`.

3. **KV ptr/seq_lens buffers** (M vs max_M): Auxiliary buffers (batched_seq_lens_gpu,
   batched_k_ptrs, batched_v_ptrs) allocated at high-water mark. Fix: `copy_from_host_at(0)`
   for sync copies, `from_raw_parts` exact-M views for async copies.

**Key finding — FALSIFIED H-CB4:** Sequential M=1 forward passes per slot were predicted to
eliminate the 2.55x M=4 penalty. In fact, the iteration scheduler does NOT run M=1 per slot —
it still uses the batched M=4 forward pass (the existing PMAT-072 infrastructure). The +10.4%
gain comes from **waiting queue integration** (requests join faster via waiting queue vs rx
channel polling) and improved scheduling decisions (waiting queue checked before rx channel,
reducing slot vacancy time). The batched GEMV weight amortization (828 MB read once for M=4
vs 4× for sequential M=1) means **sequential M=1 is strictly worse than batched M=4** for
decode: 4 × 6.5ms = 26ms vs 13.8ms batched. The original spec estimate of ~500 aggregate
from M=1 per-slot forward was incorrect.

**CUDA_BATCH_WINDOW_MS=5 vs 0:** No meaningful difference (256.3 vs 256.9 tok/s). The
iteration scheduler's waiting queue drain replaces the batch window function.

**Output tok min=1:** Both c=4 runs show `output_tokens_dist: [1, 128, 128, 128]` — the
min=1 is an end-of-window effect (request arrives in final second, gets truncated). Not a
correctness bug.

**Corrected architecture insight (PMAT-088b):** The path to higher aggregate throughput is
**increasing M** (more concurrent slots), not reducing per-step time. DP4A GEMV has a
fundamental compute ceiling: each additional token adds one DP4A accumulation chain per
weight row (~2.425ms per token). The aggregate ceiling is 1/2.425ms = **412 tok/s** regardless
of concurrency. HGEMM crossover was falsified (H-CB9) — FP16 3.5x BW penalty is not
compensated by tensor cores at M=4..8.

**Scaling analysis (PMAT-088b, Mar 11):**

| c | M | Aggregate tok/s | ITL P50 | TTFT P50 | vs Theoretical | vs c=1 |
|---|---|----------------|---------|----------|---------------|--------|
| 1 | 1 | 151.6 | 6.6ms | 20.4ms | — | 1.00x |
| 4 | 4 | 234.0 | 15.1ms | 82.1ms | 76% of 306 | 1.54x |
| 8 | 8 | **306.5** | 21.9ms | 163.2ms | **87% of 352** | **2.02x** |
| ∞ | ∞ | ceiling | — | — | 412 tok/s | 2.72x |

Theoretical ceiling per M: `M / (2.56 + M × 2.425 + 0.8)` ms/step, where 2.56ms = weight BW
(amortized), 2.425ms = DP4A compute per token, 2.6ms = kernel launch overhead (654 launches
× ~4µs, revised from 0.8ms — see H-CB11). Efficiency
improves with larger M because BW and launch costs amortize.

**Comparative analysis (Mar 11) — llama.cpp uses DP4A too, but faster:**

llama.cpp also uses DP4A Q4K GEMV for batched decode (M≤8, confirmed via source analysis of
`mmvq.cuh` MMVQ_MAX_BATCH_SIZE=8). Both runtimes use the same kernel architecture. The gap
is per-kernel efficiency and scheduling overhead:

| | realizr | llama.cpp | Gap | Source |
|--|---------|-----------|-----|--------|
| c=1 decode | 151.6 | 159.8 | 0.95x | Kernel efficiency |
| c=4 aggregate | 234.0 | **353.0** | **0.66x** | Kernel + scheduling |
| c=4 ITL | 15.1ms | 11.3ms | 1.34x | Per-step compute |
| c=4 TTFT | 82.1ms | 18.6ms | 4.4x | Prefill overhead |
| c=8 aggregate | 306.5 | **414.1** | **0.74x** | Kernel + scheduling |
| c=8 ITL | 21.9ms | 19.3ms | 1.13x | Per-step compute |

**Two sources of c=4 gap (0.76x llama.cpp, updated Mar 12 PMAT-088d):**
1. **Per-step compute (1.34x)**: llama.cpp's MMVQ DP4A kernel is ~25% faster per token
   (compute/token: 1.94ms vs 2.43ms). Also: llama.cpp uses CUDA graphs (`USE_GRAPHS=1`)
   and flash attention, reducing launch overhead.
2. **Scheduling overhead (~8%)**: PMAT-088c/d recycling reduced overhead from 14% to ~8%.
   Recycle prefill: 16.3ms/slot (was 82ms). TTFT P50: 60.7ms (was 82.1ms). Remaining
   overhead from: reconnect wait (~5ms), decode step wait (~7.5ms avg), recycle prefill (16ms).

**Phase status (updated Mar 12, PMAT-088d):**
- ~~Phase 2a: CUDA graph for M>1~~ → **FALSIFIED** (H-CB11). Graph replay 3ms SLOWER than eager.
- ~~Phase 2b: Multi-prompt recycle~~ → **DONE** (PMAT-088d). TTFT 63.6→60.7ms (-4.6%).
- Phase 3: Chunked prefill (PMAT-088e) → interleave prefill chunks with decode steps
- Phase 4: Paged KV cache (PMAT-088f) → dynamic KV allocation, enable c>4

**Note:** vLLM AWQ (604.7 at c=4) uses INT4→FP16 dequant + tensor core GEMM — fundamentally
different from DP4A GEMV. That's a different kernel architecture with 2x less compute per token.

### Literature Foundation

The continuous batching architecture draws from four foundational systems:

**Orca** (Yu et al., OSDI 2022) — Introduced **iteration-level scheduling**: after each
autoregressive step, finished requests leave and new requests join. Traditional request-level
scheduling forces all requests in a batch to run to completion before admitting new work.
Orca's selective batching applies batching only to compute-uniform ops (linear layers) while
handling attention per-sequence. Result: 36.9x throughput over FasterTransformer. realizr's
PMAT-072 step-wise decode is a partial implementation of this — lock release between steps
enables join/leave, but without a token budget scheduler.

**PagedAttention / vLLM** (Kwon et al., SOSP 2023) — Borrowed OS virtual memory paging for
KV cache. Fixed-size blocks (typically 16 tokens) allocated on-demand via block tables,
eliminating the 60-80% memory waste from pre-reserving max_seq_len per request. Physical
blocks are non-contiguous; a per-request block table maps logical→physical (like page tables).
Copy-on-Write enables block sharing for parallel sampling. Result: 2-4x throughput over
FasterTransformer/Orca, <4% memory waste. vLLM's scheduler is ~2100 LOC with two priority
queues (waiting, running), token budget system (default 2048 tokens/iteration), and
preemption logic for overcommitted KV cache.

**Sarathi-Serve** (Agrawal et al., OSDI 2024) — Solved the **prefill-decode interference**
problem. Long prefills block ongoing decodes for seconds (generation stalls). Sarathi splits
prefills into equal-sized chunks (256-512 tokens) and co-schedules them with decode tokens in
each forward pass. The algorithm is decode-maximal: (1) pack all ongoing decode requests
first, (2) fill remaining token budget with one prefill chunk. Prefill is compute-bound
(GEMM); decode is memory-bound (GEMV) — they overlap efficiently on GPU because prefill
saturates tensor cores while decode saturates memory bandwidth. Result: 2.6-6.9x higher
serving capacity. Critical detail: chunk sizes must align with GPU tile dimensions (257
tokens instead of 256 can increase prefill time by 32% due to wasted compute).

**Splitwise / DistServe** (Patel et al., ISCA 2024; Zhong et al., OSDI 2024) — Demonstrated
that prefill and decode have fundamentally different hardware requirements. Prefill is
compute-bound (arithmetic intensity scales with sequence length); decode is memory-bound
(arithmetic intensity ~1). This validates realizr's existing separate code paths (HGEMM
prefill, DP4A GEMV decode) but argues for distinct scheduling policies per phase. Not
directly applicable to single-GPU deployments, but the phase characterization is critical.

### Architecture Design

#### Current State (realizr v2.28.0)

```
HTTP Handler → mpsc channel → BatchScheduler → model.write() → generate_batched_streaming()
                                                                  ├─ batched_setup_and_prefill()  (all slots)
                                                                  ├─ batched_decode_step() loop   (M tokens/step)
                                                                  └─ batched_cleanup()
```

**Limitations:**
- Scheduler is 185 LOC wrapping a generate function (vLLM: ~2100 LOC dedicated scheduler)
- Batched GEMV runs all M slots through full transformer — O(M) attention KV reads per slot
- No token budget — batch size equals concurrent requests (no chunked prefill)
- Prefill blocks all decode until complete (generation stalls at c>1)

#### Target State (GH-141)

```
HTTP Handler → RequestQueue (lock-free)
                    ↓
              IterationScheduler (token budget, priority queues)
                    ├─ running_queue: ongoing decode requests
                    ├─ waiting_queue: pending prefill requests
                    ├─ token_budget: max tokens per forward pass (default 2048)
                    └─ schedule() → SchedulerOutput {
                         scheduled_decode:  Vec<(slot_id, 1 token)>
                         scheduled_prefill: Option<(slot_id, chunk_tokens)>
                         preempted:         Vec<slot_id>
                       }
                    ↓
              ModelExecutor (async, non-blocking)
                    ├─ PagedKVCache: block_pool + per-request block_tables
                    ├─ forward_mixed(decode_tokens, prefill_chunk) → logits
                    └─ sample_tokens(logits) → next_token_ids
                    ↓
              SchedulerUpdate (process outputs, recycle slots, admit new)
```

#### Component 1: Iteration-Level Scheduler

**Scheduling algorithm (per iteration):**

```
fn schedule(token_budget: usize) -> SchedulerOutput:
    remaining = token_budget
    output = SchedulerOutput::new()

    # 1. Decode-maximal: always schedule ALL running decode requests first
    for req in running_queue:
        if req.needs_decode():
            output.scheduled_decode.push((req.slot_id, 1))
            remaining -= 1

    # 2. Fill remaining budget with ONE prefill chunk
    if remaining > 0 and not waiting_queue.is_empty():
        req = waiting_queue.peek()
        chunk_size = min(remaining, req.remaining_prefill_tokens())
        chunk_size = align_to_tile(chunk_size, 256)  # tile quantization
        output.scheduled_prefill = Some((req.slot_id, chunk_size))
        remaining -= chunk_size

        if req.remaining_prefill_tokens() == 0:
            waiting_queue.pop()
            running_queue.push(req)

    # 3. Preempt if KV cache exhausted
    if kv_cache.available_blocks() < needed_blocks(output):
        preempt_lowest_priority(running_queue, output)

    output
```

**Token budget sizing (RTX 4060 Laptop, 8GB VRAM):**
- Model weights (Q4K): ~850 MB
- KV cache budget: ~2 GB (128 blocks × 16 tokens × 1536 hidden × 2 KV × 2 bytes)
- Max concurrent sequences: 8-16 (at 4096 max_seq_len)
- Default token_budget: 512 (conservative for 8GB — Sarathi recommends profiling)

#### Component 2: Paged KV Cache

**Current KV cache:** Pre-allocated contiguous buffer per slot, max_seq_len × hidden_dim.
Wastes memory for short sequences. Cannot share blocks across requests.

**Paged KV cache design:**

```
BlockPool:
    block_size: 16 tokens
    num_blocks: 128 (2 GB / (16 * 1536 * 2 * 2 bytes) ≈ 128)
    free_list:  DoublyLinkedList<BlockId>
    ref_count:  Vec<u32>  # for prefix caching (future)

KVCacheManager:
    block_tables: HashMap<RequestId, Vec<BlockId>>  # logical→physical mapping

    fn allocate_slots(request_id, num_new_tokens) -> Option<Vec<BlockId>>:
        needed = ceil(num_new_tokens / block_size)
        if free_list.len() < needed: return None  # trigger preemption
        blocks = free_list.pop_n(needed)
        block_tables[request_id].extend(blocks)
        Some(blocks)

    fn free_request(request_id):
        for block in block_tables.remove(request_id):
            ref_count[block] -= 1
            if ref_count[block] == 0:
                free_list.push(block)
```

**Block table indirection in attention kernel:**
```
// Current: contiguous KV access
kv_ptr = kv_cache[slot_id] + seq_pos * hidden_dim

// Paged: indirect via block table
block_idx = seq_pos / BLOCK_SIZE
block_offset = seq_pos % BLOCK_SIZE
physical_block = block_table[request_id][block_idx]
kv_ptr = block_pool[physical_block] + block_offset * hidden_dim
```

Memory fragmentation: <4% waste (only in last partially-filled block per request).
Current waste at c=4 with max_seq_len=4096: 75%+ (most sequences use <1024 tokens).

#### Component 3: Chunked Prefill

**Problem:** realizr c=4 TTFT is 128.5ms because prefill runs all 4 requests serially
before any decode begins. During prefill, all running decode requests stall.

**Chunked prefill algorithm:**
1. Split incoming prefill into chunks of 256 tokens (tile-aligned for sm_89)
2. Each iteration: run 1 prefill chunk + all pending decode tokens
3. Prefill GEMM (compute-bound) and decode GEMV (memory-bound) overlap on GPU
4. Generation continues uninterrupted — no stalls

**Expected impact on TTFT at c=4:**
```
Current:   4 × 46.4ms = ~186ms total prefill (sequential)
Chunked:   125-token prompt = 1 chunk. Each chunk shares iteration with decode.
           TTFT ≈ 46.4ms (same as c=1) — decode tokens piggyback at near-zero cost
```

**Expected impact on aggregate throughput at c=4:**
```
Before PMAT-088: 210.8 aggregate (34.1% efficiency)
After PMAT-088:  232.7 aggregate (38.3% efficiency)  ← MEASURED (post-bugfix)
Target:          ~460 aggregate (≥74% efficiency, A grade)
```

**~~Critical insight~~ CORRECTED (Mar 11):** The original hypothesis that M=1 per-slot
forward passes would eliminate the 2.55x penalty was **falsified**. Sequential M=1 is
SLOWER than batched M=4 because GEMV weights (828 MB) must be read from memory for each
slot independently — 4 × 828 MB = 3.3 GB vs 828 MB once for batched M=4.

**ROOT CAUSE CORRECTED (PMAT-088b analysis, Mar 11):** The per-slot degradation at M>1 comes
from **GEMV compute scaling**, NOT attention KV reads. Profiling confirmed:
- Attention KV reads at M=4: 14 MB total (2.8% of weight BW) — negligible
- GEMV DP4A compute: 4× activation loads + 4× DP4A chains per weight row = 7.3ms extra
- M=1 decode step: ~5.0ms (memory-bound, Q4K GEMV + CUDA graph)
- M=4 decode step: ~13.3ms = 2.56ms BW + 8.1ms compute + 2.6ms launch overhead (654 launches)
- Ratio: 2.66x per step (DP4A becomes compute-bound at M>1)

**Corrected path forward (PMAT-088b, updated Mar 11):** The batched Q4K GEMV reads weights
ONCE for all M vectors but runs M× independent DP4A accumulation chains. At M=4, the kernel
transitions from memory-bound to compute-bound.

**HGEMM crossover FALSIFIED (H-CB9):** Three variants tested — none beat DP4A. FP16's 3.5×
BW penalty not compensated by tensor cores at M=4..8 on RTX 4060L.

**llama.cpp comparative analysis:** llama.cpp also uses DP4A Q4K GEMV (MMVQ at M≤8), but
achieves 11.3ms ITL at M=4 vs our 15.1ms (1.34× slower). Two root causes:
(a) **CUDA graphs** (USE_GRAPHS=1): llama.cpp replays a single graph; we launch 654 kernels
    eagerly (~2.6ms overhead). Batched graph FALSIFIED (H-CB11) — 3ms slower than eager.
(b) **Kernel efficiency**: llama.cpp MMVQ compute/token ~1.94ms vs our 2.43ms (~25% faster).
    Their GEMV is more heavily optimized (years of community work). Closing this requires
    trueno GEMV kernel optimization.
(c) **Prefill interleaving**: llama.cpp processes prefill with minimal decode stall (TTFT
    18.6ms at c=4). Our sequential prefill stalls decode for 82ms per slot (328ms/round
    at c=4 = 14% overhead vs llama.cpp's 5%). Chunked prefill fixes this.

### Implementation Plan

| Phase | PMAT | Deliverable | Effort | Measured/Expected Impact |
|-------|------|-------------|--------|------------------------|
| **P1: Iteration scheduler** | PMAT-088a | Waiting queue + decode-maximal scheduling. Replace `BatchScheduler` with `IterationScheduler`. Keep contiguous KV cache. Variable-M bugfixes (3 classes). | 1 week | **DONE: +10.4% aggregate (210.8→232.7), 38.3% efficiency** |
| **P2: M>1 CUDA graph + launch reduction** | PMAT-088b | HGEMM crossover **FALSIFIED** (H-CB9). Batched CUDA graph **FALSIFIED** (H-CB11): 654 kernel launches/step = 2.6ms overhead, but graph replay 3ms slower than eager. Root cause: attention kernel grid dimensions frozen at capture time (dummy seq_lens=1), positions/seq_lens cannot be runtime parameters in graph. Both interventions exhausted — Phase 2 closed. | 1 week | **FALSIFIED: graph adds 3ms overhead vs saving 2.6ms** |
| **P2b: Multi-prompt recycle** | PMAT-088d | `recycle_slots_batch` via `prefill_multi_prompt` with `slot_indices`. Eliminates `force_workspace_reinit` overhead. | 1 day | **DONE: TTFT 63.6→60.7ms (-4.6%), N=1 recycle 17.3→16.3ms (-6%)** |
| **P3: Chunked prefill** | PMAT-088e | Prefill chunking with tile alignment. Mixed prefill+decode forward pass. | 1 week | Expected: TTFT at c=4 ≈ c=1, no decode stalls for long prompts |
| **P4: Paged KV cache** | PMAT-088f | BlockPool + KVCacheManager + block table indirection. Enable preemption. | 2 weeks | Expected: <4% memory waste, enable c>4 scaling |

**Phase 2 status: CLOSED — both interventions falsified.**
- H-CB9: HGEMM crossover falsified. Three variants tested — none beat DP4A (best: 260.5 vs 261.5).
- H-CB11: Batched CUDA graph falsified. 654 launches/step = 2.6ms overhead (revised from 0.8ms).
  Graph replay is 3ms SLOWER than eager despite 5 async H2D fixes. Root causes:
  (1) Attention grid dimensions frozen at capture (dummy seq_lens=1 → wrong grid at replay).
  (2) RoPE captured with dummy positions [0,1,...,M-1] → wrong rotation angles at replay.
  (3) HGEMM guard (`!is_capturing`) forces DP4A during capture even when HGEMM active.
  (4) 654-node graph management overhead exceeds launch savings.
  Fundamental fix: refactor attention/RoPE kernels to read seq_lens/positions from device
  buffers (not launch parameters) so graph grid dimensions are position-independent.
  This is a multi-week refactor with uncertain payoff (~2.6ms best case, 17% of ITL).
**Phase 3 (chunked prefill) is now the highest-impact next step** — reduces c=4 scheduling overhead from 14% to ~5%.
**FlashAttention-2 de-prioritized**: attention reads are 14 MB vs 491 MB weights (<3% impact).

### Falsification Conditions

| ID | Hypothesis | Prediction | Result |
|----|-----------|------------|--------|
| H-CB4 | Iteration scheduling eliminates M=4 penalty | c=4 per-slot ITL ≤ 1.2x c=1 ITL | **FALSIFIED.** ITL 15.1ms / 6.6ms = 2.29x. Batched GEMV weight amortization means sequential M=1 is worse, not better. Root cause is attention KV scaling, not scheduling. |
| H-CB5 | Chunked prefill eliminates generation stalls | c=4 TTFT ≤ 1.5x c=1 TTFT | Pending (Phase 3) |
| H-CB6 | Paged KV reduces memory waste | Memory utilization > 90% at c=8 | Pending (Phase 4) |
| H-CB7 | Per-slot M=1 matches c=1 throughput | Aggregate ≥ 0.8 × (M × c=1_tok/s) | **FALSIFIED.** 232.7 / 607.2 = 38.3% < 60% threshold. Weight BW amortization in batched GEMV is essential — M=1 per slot reads 4× more weight data. |
| H-CB8 | Waiting queue integration improves scheduling | c=4 aggregate > baseline + 10% | **CONFIRMED.** 232.7 / 210.8 = +10.4%. Waiting queue drain reduces slot vacancy. Initial 256.9 (+21.9%) was inflated by error-retry cycles before variable-M bugfixes. |
| H-CB9 | HGEMM crossover at M>1 beats Q4K DP4A | M=4 per-step time < 11ms (vs 13.3ms Q4K) | **FALSIFIED.** Three variants tested at 1900 MHz, c=4: (1) Full HGEMM: 256.0 aggregate, 13.6ms ITL (−2.1%). (2) Hybrid (fused QKV/gate+up DP4A, HGEMM output/down): 260.5 aggregate, 13.7ms ITL (−0.4%). (3) Baseline DP4A: 261.5 aggregate, 13.4ms ITL. FP16 weights read 3.5× more BW (2 B/elem vs Q4K 0.5625 B/elem) — tensor core compute savings do NOT compensate on RTX 4060L at M=4. PMAT-062 confound resolved: re-enabling fused gate+up does not change the conclusion. |
| H-CB10 | M=4 penalty is from GEMV compute, not attention | Attention ≤ 5% of M=4 forward time | **CONFIRMED.** Attention KV reads = 14 MB / 491 MB weight BW = 2.8%. M=4 step 13.3ms decomposed: 2.56ms BW + 8.1ms compute + 2.6ms launches. |
| H-CB11 | Batched CUDA graph saves 2.6ms launch overhead | M>1 graph replay ITL ≤ eager ITL − 1ms | **FALSIFIED.** Graph replay is 3ms SLOWER than eager (18.1ms vs 15.1ms ITL). 654 kernel launches/step = 2.6ms theoretical savings. Root causes: (1) Attention grid dims frozen at capture with dummy seq_lens=1 (real replay has seq_lens=128+). (2) RoPE positions frozen at [0,1,...,M-1]. (3) 654-node graph management overhead. Fix requires position-independent kernel grid designs — multi-week refactor with uncertain payoff. |

### Academic References (added)

- [Orca (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al. Iteration-level scheduling.
- [Sarathi (2023)](https://arxiv.org/abs/2308.16369) — Agrawal et al. Chunked prefill + decode piggybacking.
- [Sarathi-Serve (OSDI 2024)](https://arxiv.org/abs/2403.02310) — Agrawal et al. Decode-maximal batching, stall-free scheduling.
- [DistServe (OSDI 2024)](https://arxiv.org/abs/2401.09670) — Zhong et al. Disaggregated prefill/decode, goodput optimization.
- [DeepSpeed-FastGen (2024)](https://arxiv.org/abs/2401.08671) — Holmes et al. Dynamic SplitFuse, consistent forward size.
- [FlashInfer (MLSys 2025, Best Paper)](https://arxiv.org/abs/2501.01005) — Block-sparse attention, JIT compilation.

---

## 6. Optimization Roadmap

### Tier Summary (Updated Mar 12 2026 — decode parity achieved, c=4 at DP4A ceiling)

| Tier | Items | Status |
|------|-------|--------|
| T0: Decode parity | Fixes 1-6, GH-173/176, PMAT-040 (flash decode) | ✅ 0.96x llama.cpp |
| T0: Prefill parity | PMAT-023/024/026, FP8 pipeline (PMAT-053b→086) | ✅ 1.33x llama.cpp (PASS < 2x) |
| T0: Continuous batching | PMAT-072→074, 088a-d (iter scheduler, recycling) | ✅ 257.4 aggregate (84% of ceiling) |
| **T1: W4A16 tensor core** | **Marlin-style INT4→FP16 GEMM** | **Planned — breaks DP4A ceiling** |
| T1: Chunked prefill | Interleave prefill with decode | Planned — reduces c=4 TTFT |
| T2: GEMV optimization | Q4K dequant instruction reduction (trueno) | Planned — closes 34% per-step gap |
| T2: SageAttention INT8 | INT8 attention for long context | Planned |
| T3: EAGLE speculative | Draft-then-verify 2-3x | Planned |

### Priority Matrix

| PMAT | Optimization | Impact | Status |
|------|--------------|--------|--------|
| PMAT-023 | Batched prefill default | 9x TTFT | ✅ DONE |
| PMAT-024/026 | cuBLAS GEMM prefill (Q4K+Q6K) | 1.95x prefill | ✅ DONE |
| PMAT-053b→086 | FP8 E4M3 prefill pipeline | TTFT 46→13.4ms | ✅ DONE |
| PMAT-019/022 | Q6K MWV GEMV + default | +22% Jetson decode | ✅ DONE |
| PMAT-028 | LmHead Q6K shared mem Q8 cache | Negligible on 4060L | ✅ DONE |
| GH-176 | Half-warp DP4A Q4K (16 thr/SB) | +66.5% decode | ✅ DONE |
| PMAT-040 | Flash decode chunk_size 128→32 | +55% 4090, +10% Jetson | ✅ DONE |
| PMAT-072→074 | Continuous batching (3 phases) | +10% heterogeneous | ✅ DONE |
| PMAT-088a-d | Iteration scheduler + batch recycling | 210→257 aggregate | ✅ DONE |
| PMAT-087 | Clock correction 1500→1900 MHz | +11.7% decode | ✅ DONE |
| PMAT-070 | CORRECTNESS-013 (stream 0 race) | Correctness fix | ✅ Fixed |
| PMAT-029 | Q4K dequant instruction reduction | 108→93 insn/SB (no decode impact) | ✅ DONE (memory-bound, no throughput gain) |
| PMAT-089 | Q4K block size + warp reduction | Both FALSIFIED (no impact) | ✅ DONE (4 warps: -2% regpressure; parallel reduce: 0%) |
| PMAT-090 | FP8 cuBLASLt batched decode (M≥2) | +4.3% c=4, **+33% c=8** | ✅ DONE (breaks DP4A compute ceiling at high M) |
| PMAT-091 | W4A16 coalesced WMMA GEMM | **FALSIFIED** (3.5x slower) | Custom PTX WMMA underperforms FP8/DP4A — needs double-buffer, k-tiling |
| PMAT-092 | Fused batched residual+RMSNorm | **FALSIFIED** (-5% c=4) | Restricts CTA parallelism (1,M) vs (6,M); residual fits L2, extra pass free |
| **PMAT-054B** | **W4A16 Marlin-style GEMM** | **2-3x c=4 aggregate** | **Planned** |
| PMAT-008 | SageAttention INT8 | 2-3x attention | Planned |
| PMAT-009 | EAGLE speculative decoding | 2-3x | Planned |
| PMAT-010 | Marlin-style GPTQ kernel | 2.6x | Planned |

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
| *Decode gap (pre-040)* | *1.19x (27.8 vs 33.1 tok/s)* | *probador --stream Mar 6, GH-176 HW DP4A* |
| *Prefill gap (pre-HGEMM)* | *25x (1045 vs 41ms TTFT)* | *probador --stream Mar 6, GH-176 + cuBLAS* |
| **Jetson prefill throughput** | **481 tok/s** (vs 676 llama.cpp, **1.4x**) | HGEMM on-demand, MAXN_SUPER, Mar 12 |
| **BW utilization (realizr)** | 35.2% of 67 GB/s | HW DP4A, Mar 6 |
| **BW utilization (llama.cpp)** | 40.9% of 67 GB/s | calculated |
| **LmHead % of decode** | 25.7% (10,948µs per call) | BrickProfiler Immediate sync, Jetson, Mar 6 |
| **FFN (gate+down) % of decode** | 48.5% | BrickProfiler Immediate sync, Jetson, Mar 6 |
| **4090 decode (PMAT-040)** | **1.06x** (411.7 vs 436.9 tok/s, avg_tok=128) | isolated c=1, 60s stream, Mar 8 |
| **Jetson decode (PMAT-040+078)** | **0.88x** (40.8 vs 36.1 tok/s) — **realizr 13% FASTER** | isolated c=1, 60s stream, MAXN_SUPER, Mar 12 |
| *4090 decode gap (pre-040)* | *1.64x short, 1.92x long* | *GpuProfile auto, serial c=1, Mar 6-7* |
| **4090 TTFT P50** | 58.8 vs 5.8 ms (10.1x) | prefill-dominated, HGEMM prefill |
| **4090 prefill tok/s** | 1734 vs 17620 (10.2x) | FP16 reads vs Q4K fused GEMM |
| **Jetson prefill tok/s** | 481 vs 676 (1.4x) | HGEMM on-demand, MAXN_SUPER, Mar 12 |
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
| H-CB1 | Batched decode correctness | `\|batched(r,c) - single(r,1)\| < 1e-3` | ✅ **FIXED (commit 6f75ec3, stream 0 race)** |
| H-CB2 | No frozen slots | Slots 1..M produce distinct tokens per step | ✅ **FIXED (commit 6f75ec3)** |
| H-CB3 | KV cache populated for all slots | `batched_kv_lengths[i] == prefill_len ∀i` | ✅ **Verified (200/200 deterministic)** |

### Verification Matrix

| Section | Tests | Passing |
|---------|-------|---------|
| A: GQA Fix | 3 | ✅ 3/3 |
| B: SwiGLU Fusion | 3 | ✅ 3/3 |
| C: Attention Quant | 3 | Pending |
| D: Launch Overhead | 3 | Pending |
| E: APR GPU Regression | 3 | Pending |
| F: Batched Decode Correctness | 3 | ✅ **3/3 FIXED (6f75ec3)** |

### F: Batched Decode Correctness (CORRECTNESS-013) — FIXED

**Defect (FIXED, commit 6f75ec3):** c=4 batched decode produced frozen slots. Root cause: stream 0 / non-blocking stream race in prefill H2D copies — `GpuBuffer::from_host` uses `cuMemcpyHtoD` (stream 0) while kernels launch on `CU_STREAM_NON_BLOCKING`. Fix: replace `from_host`/`copy_from_host` with `copy_from_host_async` on `self.stream` in all prefill paths. Verified: 200/200 deterministic.

**CORRECTNESS-014 (fixed):** CUDA context corruption after 3-4 requests — `CUDA_ERROR_ILLEGAL_ADDRESS` during graph replay. Root cause: `init_prefill_workspace` reallocated workspace buffers (longer prompt exceeds `buffer_capacity`), but decode graph was NOT cleared — stale pointers. Fix: clear `decode_graph` in `init_prefill_workspace` when reallocating.

**Five-Whys Root Cause Analysis:**

| Why | Finding | Evidence |
|-----|---------|----------|
| **Why-1** | c=4 requests produce wrong output | `probador llm load --concurrency 4` shows Output tok min=0 |
| **Why-2** | Identical prompts produce different tokens across slots | Server log: `token_ids=[21338, 21338, 304, 16]` — 4 identical prompts, 3 different outputs |
| **Why-3** | First batch correct, subsequent batches corrupted | Batch 1: `[21338, 21338]` (correct). Batch 2: `[323, 16]` (wrong) |
| **Why-4** | `PMAT-058: Freed batched KV caches to reclaim VRAM for FP16 rebuild` between batches | KV cache reinitialization produces corrupted state |
| **Why-5** | `correctness_under_batching` obligation has **NO implementation** | `pv audit continuous-batching-v1.yaml --binding realizar/binding.yaml` → BIND-003 |

**Provable contract reference:** `continuous-batching-v1.yaml` equation `correctness_under_batching`
**Falsification test:** FALSIFY-CB-006 ("Same prompt at c=1 and c=4 produces equivalent output")
**Binding status:** NOT IMPLEMENTED (BIND-003) — no wired test in realizr

**Instrumentation plan:**

| Instrument | Purpose | Command |
|-----------|---------|---------|
| `probador llm test` c=1 vs c=4 | Token-level correctness comparison | `probador llm test --url ... --concurrency 1` then `--concurrency 4` |
| `X-Trace-Level: brick` | Per-op tensor timing per request | `curl -H "X-Trace-Level: brick" ...` |
| `PMAT051_TRACE=1` | Per-layer QKV/Attn/FFN timing during prefill | Env var on `apr serve` |
| `PREFILL_DETAIL_TRACE=1` | Per-HGEMM M/N/K dimensions + timing | Env var on `apr serve` |
| `pv probar` | Generate wired falsification tests | `pv probar continuous-batching-v1.yaml` |
| `nsys` kernel timeline | GPU kernel ordering for scatter/attention | `make nsys-gpu` |
| `pmat query` | Semantic search for `batched_kv_lengths` mutations | `pmat query "batched_kv_lengths" --include-source` |

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
| PMAT-019 | GH #118 | Q6K MWV GEMV kernel (31.9% GPU time) | ✅ Completed (+22% on Jetson) |
| PMAT-020 | — | Jetson Orin load test migration | ✅ Completed |
| PMAT-021 | GH #121 | DP4A Q4K default on Orin sm_87 | ✅ Completed (+13%) |
| PMAT-022 | GH #118 | Q6K MWV GEMV default (was 442µs single-warp) | ✅ Done (MWV default, Refs #118) |
| PMAT-023 | — | Batched prefill default | ✅ Completed (TTFT 7.2s→816ms, 9x) |
| PMAT-024 | — | Prefill GEMM kernel (cuBLAS FP8) | ✅ Completed (TTFT 13.4ms, 1.33x llama.cpp) |
| PMAT-025 | GH-176 | `.maxnreg 255` PTX directive (no impact — 34 regs) | ✅ Done (no perf change) |
| PMAT-026 | GH-176 | Half-warp DP4A Q4K GEMV (16 thr/SB, 7.0 insn/val) | ✅ Done (+66.5%) |
| PMAT-027 | GH-176 | BrickProfiler Immediate sync (real GPU timing in cbtop) | ✅ Done |
| PMAT-028 | — | LmHead Q6K GEMV optimization (25.7% of decode on Orin) | ✅ Completed (Q8 smem cache) |
| PMAT-029 | — | Q4K dequant instruction reduction (constant hoisting) | ✅ Done (no throughput impact — memory-bound) |
| PMAT-070 | CORRECTNESS-013 | Batched decode frozen slots (stream 0 race) | ✅ Fixed (commit 6f75ec3) |
| PMAT-071 | — | Wire FALSIFY-CB-006 (`probador llm test` c=1 vs c=4) | ✅ Completed |
| PMAT-085 | — | Batch window + deferred graph clear (TTFT 21→15ms) | ✅ Completed |
| PMAT-086 | — | cuBLASLt descriptor caching + non-blocking drain (TTFT 15→14ms) | ✅ Completed |
| PMAT-087 | — | Clock correction 1500→1900 MHz (yoga baselines) | ✅ Completed |
| PMAT-088 | — | Iteration scheduler + batch recycling (c=4 aggregate) | ✅ Completed |

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
| Continuous Batching Contract | `../provable-contracts/contracts/continuous-batching-v1.yaml` | Batched decode correctness (FALSIFY-CB-006) |
| KV Cache Equivalence Contract | `../provable-contracts/contracts/kv-cache-equivalence-v1.yaml` | Batched-to-serial KV parity |
| GPU Decode Profiling Contract | `../provable-contracts/contracts/gpu-decode-profiling-v1.yaml` | Wall coverage, sync, brick ordering |
| Realizr Binding Registry | `../provable-contracts/contracts/realizar/binding.yaml` | 27/33 bindings (82%), 3 NOT IMPLEMENTED |

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
36. [Orca (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/yu) — Yu et al. Iteration-level scheduling.
37. [SARATHI (2023)](https://arxiv.org/abs/2308.16369) — Agrawal et al. Chunked prefill piggybacking.
38. [DistServe (OSDI 2024)](https://arxiv.org/abs/2401.09670) — Zhong et al. Disaggregated prefill/decode.
39. [DeepSpeed-FastGen (2024)](https://arxiv.org/abs/2401.08671) — Holmes et al. Dynamic SplitFuse.
40. [FlashInfer (MLSys 2025)](https://arxiv.org/abs/2501.01005) — Block-sparse paged attention, JIT compilation.

### Methodology

41. [The Logic of Scientific Discovery](https://www.routledge.com/9780415278447) — Popper, 1959
42. [Scientific Benchmarking (SC15)](https://doi.org/10.1145/2807591.2807644) — Hoefler & Belli
43. [Statistically Rigorous Java Evaluation (OOPSLA 2007)](https://doi.org/10.1145/1297027.1297033) — Georges et al.
44. [The Art of Computer Systems Performance Analysis](https://www.wiley.com/en-us/9780471503361) — Jain, 1991
45. [The Toyota Way](https://www.mhprofessional.com/9780071392310-usa-the-toyota-way) — Liker, 2004

---

## 14. Revision History

| Version | Date | Changes |
|---------|------|---------|
| 2.48.0 | 2026-03-12 | **PMAT-092 FALSIFIED: Fused batched residual+RMSNorm 5% slower.** Hypothesis: fusing `batched_residual_add_into` + `batched_rmsnorm_ptr_into` into single `BatchedFusedResidualRmsNormKernel` saves 28 launches/step, improving c=4. Implementation: trueno kernel (Grid (1,M) 256 threads, 8-warp smem reduction). Pass 1: load residual+input, store sum, accumulate sq_sum. Pass 2: normalize from sum via rms_inv. Results (yoga 4060L, 1900MHz, 60s, isolated): c=4 235.0 aggregate (baseline 235.0 with revert = same — earlier 248 was from different binary). **Root cause**: Fused kernel restricts to (1,M) grid for RMSNorm reduction, losing 6x CTA parallelism vs separate residual_add (ceil(1536/256),M) = (6,M) grid. For hidden_dim=1536, residual buffer is 6KB/batch — fits entirely in L2 cache (48MB on RTX 4060L), so the extra memory round-trip between kernels is effectively free (~0μs). **Lesson**: Kernel fusion only helps when intermediate data exceeds L2 cache or when launch overhead dominates. For small hidden dims and modern GPUs with large L2, separate memory-bound kernels outperform fused compute-constrained kernels. Reverted to separate kernels. Kernel retained in trueno for potential use on Jetson (smaller L2). |
| 2.47.0 | 2026-03-12 | **PMAT-091 Phase 4 FALSIFIED: W4A16 interleaved WMMA 3.5× SLOWER than FP8 baseline.** Benchmark (yoga 4060L, 1900MHz, 60s, isolated, c=4): W4A16 71.4 aggregate / 18.8 decode / 53.3ms ITL vs baseline 248.2 aggregate / 64.3 decode / 15.6ms ITL. step_ms=50.5 (W4A16) vs ~15ms (FP8/DP4A). **Root cause analysis**: Custom PTX WMMA kernel is dramatically underperforming — expected, as first implementation lacks: (1) shared memory double-buffering for A/B tiles, (2) k-dimension tiling to amortize WMMA overhead, (3) proper register management for 4-warp occupancy. The coalescing fix (interleaved layout) is correct in principle but the WMMA compute path itself is too slow. **Key insight**: cuBLAS FP8 GEMM already provides tensor core acceleration with 1 B/elem BW — the benefit of W4A16 (0.5625 B/elem) requires a highly optimized WMMA implementation to overcome the dequantization overhead. Default remains OFF. Next step: profile individual WMMA kernel to identify bottleneck (register pressure vs shared memory bank conflicts vs WMMA utilization). |
| 2.46.0 | 2026-03-12 | **PMAT-091 Phase 3 complete: W4A16 wired into realizr batched decode.** Phase 3 (realizr 92cc697): `GpuProfile.w4a16_interleaved` detection (default OFF, `W4A16_INTERLEAVED=1`), `InterleavedWmmaQ4KGemm` kernel type + PTX generation, `interleaved_weight_cache` HashMap for repacked Q4K tiles, `warmup_interleaved_cache()` at model init (download Q4K → CPU repack → re-upload), dispatch guard in `batched_gemv_or_gemm()` for M>=2 Q4K. trueno re-export fix (2ec833a): `repack_q4k_interleaved` accessible from `trueno_gpu::kernels::`. All 3 code phases complete — Phase 4: benchmark on yoga. |
| 2.45.0 | 2026-03-12 | **PMAT-091: W4A16 coalesced WMMA GEMM — Phase 1-2 complete (weight repack + kernel).** Root cause: current WMMA Q4K GEMM kernels (PMAT-045/064) are 5.1× slower than cuBLAS due to uncoalesced B (weight) loads. Adjacent threads access super-blocks ~864 bytes apart (num_sb × 144B stride per column). Fix: column-interleaved Q4K weight layout groups 16 columns' SBs into 2304-byte tiles with byte-interleaved qs. New thread-to-element mapping (col = tid % 32) ensures 16 adjacent threads load 16 consecutive qs bytes — single 128B cache line vs 16 separate fetches. Phase 1: `repack_q4k_interleaved()` CPU repack function (trueno 20c0b57). Phase 2: `InterleavedWmmaQ4KGemmKernel` coalesced 4-warp WMMA GEMM (trueno 00f40c5). Phase 3 pending: wire into realizr batched decode, benchmark vs FP8. At 70% WMMA efficiency: est. ~355 tok/s c=4 (+34% from 264.6), approaching llama.cpp's 338.6. |
| 2.44.0 | 2026-03-12 | **PMAT-090: FP8 cuBLASLt batched decode — breaks DP4A compute ceiling at high M.** Hypothesis: DP4A Q4K GEMV is compute-bound at M>1 (4 independent DP4A accumulation chains saturate INT32 units, theoretical ceiling 306 tok/s at M=4). FP8 E4M3 cuBLASLt GEMM (1 B/elem) stays memory-bound via tensor cores despite 1.78× more BW than Q4K (0.5625 B/elem). Implementation: new `fp8_decode` field in GpuProfile (auto-detected on sm_89+), routes batched decode through `cublas_prefill_gemm()` when M≥2. Fused gate+up DP4A disabled when FP8 active (individual projections route through cuBLASLt). **A/B results (yoga 4060L, 1900MHz, 60s, isolated):** c=1: 153.8 tok/s (no regression, FP8 inactive at M=1). c=4: 264.6 vs 253.8 DP4A (+4.3% aggregate, step_ms 13.5 vs ~15ms). c=8: **432.3 vs 324.8 DP4A (+33% aggregate)**, ITL 17.2 vs 23.1ms (-25.5%). **Key insight:** FP8 benefit scales strongly with M — at M=4 the per-GEMM pipeline overhead (~2ms conversion + dispatch) nearly cancels compute savings; at M=8 overhead amortizes and tensor cores dominate. Original 1.5× c=4 hypothesis PARTIALLY FALSIFIED (actual +4.3%), but c=8 result (+33%) is substantial and breaks the DP4A ceiling. Default ON for sm_89+ (override: FP8_DECODE=0). realizr commit dd85809. |
| 2.43.0 | 2026-03-12 | **PMAT-089: Q4K block size + warp reduction — BOTH FALSIFIED.** Two hypotheses tested: (1) 4 warps (128 threads): -2% decode regression from register pressure (128×255 = 32K regs/block → reduces blocks/SM from ~3 to ~2). Reverted to 3 warps. (2) Parallel warp-0 shuffle reduction: 0% measurable improvement — serial loop of 6 iterations is negligible fraction of kernel time (<0.1%). Code kept (cleaner than serial). Benchmarks: c=1 decode 153.1 tok/s (+0.4% = noise), c=4 aggregate 251.6 (-1.8% = noise). Key insight: 25% per-step gap vs llama.cpp is NOT block-level or reduction-level — must come from deeper architectural differences (CUDA graphs, attention kernels, dispatch patterns). W4A16 tensor core GEMM (PMAT-054B) remains next impactful intervention. |
| 2.42.0 | 2026-03-12 | **PMAT-029 Phase 1 benchmarked: NO measurable decode impact — memory-bound CONFIRMED.** Yoga benchmark (1900MHz, 60s, isolated): c=1 decode 152.5 tok/s (was 154.8, -1.5% = noise), ITL 6.6ms (was 6.5), µs/layer 234.2 (was 230.8). c=4 aggregate 256.2 (was 257.4, -0.5% = noise), ITL 14.6ms (was 15.1, -3.3%). Instruction reduction (108→93 insn/SB, -14%) does not improve throughput because Q4K GEMV is **memory-bound** at M=1 — compute is fully hidden behind DRAM latency. At M=4 (more compute-bound), marginal ITL improvement within noise. PMAT-029 Phase 1 CLOSED: correct optimization but wrong bottleneck. Remaining instruction reduction phases deferred — next impactful intervention is W4A16 tensor core GEMM (PMAT-054B). |
| 2.41.0 | 2026-03-12 | **PMAT-029 Phase 1: Q4K constant hoisting — 108→93 insn/SB (-14%).** Hoisted 3 bitmask constants (`0x3F3F3F3F`, `0x0F0F0F0F`, `0x03030303`) before inner super-block loop in all 4 Q4K DP4A kernel variants (hw_dp4a, batched_hw_dp4a, fused_gate_up_swiglu_hw_dp4a, dp4a_gemm). Root cause: `and_u32_imm()` emits `mov.u32` + `and.b32` per call — hoisting eliminates 7-10 redundant `mov` instructions per super-block. Instruction density: 6.75→5.8 insn/val. Fused variant saves 20 insn/SB (2× scale extraction). trueno commits pushed (2 commits). |
| 2.40.0 | 2026-03-12 | **Kaizen: Stale section cleanup.** Updated executive summary (decode parity achieved, 4-way benchmark), deployment topology (yoga PRIMARY, Jetson secondary, vLLM added), Jetson baselines (40.8/36.1 at MAXN_SUPER, TTFT 1.4x PASS), optimization roadmap tiers (decode parity + prefill parity + continuous batching all DONE, W4A16 is next), priority matrix (30 items → 16, completed items consolidated). Five-Whys and PMAT ticket table already updated in v2.39.0. |
| 2.39.0 | 2026-03-12 | **Full doc update: yoga 1900MHz + Jetson MAXN_SUPER baselines across all specs.** Updated cross-platform decode table (4060L: 168.3/154.8/160.7/163.5 at 1900MHz). Updated performance.md (c=1 and c=4 sections with 1900MHz numbers). Updated README.md performance section. Updated gguf-decode.md (Jetson baselines 40.8/36.1, yoga 1900MHz, status: parity achieved). Updated gguf-prefill.md (status 1.33x yoga / 1.4x Jetson, PASS < 2x target). Updated baselines.md (8 thresholds refreshed, all passing). Updated perf-parity-spec.md to v3.24.0 (added Jetson as secondary test target, c=1 all PASS, Jetson c=4 zero-benefit documented). |
| 2.38.0 | 2026-03-12 | **Jetson MAXN_SUPER correction + PTX JIT target fix.** Root cause of apparent -27% Jetson regression (36.3→26.6 tok/s): Jetson had reverted to 15W power mode (GPU 612 MHz) after reboot, NOT a code bug. Fixed: `nvpmodel -m 2` (MAXN_SUPER, GPU 1020 MHz). Also fixed trueno PTX JIT target clamping (trueno#184, commit 756c85f): Blackwell commit (a0b243a) incorrectly clamped ALL sm_major>7 to sm_70 — broke Jetson sm_87. Fixed to clamp at sm_90 (PTX 8.0 ceiling). **Updated Jetson baselines (MAXN_SUPER):** realizr 40.8 tok/s (+12.4%), llama.cpp 36.1 tok/s (+9.1%). Realizr now **13% faster** than llama.cpp on decode (was 10%). TTFT: 47.8ms vs 34.0ms (1.4x, was 5.5x). FP16 warmup OOMs on 8GB unified memory (non-fatal). c=4 batching provides zero benefit on 8 SMs. Added `jetson-maxn` resource to forjar-jetson-realizr.yaml for automatic power mode setup. |
| 2.37.0 | 2026-03-12 | **PMAT-088d: Multi-prompt batch recycle — TTFT 63.6→60.7ms (-4.6%).** Replaced sequential `recycle_slot` (N × `run_prefill` + `force_workspace_reinit`) with single `prefill_multi_prompt` call using `slot_indices: Option<&[usize]>` for targeted KV scatter. N=1 recycle 17.3→16.3ms (-6%, no workspace reinit), N=2 batch 34.6→29.8ms (-14%, one weight read). 33% of recycles batched at N=2. c=4 aggregate 257.4 tok/s (84% of DP4A ceiling). TTFT bottleneck analysis: staggered slot finish pattern → reconnect(5ms) + decode wait(7.5ms) + recycle(16ms) + decode(15ms) = ~44ms typical, ~61ms P50. Further improvement requires chunked prefill or dual-stream prefill. H-CB14 CONFIRMED (modest). |
| 2.36.0 | 2026-03-12 | **PMAT-088b H-CB11 FALSIFIED: Batched CUDA graph 3ms SLOWER than eager.** 654 kernel launches/step (23/layer × 28 + 10 final) = 2.6ms overhead (revised from 0.8ms). Graph replay 18.1ms vs eager 15.1ms ITL despite async H2D fixes. Root causes: (1) attention grid dims frozen at capture (dummy seq_lens=1 → wrong grid for seq_lens=128+), (2) RoPE positions frozen at [0,...,M-1], (3) 654-node graph management overhead, (4) HGEMM guard blocks cuBLAS during capture. Phase 2 CLOSED — both HGEMM (H-CB9) and CUDA graph (H-CB11) falsified. Revised priorities: Phase 2b (trueno GEMV kernel optimization) and Phase 3 (chunked prefill) are the remaining impactful interventions. Theoretical model corrected: launch overhead 2.6ms not 0.8ms. |
| 2.35.0 | 2026-03-11 | **PMAT-088b comparative analysis: llama.cpp also uses DP4A but 34% faster per-step.** First proper comparative benchmark with llama.cpp --parallel 8. llama.cpp c=4: 353.0 aggregate (11.3ms ITL), c=8: 414.1 (19.3ms ITL). Two sources of gap: (1) Per-step compute 1.34× slower (compute/token 2.43ms vs 1.94ms, kernel efficiency + no CUDA graphs). (2) Scheduling overhead 14% vs 5% (sequential prefill 82ms/slot vs llama.cpp 18.6ms). HGEMM crossover FALSIFIED (H-CB9) — both runtimes use DP4A at M≤8, gap is kernel efficiency not architecture. Revised Phase 2: (a) CUDA graph for M>1 (~0.8ms/step), (b) trueno GEMV optimization, (c) chunked prefill (reduce 14%→5% scheduling overhead). |
| 2.34.0 | 2026-03-11 | **PMAT-088b scaling analysis: DP4A aggregate ceiling = 412 tok/s.** First c=8 benchmarks with iteration scheduler (max_slots=8): 306.5 aggregate (2.02x c=1), 87% of theoretical ceiling, 21.9ms ITL. Theoretical model: `M / (2.56 + M×2.425 + 0.8)` tok/s — compute per token (2.425ms DP4A) is the binding constraint. The 3.0x c=4 target (455 tok/s) exceeds the DP4A ceiling (306 at M=4, 412 at M→∞). Reaching 3x requires W4A16 tensor core GEMM (like vLLM AWQ: Q4 storage + FP16 compute). c=16 with max_slots=16 hits CUDA_ERROR_ILLEGAL_ADDRESS during prefill (KV cache reuse bug — separate ticket). Updated forjar-yoga-realizr.yaml with ITERATION_SCHEDULER=1, CUDA_MAX_BATCH=8. |
| 2.33.0 | 2026-03-11 | **PMAT-088b H-CB9 FALSIFIED: HGEMM crossover does NOT beat DP4A at M=4.** Three variants tested at 1900 MHz on RTX 4060L: (1) Full HGEMM: 256.0 aggregate, 13.6ms ITL (−2.1%). (2) Hybrid (fused QKV/gate+up DP4A + HGEMM output/down): 260.5 aggregate, 13.7ms ITL (−0.4%). (3) Baseline DP4A: 261.5 aggregate, 13.4ms ITL. FP16 weights (2 B/elem, 2944 MB) cost 3.5× more BW than Q4K (0.5625 B/elem, 850 MB) — tensor core compute savings do not compensate. PMAT-062 confound resolved: re-enabling fused gate+up does not change conclusion. Phase 2 redesigned: HGEMM dropped, remaining intervention is CUDA graph for M>1 (~1ms launch savings). Phase 3 (chunked prefill) elevated to higher priority. Infrastructure preserved: FP16 cache warmup code env-gated (`HGEMM_BATCHED_DECODE=1`) for future experimentation. |
| 2.32.0 | 2026-03-11 | **PMAT-088b root cause corrected: M=4 penalty is GEMV compute (DP4A chains), NOT attention KV reads.** Profiled M=4 decode step (13.3ms): 2.56ms BW + 9.7ms compute + 1ms launches. Attention KV reads are 14 MB = 2.8% of weight BW (491 MB) — FlashAttention-2 would yield <3% improvement. Root cause: batched Q4K GEMV reads weights once for M=4 but runs 4× independent DP4A accumulation chains, transitioning from memory-bound (M=1) to compute-bound (M=4). Phase 2 redesigned: (a) HGEMM crossover at M>1 (tensor cores vs DP4A), (b) CUDA graph for M>1 (save 1ms launch overhead). FlashAttention-2 deferred — wrong bottleneck. Added H-CB9 (HGEMM crossover hypothesis) and H-CB10 (attention fraction, CONFIRMED at 2.8%). |
| 2.31.0 | 2026-03-11 | **PMAT-088 corrected numbers: +10.4% aggregate (post-bugfix), variable-M buffer fixes.** Initial 256.9 measurement was inflated by error-retry cycles from 3 classes of buffer length mismatch exposed by variable M transitions (M=4→M=1→M=3). Corrected final: 232.7 aggregate (+10.4% from 210.8), 38.3% efficiency (was 34.1%). Bug classes: (1) logits buffer M×vocab vs vocab in prepare_capture_buffers, (2) input/hidden grow-only buffer vs exact copy_from_host, (3) KV ptr/seq_lens high-water-mark buffers. Fixes: exact-size realloc, from_raw_parts M-views, copy_from_host_at, clear_decode_graph on realloc. H-CB8 revised to +10.4% (still CONFIRMED). |
| 2.30.0 | 2026-03-11 | **PMAT-088a DONE: Iteration scheduler — +21.9% initial (pre-bugfix), H-CB4/CB7 FALSIFIED.** Two hypotheses falsified: (1) H-CB4: sequential M=1 NOT faster than batched M=4 — GEMV weight amortization (828MB shared vs 4×828MB) means batched is essential. (2) H-CB7: 42.0% < 60% threshold — per-slot M=1 reads 4× more weight data. Gain came from waiting queue integration reducing slot vacancy, not M=1 scheduling. Corrected architecture: path to high scaling is reducing attention cost within batched execution (FlashAttention-2 batched, Phase 2 = new critical path), not eliminating batching. |
| 2.29.0 | 2026-03-11 | **PMAT-088: Continuous batching architecture design (GH-141).** Added §5b with full architecture design for iteration-level scheduling, paged KV cache, and chunked prefill. Literature review: Orca (OSDI '22, iteration-level scheduling, 36.9x), PagedAttention/vLLM (SOSP '23, <4% memory waste, 2-4x), Sarathi-Serve (OSDI '24, chunked prefill, 2.6-6.9x), DistServe (OSDI '24, disaggregated phases), DeepSpeed-FastGen (Dynamic SplitFuse), FlashInfer (MLSys '25, block-sparse attention). Key insight: 2.55x per-slot M=4 degradation is artifact of batched GEMV, NOT inherent to c=4. Iteration-level scheduling with M=1 forward passes eliminates the penalty. 4-phase plan: (P1) iteration scheduler → ~500 agg, (P2) paged KV cache → <4% waste, (P3) chunked prefill → no generation stalls, (P4) CUDA graphs for paged. Added 4 falsification conditions (H-CB4 through H-CB7), 5 new academic references (#36-40). |
| 2.28.0 | 2026-03-11 | **PMAT-087: Clock correction 1500→1900 MHz — 8/9 dimensions at A+.** SM clock was locked at 1500 MHz (26% below natural sustained boost of 1890 MHz). Corrected to 1900 MHz across all 4 yoga forjar configs. realizr c=1: 138.6→154.8 tok/s (+11.7%), 258.6→230.8 µs/layer, TTFT 46.4→13.4ms. Higher SM clocks help because Q4K dequantization is ~15% compute (DP4A, shared mem ops). Layer decode: 88 A-→97 A+. Composite: 91 A→98 A+. 8 of 9 dimensions now A+. Only remaining gap: concurrency scaling (34.1% efficiency, score 51 C) — requires GH-141 continuous batching. llama.cpp also improved: 142.9→160.7 tok/s (+12.5%), gap to realizr unchanged at ~3.7%. |
| 2.27.0 | 2026-03-11 | **PMAT-086: Host-side batched decode optimization — IMPLEMENTED, FALSIFIED.** Pre-allocated GPU input/logits buffers (grow-only) + removed redundant stream.synchronize() before batched argmax + pre-allocated pos_buf in BatchedDecodeState. Measured impact: <0.1ms — kernel time (17-18ms at M=4) dominates. Five-Whys confirmed: per-slot ITL degradation (2.55x at M=4) is structural — attention KV scales linearly with M, GEMV computes M dot products. Host overhead was <0.5ms (not 2-4ms as estimated). Corrected scorecard: layer decode 88 A- (258.6 us/layer), concurrency scaling 57 C+ (38.0%). A-grade on scaling requires GH-141 continuous batching (architecture change). |
| 2.26.0 | 2026-03-11 | **Scorecard v3.0.0: 9 scoring dimensions.** Added `probador llm score` with 9 dimensions: composite, layer decode, prompt profile, output length, correctness, memory efficiency, cold start, power efficiency, concurrency scaling. realizr at A or above on 5/7 dimensions. Two gaps to A: layer decode c=1 (88 A- → need 90), concurrency scaling (38% C+ → need 90%). Correctness 100%, memory A+, power A+. |
| 2.25.0 | 2026-03-11 | **PMAT-076: Dead slot masking — IMPLEMENTED.** Added `batched_done_mask` field to CudaExecutor, set from `BatchedDecodeState.done` before each forward pass. All three attention paths (batched, flash decode, graph replay) zero `seq_lens[i]` for done slots → early-exit (zero KV iterations). Verified: heterogeneous max_tokens (10+128) batch completes correctly. No regression: c=1=138.9, c=4=199.0 tok/s on uniform traffic. Impact scales with dead-slot ratio — needs PMAT-077 heterogeneous probador for quantitative measurement. |
| 2.24.0 | 2026-03-11 | **PMAT-075: Batched CUDA graph infrastructure — IMPLEMENTED, FALSIFIED.** Three fixes: (1) `batched_cleanup` preserves workspace buffers via PAR-200 skip path, (2) preserves KV caches + auxiliary pointer buffers for address stability, (3) `init_batched_kv_cache_gpu` skips auxiliary realloc on reuse. Latent bug fixed: `init_prefill_workspace` now clears batched graphs on realloc. **Result:** Graphs persist across batches (confirmed via logs). But graph replay is **2.8ms SLOWER** than eager (ITL 22.0ms vs 19.2ms). Root cause: 5 synchronous `cuMemcpyHtoD` calls + graph dispatch overhead > kernel launch savings (~0.8ms). `BATCHED_GRAPH=1` remains opt-in. **Falsified hypothesis:** kernel launch overhead was NOT ~5ms/step — measured ~0.8ms. M=4 degradation is primarily batched attention scaling + L2 working set pressure. No regression on eager path: c=1=139.1, c=4=198.9. |
| 2.23.0 | 2026-03-11 | **Post-continuous batching five-whys analysis.** Three findings: (1) Per-slot decode degrades 2.67× at M=4 (ITL 19.2ms vs 7.2ms) — root cause: no CUDA graphs in batched path (+~5ms kernel launch overhead), L2 cache spreading (+12%), batched attention 4× scaling. (2) Dead slot compute waste — done slots get zero embeddings through full forward pass, ~25% wasted compute per dead slot. (3) Uniform traffic defeats recycling — probador sends identical requests, all finish at same gen_idx. Three new work items: PMAT-075 (batched CUDA graphs), PMAT-076 (dead slot masking), PMAT-077 (probador heterogeneous traffic). Updated trajectory: 216→~290 aggregate predicted. |
| 2.22.0 | 2026-03-10 | **PMAT-074 DONE: Slot recycling (continuous batching Phase 3).** `recycle_slot()` reuses finished slot indices — KV cache overwrite, state replacement at slot index, `max_tokens_max` extended for gen_idx offset. 159 recycling events in single 31s continuous batch. Heterogeneous traffic (mixed max_tokens 16/32/64/128, c=8): **216.4 tok/s** (+10% vs uniform 197). Uniform traffic (probador max_tokens=128): no gain (all slots finish simultaneously, no recycling opportunity). Prefill overhead: ~16ms per recycled slot. No regressions: c=1=138.8, c=4=196.7. All 3 continuous batching phases complete (PMAT-072/073/074). |
| 2.21.0 | 2026-03-10 | **PMAT-073 DONE: Mid-batch joins.** Three bugs fixed: (1) GPU buffer length mismatch — attention vectors padded to match pre-allocated KV buffer size, (2) RwLock contention — `model_architecture()` and `model_eos_token_id()` blocked HTTP handlers ~2s during batch decode, fixed by caching at AppState construction, (3) m=1 fast path preserved — batched path 3x slower at c=1, keep monolithic path for single requests. Mid-batch joins verified: 4 joins during c=4 benchmark (slots join running batch with ~31ms prefill). c=1: 138.9 tok/s (no regression), c=4: 197.4 tok/s (no regression). No throughput gain at c=4 because probador sends all requests simultaneously → initial batch is already full. |
| 2.20.0 | 2026-03-10 | **PMAT-072 DONE: Step-wise batched decode.** Refactored `generate_batched_streaming` into 3-method API: `batched_setup_and_prefill` → `batched_decode_step` → `batched_cleanup`. Scheduler releases model lock between decode steps (~19ms hold vs ~660ms). Result: 196.9 tok/s c=4 (baseline 197.5) — lock release alone does NOT improve throughput (single scheduler thread). Confirmed falsification: bottleneck is absence of mid-batch scheduling, not lock scope. PMAT-073 now unblocked. c=1: 138.9 tok/s (no regression). |
| 2.19.0 | 2026-03-10 | **Five-Whys RCA for 3 remaining gaps + PMAT-071/072/073/074 implementation plan.** Gap 1: TTFT/Prefill 3.8x — cuBLAS HGEMM reads FP16 (3.5x BW vs Q4K), fix: tiled Q4K GEMM (PMAT-071). Gap 2: c=4 0.67x — `model.write()` held 2.7s, fix: continuous batching in 3 phases (PMAT-072/073/074). Gap 3: c=4 0.33x vs vLLM — architectural (scheduler + PagedAttention). Updated baselines to PMAT-062 numbers. Falsification conditions for each gap. |
| 2.18.0 | 2026-03-10 | **Full 4-runtime benchmark with vLLM baseline.** First isolated serial benchmark including vLLM (AWQ INT4). c=1: vLLM 160.1, ollama 150.7, llama.cpp 143.5, realizr 140.0 — decode 4-way near-parity. TTFT improved 50.9→25.4ms via CORRECTNESS-014 fix. c=4: vLLM **567.3** aggregate (3.1x over realizr), llama.cpp 302.2 (1.7x), realizr 180.9 (+108% from 86.8 via CORRECTNESS-014 + PMAT-046). Updated performance.md, README.md, cross-platform summary with vLLM column. |
| 2.17.0 | 2026-03-10 | **CORRECTNESS-014 FIXED: CUDA context corruption.** Root cause: `init_prefill_workspace` reallocated workspace buffers when prompt length exceeded `buffer_capacity`, but decode graph was NOT cleared — graph replayed with stale GPU pointers → `CUDA_ERROR_ILLEGAL_ADDRESS` on 4th request. Fix: clear `decode_graph` in `init_prefill_workspace` before reallocation. Also fixed wrong comment "workspace pointers are stable" in `generate_2.rs`. **CORRECTNESS-013 narrowed:** Identical prompts produce different tokens across slots in batch 2+. First batch after server start is correct. Root cause: `PMAT-058` KV cache free/reinit between batches produces corrupted state. Env var `CUDA_GRAPH_DISABLE=1` or `SKIP_CUDA_GRAPH=1` disables graph capture (NOT `DECODE_GRAPH=0` which doesn't exist). |
| 2.16.0 | 2026-03-10 | **CORRECTNESS-013: Batched decode frozen slots.** c=4 slots 1-3 produce constant tokens per step — confirmed in BOTH HGEMM and DP4A paths. Five-whys: `correctness_under_batching` contract obligation has NO realizr implementation (BIND-003). `pv audit` confirms gap. Bug is in common path (prefill→decode KV cache transition), NOT path-specific. Added falsification tests H-CB1/CB2/CB3, instrumentation plan (7 tools), provable contract references to §12. PMAT-070 ticket for wired FALSIFY-CB-006 test. |
| 2.15.0 | 2026-03-08 | **DECODE PARITY CONFIRMED — yoga RTX 4060 Laptop (PMAT-044).** Full 3-runtime serial isolated benchmarks on yoga (sm_89, 24 SMs, 8GB). Decode: realizr 140.3, llama.cpp 143.6, ollama 150.9 — **0.98x parity** (within noise). Three-way decode within 7%. Prefill gap: 4.6x (452 vs 2097 tok/s) — HGEMM FP16 vs fused Q4K GEMM. c=4 concurrency: 3.4x aggregate gap (86.8 vs 291.9 tok/s) — RwLock serialization ([realizr#141](https://github.com/paiml/realizar/issues/141)). Cross-platform summary: Jetson 0.91x (faster), 4060L 0.98x (parity), 4090 1.06x (near parity). Provable contracts: ptx-target-parity-v1.yaml with 3 bindings, contract-falsify passes. Updated README, performance.md, perf-parity-spec.md. |
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
