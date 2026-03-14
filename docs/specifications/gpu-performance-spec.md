# GPU Decoder Throughput Performance Specification

**Document ID:** REALIZAR-GPU-PERF-001
**Version:** 2.86.0
**Status:** ACTIVE
**Date:** 2026-03-14
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

**Competition Reality (Mar 13, 2026 — yoga RTX 4060L @ 1900MHz, PMAT-109):** Under standardized load testing (60s, streaming, short prompt):
- **c=1 decode:** realizr 149.5 vs llama.cpp ~150 (**1.00x**), TTFT 13.2ms vs 10.2ms (1.29x, PASS < 2x). **PMAT-109: TTFT P99 14.2ms (bimodal tail ELIMINATED)**
- **c=4 aggregate:** realizr 357.2 vs llama.cpp 365.8 (**0.98x**, PARITY, short prompt) — ITL 10.4 vs 10.7ms (realizr better), TTFT 36 vs 19ms (llama.cpp better). **PMAT-113: Medium prompt (~102 tok) shifts to 293.6 vs 362.7 (0.81×)** — FP8 prefill BW overhead exposed
- **c=8 aggregate:** realizr **637.8** vs llama.cpp 430.0 (**1.48x**, DOMINATES) — FP8 tensor cores at M>=5. Medium prompt: 474.9 vs 441.5 (1.08x, still wins)
- **c=12 aggregate:** realizr 899.3 vs llama.cpp 906.0 (**0.99x**, PARITY) — ITL 11.4 vs 12.5ms
- **c=16 aggregate:** realizr **1139.5** vs llama.cpp 1000.4 (**1.14x**, WINS) — 0% errors vs 2.2%. Medium prompt: 749.7 vs 1045.3 (0.72x — TTFT penalty dominates)
- **vLLM reference (medium prompt, PMAT-113/121):** c=4 551.0, c=8 1023.2, c=12 **1418.5**, c=16 **1778.5** — W4A16 Marlin + PagedAttention dominates all prompt profiles. Complete prompt matrix: ±6% invariant across short/medium/long at all c
- **PMAT-114 FALSIFICATION:** Initial c=16 medium run (504.5 tok/s llama.cpp) was measurement artifact — GPU contention from freshly-killed vLLM. Verification: 1045.3 (+0.8% from short). **llama.cpp is prompt-length invariant at all concurrency levels.**
- **PMAT-117 ollama:** Best M=1 decode (164.5 tok/s, +3% vs llama.cpp) but **serial processing** — TTFT explodes at c=4 (602ms, 32× llama.cpp). Production-incompatible at c>1. Only suitable for single-user interactive use.
- **Production Workload Guide (PMAT-113→117):** No single runtime optimal. c=1-7 + medium prompts → llama.cpp. c≥8 → realizr (FP8 decode, 1.25× at c=8 128-tok). All concurrencies → vLLM (2-3× all others). Single-user → ollama (best M=1 decode, zero-config).
- **Cross-platform:** Jetson Orin realizr **13% FASTER** than llama.cpp on decode (40.8 vs 36.1 tok/s)
- **PMAT-105 breakthrough:** LmHead (Q6K, 151,936×1536) was using batched GEMV (reads weights M times) instead of FP8 cuBLASLt (reads weights once). Routing through `batched_gemv_or_gemm` enables FP8 dispatch at M>=5. Single biggest optimization since FP8 prefill. ITL now nearly flat c=4→c=16 (10.4→11.7ms).

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

**RTX 4060 Laptop — yoga (Mar 12 2026, c=1, isolated, streaming, 1900MHz, PMAT-097 era):**

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|-------------|--------------|---------------|-------------|
| llama.cpp | **161.7** | **2,255** | **10.2** | **6.2** |
| vLLM | 153.6 | 1,831 | 12.6 | 6.5 |
| realizr | 148.5 | 1,640 | 14.0 | 6.7 |
| ollama | **164.6** | 330 | 69.8 | **6.1** |

**c=1 decode: near-parity.** realizr 148.5 vs llama.cpp 161.7 = **0.92x**. Ollama has **best decode** (164.6, +2% vs llama.cpp) from exclusive GPU access, but worst TTFT (69.8ms, 7× llama.cpp) from serial HTTP layer.
TTFT: realizr 14.0ms vs llama.cpp 10.2ms = **1.37x** (PASS < 2x target). FP8 prefill via cuBLASLt.

**RTX 4060 Laptop — yoga (Mar 12 2026, c=4, isolated, streaming, 1900MHz, PMAT-097 era):**

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| **vLLM** | **594.8** | **150.4** | 25.3 | **6.7** |
| llama.cpp | 348.7 | 89.2 | **26.0** | 11.2 |
| realizr | 259.7 | 65.6 | 39.8 | 15.3 |
| ollama | 159.1 | **161.5** | 612.3 | **6.2** |

**RTX 4060 Laptop — yoga (Mar 12 2026, c=8, isolated, streaming, 1900MHz, PMAT-097 era):**

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Error % |
|---------|----------------|-------------|---------------|-------------|---------|
| **realizr** | **456.4** | 61.8 | 58.6 | 16.2 | **0%** |
| llama.cpp | 416.2 | 53.0 | 30.8 | 18.9 | 1.6% |

**c=8: realizr BEATS llama.cpp** (456.4 vs 416.2, **+10%**). FP8 tensor core GEMM at M>=5 (PMAT-093). llama.cpp has 1.6% error rate; realizr 0%. Both tested with 16-slot parallel/batch config.

**c=4: Near-parity (0.98x) with corrected --parallel 16.** PMAT-105 routed LmHead through FP8 cuBLASLt, eliminating the Q6K batched GEMV bottleneck (151,936×1536 weights read M times → read once). ITL dropped 25% (13.9→10.4ms). realizr has better ITL (10.4 vs 10.7ms) and 0% errors. llama.cpp has better TTFT (19ms vs 36ms) from fused Q4K prefill. realizr **dominates at c=8** (1.48x) and **wins at c=16** (1.14x).

**Concurrency scaling — realizr vs llama.cpp (yoga RTX 4060L, 1900MHz, PMAT-111 fresh v2, both --parallel/batch 16):**

| c | realizr tok/s | llama.cpp tok/s | Ratio | realizr ITL | llama.cpp ITL | realizr TTFT | llama.cpp TTFT | realizr err | llama.cpp err |
|---|--------------|----------------|-------|------------|--------------|-------------|---------------|------------|--------------|
| 1 | 149.5 | 160.2 | 0.93x | 6.7ms | 6.2ms | 13.2ms | 9.8ms | 0% | 1.7% |
| 4 | 355.5 | 352.7 | **1.01x** | **10.4ms** | 11.1ms | 36.1ms | 19.6ms | 0% | 2.8% |
| **8** | **631.9** | 428.9 | **1.47x** | **11.3ms** | 18.4ms | 52.7ms | 28.2ms | **0%** | 3.0% |
| 12 | 898.4 | 927.3 | 0.97x | **11.4ms** | 12.6ms | 70.6ms | 22.2ms | **0%** | 1.3% |
| **16** | **1,139.8** | 1,037.1 | **1.10x** | **11.7ms** | 14.8ms | 87.0ms | 32.3ms | **0%** | 0.9% |

**PMAT-111 TTFT analysis (Mar 13):** TTFT_TRACE reveals realizr's c=4 TTFT (36ms) = 21ms multi-prompt FP8 prefill (4×29=116 tokens, weights read once) + 11ms first decode step (M=4 DP4A) + 4ms HTTP/queue overhead. llama.cpp's c=4 TTFT (19.6ms) is lower because it interleaves prefill with ongoing decode (no "wait for previous batch" delay). TTFT scales ~linearly with concurrency for realizr (13→87ms, 6.6×) while llama.cpp is sublinear (9.8→32.3ms, 3.3×). This is a structural consequence of realizr's batch-and-step scheduler architecture vs llama.cpp's continuous batching.

**Full concurrency scaling — all 4 runtimes (yoga 4060L, 1900MHz, short prompt, 60s, warmup, isolated, PMAT-120):**

| c | vLLM | realizr | llama.cpp | ollama | realizr/vLLM | Scaling eff (vLLM) |
|---|------|---------|-----------|--------|-------------|-------------------|
| 1 | 153.6 | 149.5 | 160.2 | 164.6 | 0.97x | — |
| 4 | 594.8 | 355.5 | 352.7 | 159.1 | 0.60x | 97% |
| 8 | **1,058.5** | 631.9 | 428.9 | — | 0.60x | 86% |
| 12 | **1,461.3** | 898.4 | 927.3 | — | 0.61x | 79% |
| 16 | **1,832.2** | 1,139.8 | 1,037.1 | — | 0.62x | **74%** |

**vLLM scaling: 11.9x (c=1→c=16, 74% efficiency).** W4A16 Marlin + PagedAttention + continuous batching. realizr: 7.6x (47.5%). llama.cpp: 6.5x (40.4%). ollama: ~1.0x (serial). vLLM ITL c=1→c=16: 6.5→7.4ms (+14%). realizr: 6.7→11.7ms (+75%). llama.cpp: 6.2→14.8ms (+139%). **vLLM has the most stable per-request latency under concurrency.**

**Realizr competitive advantages:**
- **0% error rate** at all concurrency levels (llama.cpp: 0.9-3.0%)
- **ITL nearly flat** c=1→c=16 (6.7→11.7ms, +75%) vs llama.cpp (6.2→14.8ms, +139%)
- **Dominates at c=8** (1.47x) where FP8 tensor cores at M>=5 provide maximum advantage
- **Wins at c=16** (1.10x) with significantly better tail latency (ITL CV: 0.00 vs 0.02)

**PMAT-105 breakthrough (supersedes PMAT-103):** LmHead (Q6K, 151,936×1536 = 175 MB Q6K weights) was using `batched_gemv_with_fallback`, which always dispatches to Q6K batched GEMV — reads weights M times (once per sequence). Routing through `batched_gemv_or_gemm` enables FP8 cuBLASLt at M>=5, reading FP8 weights **once** (233 MB). Result: ITL nearly flat from c=4→c=16 (10.4→11.7ms, only +12.5%). Previous ITL scaling was 13.9→19.8ms (+42.4%). LmHead was the dominant source of ITL scaling with M because it's the largest projection in the model (151,936 output dim).

**PMAT-105 root cause:** The LmHead accounts for ~20% of decode ITL at high M. Q6K batched GEMV reads 175 MB weights per sequence; at M=12, effective DRAM reads were ~660 MB (3.8× single-read with L2 sharing). FP8 cuBLASLt reads 233 MB once regardless of M, using NVIDIA's optimized GEMM scheduling. The 3.5ms → 1.2ms per-step LmHead improvement (saving 2.3ms, -13% ITL) compounds across all concurrency levels. FP8 weight cache was already populated from prefill — zero additional warmup cost.

**Realizr reliability advantage:** 0% error rate at all concurrency levels (c=1-16) vs llama.cpp 1.3-3.2% (with --parallel 16). This matters for production SLA compliance.

**Cross-platform decode summary (c=1, isolated, streaming, short prompt):**

| Platform | vLLM | realizr | llama.cpp | ollama |
|----------|------|---------|-----------|--------|
| **RTX 4060 Laptop** (24 SMs, 1900MHz) | 153.6 | 148.9 | **162.0** | 164.6 |
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

### Scorecard (Mar 13 2026, v3.3.0 — PMAT-109/110 corrected)

**Tool:** `probador llm score` with 9 scoring dimensions (contract: `scoring.yaml` v3.0.0).

**RTX 4060 Laptop — yoga (c=1, isolated, streaming, locked clocks 1900MHz, PMAT-109):**

| Dimension | realizr | llama.cpp | vLLM | ollama | Target |
|-----------|---------|-----------|------|--------|--------|
| **Composite (c=1)** | **98 A+** | 97 A+ | 100 A+ | 78 B | >= 90 A |
| Decode | 98 (149.5) | 100 (158.9) | 100 (153.6) | 100 | >= 90 A |
| TTFT | 100 (13.2ms) | 100 (10.2ms) | 100 (12.6ms) | 53 C (70.5ms) | >= 90 A |
| ITL | 98 (6.7ms) | 99 (6.2ms) | 98 (6.5ms) | 98 | >= 90 A |
| Tail | **100** (P99 14.2ms) | 100 | 100 | 47 | >= 90 A |
| Error | 100 (0%) | 92 (2.6%) | 100 | 100 | >= 90 A |

**RTX 4060 Laptop — yoga (c=4, isolated, streaming, locked clocks 1900MHz, PMAT-111 fresh v2, --parallel/batch 16):**

| Dimension | realizr | llama.cpp | vLLM | Target |
|-----------|---------|-----------|------|--------|
| **Composite (c=4)** | **78 B** | 70 B | 98 A+ | >= 90 A |
| Aggregate | 80 (355.5) | 78 (352.7) | 100 (594.8) | >= 90 A |
| Decode | 60 (95.9) | 54 (90.1) | 100 (150.4) | >= 90 A |
| TTFT | 84 (36.1ms) | 97 (19.6ms) | 93 (25.3ms) | >= 90 A |
| ITL | 72 (10.4ms) | 64 (11.1ms) | 98 (6.7ms) | >= 90 A |
| **Scaling** | **80 (59.5%)** | 55 (55.1%) | 97 (88.9%) | >= 90 A |
| Error | 100 (0%) | 25 (2.8%) | 100 | >= 90 A |

**PMAT-111 fresh data (Mar 13):** realizr 78 B > llama.cpp 70 B at c=4. realizr has better aggregate (+1.01x), decode (+6.4%), ITL (10.4 vs 11.1ms), 0% errors (llama.cpp 2.8%). llama.cpp has better TTFT (19.6 vs 36.1ms) due to continuous batching architecture. Gap to 90 A: +12 points needed, blocked by DP4A compute ceiling.

**Remaining gaps to A (score >= 90):**

| Dimension | Score | Need | Root Cause |
|-----------|-------|------|------------|
| Scaling | 85 A- (59.9%) | 90 A (≥74%) | DP4A GEMV compute scales linearly with M; Q4K at M=4 uses 4× DP4A chains |
| ~~Tail (c=1)~~ | ~~86 A-~~ → **100 A+** | ~~90 A~~ | **PMAT-109 FIXED: Graph persistence eliminates cuGraphExecDestroy from TTFT** |

**PMAT-109: c=1 tail FIXED.** 9/9 interactive dimensions now A- or above. Tail jumped 86→100 (TTFT P99 14.2ms, P99/P50 = 1.08x). c=1 scorecard: **98 A+** (was 94 A).

**Only remaining sub-A dimension: Scaling (85 A-).** Requires 74% efficiency. Q4K DP4A at M=4 is compute-bound — **13 kernel approaches falsified** (including PMAT-110: FP8 for all Q4K at M=4, −5.3%). Only EAGLE speculative (PMAT-009, multi-week) could break this ceiling.

**PMAT-112: TTFT tail is cold-start, not structural (Mar 13).** With `--warmup 5` (excludes first 5s from measurement):

| c | TTFT P50 | P99.9 (no warmup) | P99.9 (warmup) | Tail ratio (warmup) |
|---|---------|-------------------|----------------|---------------------|
| 1 | 13.2ms | 54.2ms | 34.0ms | 1.1x |
| 4 | 36.3ms | 328.6ms | **46.4ms** | 1.3x |
| 8 | 52.6ms | 118.8ms | 54.7ms | 1.0x |
| 16 | 87.1ms | 90.0ms | 88.0ms | 1.0x |

Root cause: first batch allocates KV cache (PAR-119: 28 layers × 16 sequences = 1792MB). Subsequent batches reuse (PMAT-075). c=4 P99.9 drops **7x** with warmup (328→46ms). Production systems with persistent servers should use warmup-representative data.

**PMAT-110 c=4 scoring analysis (Mar 13):** realizr 78 B vs llama.cpp 70 B at c=4 (fresh v2, --parallel 16). realizr wins on Aggregate (1.01x), Decode (+6.6%), ITL (10.4 vs 11.1ms), Error (0% vs 2.8%). llama.cpp wins on TTFT (18.7 vs 36.3ms). Gap to 90 A: +12 points needed, blocked by DP4A compute ceiling. M=4 batched decode: 354.1-354.8ms per 32 tokens (±0.1%), 92% of theoretical DP4A ceiling.

**PMAT-109 details (Mar 13):** Removed `force_workspace_reinit()` from `run_prefill()` and `clear_decode_graph()` from `generate_gpu_resident_streaming()`. PAR-200 workspace reuse in `init_prefill_workspace` already clears graphs when actual reallocation occurs (longer prompt exceeds buffer_capacity). When capacity is sufficient (same/shorter prompt), workspace buffer addresses are stable → CUDA graph persists across requests → no cuGraphExecDestroy per request.

TTFT distribution before (bimodal): P50=14.0ms, P95=~20ms, P99=~35ms, P99.9=43.6ms
TTFT distribution after (uniform): P50=13.2ms, P90=13.4ms, P95=13.7ms, P99=14.2ms, P99.9=41.4ms (first request only)
Decode: 149.5 tok/s (unchanged). c=4 aggregate: 315.4 tok/s (no regression).

**Historical context:** Concurrency scaling was the single dimension below A since Mar 11. PMAT-105 closed the gap from 51 C to 85 A-. PMAT-109 fixed the tail gap (86→100). Only scaling (85 A-) remains below A.

**PMAT-110 batch limit discovery (Mar 13):** CUDA_MAX_BATCH=8 in forjar config was silently capping throughput at c>8 (c=12 gave 636.8 with batch=8 vs 899.3 with batch=16). Updated forjar-yoga-realizr.yaml to CUDA_MAX_BATCH=16. Full scaling curve verified with batch=16: c=1 149.5, c=4 357.2, c=8 637.8, c=12 899.3, c=16 1139.5 tok/s (all short prompt, 0% errors, ITL 6.7→11.7ms).

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

#### Gap 1b: c=1 decode 0.92x vs llama.cpp (148.5 vs 161.7 tok/s) — PMAT-106

**DECODE_TIMING analysis (PMAT-106, Mar 13):** Per-step timing breakdown at M=1:
```
[GRAPH-TIMING] h2d=6µs launch=3µs wait=0µs argmax+sync=6690µs total=6700µs
[DECODE-TIMING] embed=0µs gpu=6700µs total=6700µs (149 tok/s)
```

**Finding: CPU overhead is negligible (9µs = 0.13%).** H2D copies (6µs), graph launch (3µs), and embedding lookup (0µs) add almost nothing. The entire 0.5ms gap to llama.cpp (6.7 vs 6.2ms = 7.5%) is GPU kernel execution time.

**BW analysis:**
- Total weight reads per step: ~986 MB (811 MB Q4K across 28 layers + 175 MB Q6K LmHead)
- realizr: 986 MB / 6.7ms = **147 GB/s** (57.4% of 256 GB/s theoretical)
- llama.cpp: 986 MB / 6.2ms = **159 GB/s** (62.1% of theoretical)
- Gap: llama.cpp achieves 4.7 percentage points more BW utilization

**Root cause candidates (ordered by likely impact):**
1. **Q6K GEMV kernel efficiency**: LmHead (175 MB, ~25% of decode time) uses Q6K MWV kernel. Q6K has ~37% lower BW utilization than Q4K HwDp4a, possibly from coalescing differences.
2. **Kernel launch spacing inside graph**: ~280 graph nodes with ~0.1µs scheduling gap each = ~28µs (0.4% of step, negligible).
3. **Attention overhead**: Flash decode with chunk_size=32 has fixed overhead per chunk.

**PMAT-107 FALSIFIED: Deferring cuGraphExecDestroy after first token emission worsened tail latency.**
Moving `clear_decode_graph()` from before first token emission to after it caused TTFT P99 regression: 42.5→60.9ms (+43%), P99.9: 43.6→611.6ms (14x worse). The cuGraphExecDestroy immediately followed by graph capture in the decode loop creates worse CUDA driver contention than the original position (driver cleanup + graph capture back-to-back). Reverted.

**Status:** MEASURED, no fix planned. At 0.92x parity, diminishing returns. Focus shifts to c>=4 where realizr already WINS.

#### ~~Gap 2: c=4 aggregate vs llama.cpp~~ PARITY (PMAT-105/110, Mar 13)

**PMAT-105 breakthrough + PMAT-110 correction: realizr 357.2 vs llama.cpp 365.8 = 0.98x (PARITY).** Previously 0.62x (pre-PMAT-105). Root cause was LmHead Q6K GEMV reading weights M times; FP8 cuBLASLt reads once. Single-line fix: route LmHead through `batched_gemv_or_gemm` instead of `batched_gemv_with_fallback`. PMAT-110 corrected llama.cpp comparison from `--parallel 8` to `--parallel 16` (matching CUDA_MAX_BATCH=16).

**Five-Whys (resolved):**

1. Why was realizr 0.62x? → LmHead (Q6K, 151,936×1536 = 175 MB) accounted for ~20% of decode ITL, reading weights M times.
2. Why M times? → `batched_gemv_with_fallback` always dispatches to Q6K batched GEMV, which runs M sequential/batched kernel launches reading weights each time.
3. Why not FP8? → `batched_gemv_with_fallback` lacks FP8 routing. Only `batched_gemv_or_gemm` has the full dispatch chain (W4A16 → FP8 → HGEMM → DP4A → cuBLAS).
4. Fix: Route LmHead through `batched_gemv_or_gemm`. At M=4, Q6K hits cuBLAS fallback (M>=4) → `cublas_prefill_gemm` → FP8 cuBLASLt (cc>=89). At M>=5, direct FP8 path fires.
5. Result: ITL dropped 25% (13.9→10.5ms). Q4K projections stay on DP4A (optimal at M=4). Q6K projections (LmHead, ffn_down, attn_v) now use FP8 cuBLASLt.

**Trajectory (actual, PMAT-072→105 all DONE):**
```
Pre-CB:         197.5 aggregate tok/s (0.33x vLLM)
+ PMAT-072:     197.5 (lock release — structural prerequisite) ✓
+ PMAT-073:     197.4 (mid-batch joins — no gain when all arrive together) ✓
+ PMAT-074:     216.4 (slot recycling — +10% heterogeneous traffic) ✓
+ PMAT-088a:    232.7 (iteration scheduler — +10.4%) ✓
+ PMAT-088c/d:  257.4 (batch recycling + multi-prompt — 84% of DP4A ceiling) ✓
+ PMAT-097:     259.7 (adaptive batch wait — tail latency fix, +1%) ✓
+ PMAT-054B:    162.1 ✗ FALSIFIED (W4A16 WMMA 1.78x slower than DP4A at M=4)
+ PMAT-105:     355.7 ✓ LmHead FP8 routing (+37%)
+ PMAT-110:     357.2 (corrected baseline, CUDA_MAX_BATCH=16)
llama.cpp:      365.8 tok/s (--parallel 16, corrected)
vLLM AWQ:       598.0 (W4A16 tensor core GEMM, fundamentally different arch)
```

#### Gap 3: c=4 aggregate 0.60x vs vLLM (355.7 vs 598.0 tok/s)

**Five-Whys (updated for PMAT-105):**

1. Why 1.68x gap to vLLM? → vLLM c=4 decode barely degrades from c=1 (150 vs 154, -2.6%). realizr per-slot decode drops 35% (95.8/4=24.0 vs 148.5).
2. Why does realizr degrade? → Q4K projections use DP4A GEMV (compute-bound at M>1). Q6K now uses FP8 cuBLASLt (BW-bound, scales well). But Q4K is 5/7 projections per layer.
3. Why doesn't vLLM degrade? → W4A16 Marlin reads INT4 (0.5 B/elem) + FP16 tensor cores. Memory-bound even at M=4. PagedAttention: <4% memory waste.
4. Why can't realizr use Marlin? → PMAT-054B FALSIFIED: WMMA 32×32 tiles waste 87.5% at M=4 (4/32 rows). Marlin needs M=32+ (vLLM accumulates via continuous batching+PagedAttention).
5. What remains? → EAGLE speculative (M=1 verify with multi-token acceptance), chunked prefill (TTFT improvement), or higher concurrency (already winning at c>=8).

**Root cause:** Q4K DP4A GEMV is compute-bound at M=4 for 5/7 projections per layer. PMAT-105 fixed Q6K projections (3/7: LmHead, ffn_down, attn_v) via FP8, improving c=4 from 0.43x to 0.60x vs vLLM. The remaining 0.40x gap is Q4K DP4A ceiling.

**PMAT-054B FALSIFIED: W4A16 WMMA is 1.78x SLOWER than DP4A at M=4.**
Pre-computed FP16 scales reduced gap from 3.5x (PMAT-091) to 1.78x but WMMA 32×32 tiles waste 87.5% compute at M=4.

**Remaining improvement paths:**
- ~~Higher concurrency where FP8 wins~~ ✅ DONE (PMAT-105: wins at ALL c>=4)
- ~~FP8 for all projections at M=4~~ **FALSIFIED** (PMAT-110: −5.3% c=4, 1.78× BW overhead > tensor core gain)
- Chunked prefill — interleave prefill with decode, reduce c=4 TTFT (36→~20ms)
- EAGLE speculative decoding (PMAT-009) — 2-3x effective throughput via multi-token verify

**PMAT-110 batch timing analysis (Mar 13, DECODE_TIMING=1):** M=4 batched decode is remarkably stable — 354.1-354.8ms per 32-token batch (±0.1%). Per-step ITL: 11.07ms. Prefill: 20.7-20.8ms for 116 tokens (5,765 tok/s). KV cache reuse (PAR-200) confirmed across all batches. No idle time between batches. The system operates at 92% of the theoretical DP4A ceiling (357 vs 386 tok/s).

#### Gap 4: TTFT scaling — 6.6× growth vs 3.3× (PMAT-111, Mar 13)

**realizr TTFT scales linearly with concurrency** (13.2→87.0ms at c=1→16, 6.6×). llama.cpp TTFT is sublinear (9.8→32.3ms, 3.3×). This is a structural consequence of realizr's batch-and-step scheduler architecture.

**Five-Whys (TTFT_TRACE, c=4):**

1. Why is c=4 TTFT 36ms (vs c=1 TTFT 13ms)? → Multi-prompt prefill (21ms) + first decode step (11ms) + HTTP/queue (4ms) = 36ms.
2. Why is multi-prompt prefill 21ms vs single-prompt 13ms? → 4×29=116 tokens total (FP8 cuBLASLt, weights read once, M_total=116). Sublinear: 2.3x time for 4x tokens.
3. Why does TTFT include one decode step? → Batch scheduler runs prefill for ALL prompts, then starts decode. First token requires prefill + 1 decode step.
4. Why doesn't llama.cpp need a decode step in TTFT? → Continuous batching interleaves new prefill with ongoing decode. A new request's prefill runs in the same batch step as existing requests' decode.
5. What would fix this? → (a) Pipeline parallelism: prefill next batch on stream B while current batch decodes on stream A. (b) Continuous batching: interleave prefill tokens with decode in same batch step. Both require multi-week realizr changes.

**Quantitative impact at c=4 (TTFT score: 84 → ~96 if fixed):**
- Current: 36ms TTFT, score 84
- Target: ~20ms TTFT (match llama.cpp), score ~96
- Composite impact: +1.8 points (84→96 × 0.15 weight). 78→80 B (not enough for A).

**Conclusion:** TTFT improvement adds only ~2 composite points at c=4 — insufficient for grade boundary change. The dominant blockers remain Aggregate (+6.0 max) and Decode (+6.0 max), both DP4A-limited. TTFT fix is low-priority relative to EAGLE speculative (PMAT-009) which could lift multiple dimensions simultaneously.

#### Gap 5: Prompt-profile sensitivity — FP8 prefill BW overhead exposed at medium prompts (PMAT-113, Mar 13)

**Hypothesis:** realizr's FP8 cuBLASLt prefill reads 1.78× more BW than llama.cpp's fused Q4K GEMM. At medium prompts (~102 tokens), the TTFT difference should amplify, potentially shifting the c=4 competitive balance.

**Benchmark (yoga 4060L, 1900MHz, 60s, `--prompt-profile medium`, `--warmup 5`, isolated):**

| | realizr short | realizr medium | Δ | llama.cpp short | llama.cpp medium | Δ |
|---|---|---|---|---|---|---|
| **c=1 Decode** | 149.5 | 148.7 | −0.5% | 160.2 | 159.9 | −0.2% |
| **c=1 TTFT** | 13.2ms | 18.6ms | +41% | 9.8ms | 10.3ms | **+5%** |
| **c=1 ITL** | 6.7ms | 6.7ms | 0% | 6.2ms | 6.3ms | +2% |
| **c=1 Aggregate** | 149.5 | 140.9 | −5.8% | 160.2 | 156.8 | −2.1% |
| **c=4 Decode** | 96.1 | 86.2 | −10.3% | ~89 | 92.7 | +4.1% |
| **c=4 TTFT** | 36.1ms | **75.8ms** | **+110%** | 19.6ms | **18.8ms** | **−4%** |
| **c=4 ITL** | 10.4ms | 11.6ms | +11.5% | 11.1ms | 10.8ms | −2.7% |
| **c=4 Aggregate** | 355.5 | **293.6** | **−17.4%** | 352.7 | **362.7** | **+2.8%** |
| **c=4 Errors** | 0% | 0% | — | 2.8% | 0.7% | improved |

**Key findings:**

1. **llama.cpp TTFT is prompt-length invariant.** c=4 TTFT barely changes: 19.6→18.8ms (−4%). llama.cpp's fused Q4K GEMM computes prefill in a single kernel launch that reads quantized weights directly — no dequantization to FP8/FP16 intermediate format. The 102-token prefill completes in ~18ms regardless of whether it's 29 or 102 tokens.

2. **realizr TTFT doubles with 3.5× more tokens.** c=4 TTFT: 36.1→75.8ms (+110%). FP8 cuBLASLt prefill reads weights in FP8 format (1 B/elem) vs Q4K (0.5625 B/elem), costing 1.78× more memory BW. At 4×102=408 input tokens, this BW overhead dominates. Prefill throughput: 1345 tok/s at medium (vs 5423 tok/s for llama.cpp, **4.0× gap**).

3. **At c=4 medium, llama.cpp retakes aggregate lead (362.7 vs 293.6, 1.24×).** The short-prompt parity (1.01×) breaks because realizr's longer prefill eats into generation time. llama.cpp's continuous batching also amortizes prefill across decode steps, preventing TTFT from blocking the pipeline.

4. **Decode rates tell opposite stories.** realizr c=4 decode drops 10.3% (96.1→86.2) with medium prompts — larger KV caches increase attention BW. llama.cpp c=4 decode *increases* 4.1% (89→92.7) — possibly from better cuBLAS GEMM tile utilization with longer sequences.

**Root cause:** FP8 cuBLASLt is a 2-step pipeline (convert Q4K→FP8 + FP8 GEMM) that reads 1.78× more weight BW than llama.cpp's fused 1-step Q4K GEMM. At short prompts (~29 tokens), prefill is <2% of total latency — the BW overhead is invisible. At medium prompts (~102 tokens), prefill becomes 17% of c=4 latency, exposing the gap. At long prompts (>500 tokens), this would be the dominant bottleneck.

**Fix path:** Fused Q4K dequant→GEMM kernel (reads Q4K weights once, dequantizes in registers, computes GEMM — matching llama.cpp's approach). This is the same fix needed for Gap 4 (TTFT scaling) and for c=12+ aggregate parity. Multi-week effort, but would close three gaps simultaneously.

**Production relevance:** Most API workloads use medium-to-long prompts (100-500 tokens). The short-prompt parity (1.01×) is not representative of production performance. **realizr's competitive position is prompt-length dependent** — dominates at c≥8 regardless of prompt length (FP8 decode wins), but loses c=4 aggregate lead as prompts grow.

**Scoring impact (probador llm score, c=4 medium):** llama.cpp **82 B+** vs realizr **67 C+**. Compare to short prompt: realizr 78 B > llama.cpp 70 B. The grade inversion is driven by TTFT (realizr 49/100 vs llama.cpp 100/100) and aggregate (73 vs 85). This means **realizr's c=4 grade advantage disappears at production-representative prompt lengths.**

**c=8 medium prompt verification (PMAT-113b):** realizr **still wins at c=8** even with medium prompts:

| Metric | realizr c=8 medium | llama.cpp c=8 medium | Ratio |
|--------|-------------------|---------------------|-------|
| Aggregate | **474.9** | 441.5 | **1.08x** |
| Decode | **79.3** | 55.9 | **1.42x** |
| ITL | **12.6ms** | 17.9ms | **0.70x** |
| TTFT | 147.7ms | 24.4ms | 6.1x (realizr worse) |
| Errors | **0%** | 2.0% | realizr better |

FP8 tensor core decode at M≥5 overcomes the FP8 prefill BW penalty. Compare c=8 short: realizr 631.9 vs llama.cpp 428.9 (1.47x). Medium narrows the gap (1.47x→1.08x) but does not close it. **The concurrency crossover holds across prompt profiles** — realizr's competitive advantage at c≥8 is structural (FP8 decode) not prompt-dependent.

**c=12 medium (PMAT-114):** realizr 627.3 vs llama.cpp **938.1** (**0.67x**). llama.cpp aggregate unchanged from short (927.3→938.1, +1.2%). realizr drops 30.2% (898.4→627.3).

**c=16 medium (PMAT-114 corrected):** realizr 749.7 vs llama.cpp **1045.3** (**0.72x**). ~~Initial run showed llama.cpp 504.5 (−51.3% "collapse") — FALSIFIED: measurement artifact from GPU resource contention after killing vLLM.~~ Verification: 1045.3 (+0.8% from short). **llama.cpp is prompt-length invariant at all concurrency levels** thanks to fused Q4K GEMM.

**Full prompt-profile sensitivity matrix (yoga RTX 4060L, 1900MHz, 60s, warmup 5s, medium ~102 tok):**

| c | realizr short | realizr medium | Δ | llama.cpp short | llama.cpp medium | Δ | Ratio (short) | Ratio (medium) |
|---|--------------|---------------|---|----------------|-----------------|---|--------------|---------------|
| 1 | 149.5 | 140.9 | −5.8% | 160.2 | 156.8 | −2.1% | 0.93x | 0.90x |
| 4 | 355.5 | 293.6 | **−17.4%** | 352.7 | 362.7 | +2.8% | **1.01x** | **0.81x** |
| **8** | **631.9** | **474.9** | −24.9% | 428.9 | 441.5 | +2.9% | **1.47x** | **1.08x** |
| 12 | 898.4 | 627.3 | −30.2% | 927.3 | 938.1 | +1.2% | 0.97x | **0.67x** |
| 16 | 1,139.8 | 749.7 | −34.2% | 1,037.1 | 1,045.3 | +0.8% | **1.10x** | **0.72x** |

**Key insight (PMAT-114 CORRECTED):** llama.cpp aggregate is prompt-length invariant at ALL concurrency levels (−2.1% to +2.8%). realizr aggregate drops monotonically (−5.8% to −34.2%). The fused Q4K GEMM makes llama.cpp immune to prompt length. realizr's FP8 cuBLASLt 2-step pipeline (convert + GEMM) imposes a per-token BW tax that compounds with M×prompt_len. **realizr only wins at c=8 medium** (1.08x, narrow) — the c=8 short dominance (1.47x) shrinks because TTFT eats into aggregate time. At c≥12 medium, llama.cpp dominates (0.67x, 0.72x) where short prompt showed parity/advantage.

**vLLM reference (medium prompt, W4A16 Marlin + PagedAttention):**

| c | vLLM medium | realizr medium | llama.cpp medium | realizr/vLLM | llama.cpp/vLLM |
|---|------------|---------------|-----------------|-------------|---------------|
| 4 | **551.0** | 293.6 | 362.7 | 0.53x | 0.66x |
| 8 | **1,023.2** | 474.9 | 441.5 | 0.46x | 0.43x |
| 12 | **1,418.5** | 627.3 | 938.1 | 0.44x | 0.66x |
| 16 | **1,778.5** | 749.7 | 1,045.3 | 0.42x | 0.59x |

vLLM dominates at all concurrencies. PMAT-121 fills c=12 medium gap: vLLM 1418.5 (realizr 0.44x, llama.cpp 0.66x). llama.cpp at c=16 medium (1045.3) is 0.59x of vLLM (1778.5) — competitive thanks to prompt-length invariance. realizr at 0.42x. The architectural gap: W4A16 Marlin (1-step dequant+GEMM) + PagedAttention (no per-batch prefill blocking) + continuous batching (interleaved prefill+decode).

**PMAT-116: Output length sensitivity (c=4 medium, 32 vs 128 output tokens):**

| Metric | realizr 32-tok | realizr 128-tok | llama.cpp 32-tok | llama.cpp 128-tok |
|--------|---------------|----------------|-----------------|------------------|
| Aggregate | 293.6 | 317.7 (+8.2%) | 362.7 | 372.7 (+2.8%) |
| Decode | 86.2 | 82.8 (−3.9%) | 92.7 | 93.8 (+1.2%) |
| TTFT | 75.8ms | 76.4ms | 18.8ms | 17.0ms |
| Ratio | **0.81x** | **0.85x** | — | — |

Longer output dilutes TTFT impact: TTFT is ~5% of latency at 128 tokens vs ~17% at 32. Ratio improves 0.81x→0.85x. realizr decode drops 3.9% at 128 tokens (larger KV cache → more attention BW). At 512+ output tokens, ratio would converge toward decode parity (~0.90x). **For production workloads with long responses, TTFT matters less — but decode rate still favors llama.cpp by ~10% at M=4.**

**c=8 medium 128-tok (PMAT-116b):** Ratio jumps **1.08x→1.25x** with 128 output tokens:

| Metric | realizr 32-tok | realizr 128-tok | llama.cpp 32-tok | llama.cpp 128-tok |
|--------|---------------|----------------|-----------------|------------------|
| Aggregate | 474.9 | 558.5 (+17.6%) | 441.5 | 446.6 (+1.2%) |
| Decode | 79.3 | 75.5 (−4.8%) | 55.9 | 56.2 (+0.5%) |
| ITL | 12.6ms | 13.2ms | 17.9ms | 17.8ms |
| **Ratio** | **1.08x** | **1.25x** | — | — |

realizr's aggregate grows +17.6% (TTFT diluted) while llama.cpp barely changes (+1.2%). **FP8 decode advantage at M≥5 becomes dominant when TTFT is amortized over more output tokens.** For code generation workloads (typically 100-500 output tokens), realizr's c≥8 advantage is much stronger than 32-token benchmarks suggest.

**PMAT-122: vLLM output length sensitivity (medium prompt, 32 vs 128 output tokens, Mar 14):**

| c | vLLM 32-tok | vLLM 128-tok | Δ | realizr 128-tok | realizr/vLLM 128-tok |
|---|-----------|------------|---|----------------|---------------------|
| 4 | 551.0 | **588.4** | +6.8% | 317.7 | 0.54x |
| 8 | 1,023.2 | **1,117.7** | +9.2% | 558.5 | 0.50x |
| 12 | 1,418.5 | **1,592.0** | +12.2% | — | — |
| 16 | 1,778.5 | **2,049.3** | +15.2% | — | — |

vLLM aggregate grows monotonically with output length — TTFT dilution effect grows with concurrency (+6.8% at c=4 to +15.2% at c=16). Decode rates stable: 150.3 (c=4), 145.1 (c=8), 138.4 (c=12), 134.0 (c=16) — virtually unchanged from 32-tok. **Output length does NOT close the realizr-vLLM gap** — vLLM also benefits from TTFT dilution. realizr/vLLM ratio: 0.54x at c=4, 0.50x at c=8 (128-tok) vs 0.53x/0.46x (32-tok). Both improve ~4%, net gap unchanged. The 2× architectural gap (W4A16 Marlin + PagedAttention) persists regardless of output length.

**PMAT-115: Theoretical fused Q4K→GEMM impact (medium prompt):**

If realizr had llama.cpp-equivalent TTFT (fused Q4K GEMM, 1-step dequant), decode rate unchanged:

| c | Current medium | Theoretical | Gain | vs llama.cpp medium |
|---|---------------|------------|------|-------------------|
| 1 | 140.9 | 146.3 | +3.8% | 0.93x (decode-limited) |
| 4 | 293.6 | 338.2 | **+15.2%** | 0.93x (decode-limited) |
| 8 | 474.9 | 616.4 | **+29.8%** | **1.40x (WINS)** |
| 12 | 627.3 | 917.8 | **+46.3%** | 0.98x (PARITY) |
| 16 | 749.7 | 1,181.4 | **+57.6%** | **1.13x (WINS)** |

The fused kernel would restore c=8 medium dominance (1.08x→1.40x) and c=12/16 competitiveness (0.67x→0.98x, 0.72x→1.13x). But c=1/4 remain at 0.93x — decode rate (not TTFT) is the bottleneck. **This quantifies the value of fused Q4K→GEMM: up to +57.6% aggregate at c=16 medium.** The fix simultaneously closes Gap 4 (TTFT scaling) and Gap 5 (prompt-profile sensitivity).

**Medium prompt scorecards (probador llm score):**

| c | vLLM | realizr | llama.cpp | realizr TTFT score |
|---|------|---------|-----------|-------------------|
| 1 | — | 95 A+ | **99 A+** | 96 |
| 4 | **99 A+** | 67 C+ | 78 B | **49** ← bottleneck |
| 8 | **97 A+** | **70 B** | 69 C+ | **25** ← bottleneck |
| 12 | — | 74 B | **84 B+** | **17** ← bottleneck |
| 16 | **93 A** | 73 B | **78 B** | **13** ← bottleneck |

**TTFT is the scoring bottleneck** at every c≥4 for medium prompts. realizr's decode and aggregate dimension scores are competitive (e.g., c=8: aggregate 90/100, decode 50/100), but TTFT (25-49/100 at c=4-8, 13-17 at c=12-16) drags the composite. llama.cpp wins at c=4/12/16 despite lower aggregate at c=8 because its TTFT (97-100/100) is nearly perfect. **The fused Q4K→GEMM fix would lift TTFT scores from 13-49 to ~90+, directly adding 5-12 composite points.**

**PMAT-117: Ollama characterization — best M=1 decode, worst TTFT (Mar 13):**

Ollama (v0.6+, systemd `OLLAMA_HOST=0.0.0.0:8082`) uses llama.cpp as its backend but adds its own serving layer. Serial request processing — no continuous batching, no request interleaving. Each request gets exclusive GPU access.

**4-runtime comparison — medium prompt (~102 tok, yoga 4060L, 1900MHz, 60s, warmup 5s, isolated):**

**c=1 medium:**

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Prefill tok/s |
|---------|----------------|-------------|---------------|-------------|--------------|
| llama.cpp | **156.8** | 159.9 | **10.3** | **6.3** | **5,423** |
| realizr | 140.9 | 148.7 | 18.6 | 6.7 | 1,345 |
| ollama | 123.4 | **164.5** | 70.1 | **6.1** | 1,456 |

Ollama has the **best M=1 decode** (164.5 tok/s, 3% above llama.cpp) from exclusive GPU access — zero scheduling overhead. But worst TTFT (70.1ms, 7× llama.cpp) from ollama's Go HTTP layer + serial model loading. Best ITL (6.1ms) confirms pure M=1 DP4A GEMV without batching overhead.

**c=4 medium:**

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Error % |
|---------|----------------|-------------|---------------|-------------|---------|
| **vLLM** | **551.0** | **149.4** | 24.8 | **6.7** | 0% |
| llama.cpp | 362.7 | 92.7 | **18.8** | 10.8 | 0.7% |
| realizr | 293.6 | 86.2 | 75.8 | 11.6 | 0% |
| ollama | 161.6 | 164.5 | 602.6 | 6.1 | 0% |

Ollama at c=4 exposes serial processing: aggregate barely improves from c=1 (161.6 vs 123.4, +31%) because requests queue behind each other. TTFT explodes to **602.6ms** (32× llama.cpp) — 3 queued requests × ~150ms each. Decode stays at 164.5 (same as c=1 — always M=1). **Ollama is the wrong architecture for concurrent workloads** — it serializes all requests, wasting GPU parallelism. Short-prompt c=4 TTFT is actually worse (612ms vs 602ms medium) — TTFT is dominated by serial queue wait, not prefill compute. **Ollama TTFT is prompt-length invariant but concurrency-dominated** (opposite of realizr, which is prompt-length sensitive but concurrency-optimized).

**Runtime architecture comparison:**

| Property | vLLM | realizr | llama.cpp | ollama |
|----------|------|---------|-----------|--------|
| Batching | Continuous (PagedAttention) | Batch-and-step | Continuous (cuBLAS) | Serial (no batching) |
| Decode quant | W4A16 Marlin (1-step) | DP4A M≤4, FP8 M≥5 | Fused Q4K GEMM | Same as llama.cpp |
| Prefill quant | W4A16 Marlin | FP8 cuBLASLt (2-step) | Fused Q4K GEMM | Same as llama.cpp |
| TTFT scaling | Sublinear (interleaved) | Linear (batch blocking) | Sublinear (interleaved) | Linear (serial queue) |
| Best regime | All concurrencies | c≥8 (FP8 decode) | c=1-4 (fused Q4K) | c=1 only (exclusive GPU) |

#### Production Workload Guide (PMAT-113 through PMAT-117 synthesis)

**Which runtime to deploy?** The answer depends on 3 workload parameters: concurrency (c), prompt length, and output length. No single runtime is optimal across all regimes.

**Decision matrix (yoga RTX 4060L, Q4_K_M, production recommendation):**

| Workload Profile | Best Runtime | Why | Score |
|-----------------|-------------|-----|-------|
| Interactive chat, c=1, short prompts | llama.cpp | Best decode (162), lowest TTFT (10ms) | 97 A+ |
| Interactive chat, c=1-4, medium prompts | llama.cpp | TTFT invariant to prompt length (18ms) | 78-99 |
| API server, c=4-7, mixed prompts | llama.cpp | Continuous batching + fused Q4K GEMM | 70-78 B |
| API server, c≥8, short prompts | **realizr** | FP8 tensor core decode (1.47x at c=8) | 70+ B |
| API server, c≥8, medium prompts, long output | **realizr** | FP8 decode dominates when TTFT diluted (1.25x at c=8 128-tok) | 70+ B |
| High-throughput API, any concurrency | **vLLM** | W4A16 Marlin + PagedAttention (2-3x all others) | 93-100 A+ |
| Single-user local IDE | ollama | Best M=1 decode (164.5), simplest setup | 78 B (c=1 only) |

**Key insights from PMAT-113 through PMAT-117:**

1. **Short-prompt benchmarks are misleading.** realizr shows 1.01× parity at c=4 with short prompts, but drops to 0.81× at medium prompts (PMAT-113). All production decisions should use medium prompt data.

2. **Output length favors realizr at c≥8.** TTFT dilution at 128+ output tokens improves realizr's ratio from 1.08× to 1.25× at c=8 (PMAT-116). Code generation workloads (100-500 output tokens) strongly favor realizr at high concurrency.

3. **llama.cpp is prompt-length invariant (PMAT-114).** Fused Q4K GEMM gives constant TTFT regardless of prompt length or concurrency. This architectural advantage is unmatched — realizr's FP8 2-step pipeline imposes a per-token BW tax that compounds with M×prompt_len.

4. **vLLM dominates everything (PMAT-120).** c=1→c=16 scaling: 11.9× (74% efficiency) vs realizr 7.6× (47.5%) vs llama.cpp 6.5× (40.4%). ITL: 6.5→7.4ms (+14%) vs realizr 6.7→11.7ms (+75%). The gap is architectural: W4A16 Marlin (fused 1-step quantized GEMM) + PagedAttention (dynamic KV cache) + continuous batching (interleaved prefill+decode). At c=16 short: **1832 tok/s** (1.61× realizr, 1.77× llama.cpp).

5. **Ollama is production-incompatible at c>1 (PMAT-117).** Serial processing wastes GPU parallelism. TTFT at c=4 is 602ms (32× llama.cpp). Only suitable for single-user interactive use where its M=1 decode (164.5 tok/s, best of all 4) and zero-config deployment provide value.

6. **The fused Q4K→GEMM kernel is the single highest-value optimization (PMAT-115).** Would close Gap 4 (TTFT scaling) and Gap 5 (prompt sensitivity) simultaneously, lifting realizr from 0.67-0.72× at c=12-16 medium to 0.98-1.13× (PARITY or WINS).

#### Gap 6: Long prompt (~311 tok) — realizr loses c=8 lead, llama.cpp context tradeoff (PMAT-118, Mar 13)

**Hypothesis:** At long prompts (~311 tokens), FP8 prefill BW overhead scales with M×prompt_len and may overwhelm FP8 decode advantage even at c=8, where realizr currently wins.

**Benchmark (yoga 4060L, 1900MHz, 60s, `--prompt-profile long`, `--warmup 5`, isolated):**

**Full 3-prompt-profile sensitivity matrix (aggregate tok/s):**

| c | realizr short | realizr med | realizr long | Δ short→long | llama.cpp short | llama.cpp med | llama.cpp long | Δ short→long | Ratio (short) | Ratio (med) | Ratio (long) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 149.5 | 140.9 | 128.0 | −14.4% | 160.2 | 156.8 | 156.3† | −2.4% | 0.93x | 0.90x | **0.82x** |
| 4 | 355.5 | 293.6 | **217.3** | **−38.9%** | 352.7 | 362.7 | 360.9† | +2.3% | 1.01x | 0.81x | **0.60x** |
| 8 | 631.9 | 474.9 | **324.2** | **−48.7%** | 428.9 | 441.5 | 421.7† | −1.7% | 1.47x | 1.08x | **0.77x** |

†llama.cpp long prompt data uses `--parallel 8` (512 tok/slot). `--parallel 16` (256 tok/slot) **cannot serve long prompts at any concurrency level** — even c=1 fails (311 tok > 256 slot limit). The `--parallel 8` config introduces a slight confound at c=8 (~5% lower throughput vs --parallel 16, per PMAT-101), but c=1 and c=4 data is effectively unconfounded (c < parallel).

**Detailed metrics (long prompt ~311 tok):**

| Metric | realizr c=4 | llama.cpp c=4 | realizr c=8 | llama.cpp c=8 |
|--------|------------|--------------|------------|--------------|
| Aggregate | 217.3 | **360.9** | 324.2 | **421.7** |
| Decode | **74.9** | 92.1 | **67.5** | 53.6 |
| TTFT | 174.7ms | **17.2ms** | 329.8ms | **30.2ms** |
| ITL | 13.4ms | **10.9ms** | **14.8ms** | 18.6ms |
| Errors | **0%** | 1.4% | **0%** | 2.6% |

**Key findings:**

1. **realizr loses the c=8 lead at long prompts.** Short: 1.47x (DOMINATES). Medium: 1.08x (wins). Long: **0.77x (LOSES)**. The c=8 crossover only holds for short/medium prompts. At ~311 tokens, FP8 prefill BW overhead (329.8ms TTFT = 45% of total latency) overwhelms the FP8 decode advantage.

2. **realizr aggregate degradation is monotonic and accelerating.** Short→long: c=4 drops 38.9%, c=8 drops 48.7%. The degradation scales super-linearly with concurrency because TTFT compounds with M (M×311 tokens of FP8 prefill). At c=16 long, realizr would likely be <0.50x vs llama.cpp.

3. **llama.cpp is prompt-length invariant even at long prompts.** c=4: 360.9 vs medium 362.7 (−0.5%). c=8: 421.7 (−4.5% vs medium, partly from --parallel 8 confound). The fused Q4K GEMM handles 311 tokens with zero marginal TTFT cost.

4. **llama.cpp parallel slots trade prompt capacity for concurrency (PMAT-118).** With `--parallel 16` and 4096 context: max prompt = 224 tokens (256 - 32 output). `--parallel 8`: max prompt = 480 tokens. **This is a hard per-slot limit** — even c=1 fails if prompt exceeds slot size. Production deployments serving code generation prompts (200-500 tokens) are forced to `--parallel 8` or lower, limiting maximum concurrent requests to 8. realizr has no such constraint (dynamic context allocation).

5. **realizr's reliability advantage persists.** 0% errors at all prompt lengths vs llama.cpp 1.4-2.6%. This matters more for long prompts where each failed request wastes more GPU time.

**TTFT scaling across prompt profiles (realizr, c=4):**

| Prompt | Tokens | TTFT P50 | Prefill tok/s | % of latency |
|--------|--------|----------|--------------|-------------|
| Short | ~29 | 36.1ms | ~5,423 | 6% |
| Medium | ~102 | 75.8ms | 1,345 | 14% |
| Long | ~311 | 174.7ms | 1,603 | 30% |

TTFT grows ~linearly with token count (36→76→175ms, ~0.5ms/token). Prefill throughput at long is actually higher than medium (1,603 vs 1,345) — FP8 cuBLASLt GEMM tile efficiency improves with larger M×S. But absolute time still dominates because 311×4=1244 input tokens is a large matrix.

**Updated production workload guide:**

| Prompt length | c=1-4 | c=8 | c=16 | Recommendation |
|--------------|-------|-----|------|---------------|
| Short (~29 tok) | llama.cpp 1.01x | **realizr 1.47x** | realizr 1.10x | realizr for API servers |
| Medium (~102 tok) | llama.cpp 1.24x | **realizr 1.08x** | llama.cpp 1.39x | Split: realizr c=8 only |
| Long (~311 tok) | llama.cpp 1.66x | llama.cpp 1.30x | llama.cpp ~2x (est) | **llama.cpp everywhere** |

**realizr only wins when TTFT is a small fraction of total latency** — short prompts at c≥8, or medium prompts at c=8 with long output. At long prompts, the FP8 prefill BW overhead is too large for FP8 decode to overcome.

**PMAT-119/121: vLLM prompt-length invariance confirmed at all c — fused quantized GEMM is the key (Mar 13-14):**

| c | vLLM short | vLLM medium | vLLM long | Δ short→long |
|---|-----------|------------|----------|-------------|
| 1 | 153.6 | — | 149.1 | −2.9% |
| 4 | 594.8 | 551.0 | 558.5 | −6.1% |
| 8 | 1,058.5 | 1,023.2 | 1,040.7 | −1.7% |
| 12 | 1,461.3 | **1,418.5** | **1,464.7** | **+0.2%** |
| 16 | 1,832.2 | 1,778.5 | **1,717.0** | **−6.3%** |

**PMAT-121 completes the vLLM prompt-profile matrix (c=12/16 × medium/long).** vLLM is near-invariant at all concurrency levels: max deviation −6.3% (c=16 long), range ±6%. Compare realizr: −34.2% to −48.7%. The c=16 long drop (−6.3%) is slightly larger than other levels — KV cache pressure at 16×311=4976 concurrent tokens and attention BW with longer sequences contribute. W4A16 Marlin kernel is a fused 1-step quantized GEMM (dequant + tensor core multiply in single kernel). **realizr is the ONLY runtime with prompt-length sensitivity** — the 2-step FP8 cuBLASLt pipeline (convert Q4K→FP8 + FP8 GEMM) is the sole cause.

**Full 3-runtime prompt sensitivity comparison (c=4):**

| Runtime | Short | Medium | Long | Δ short→long | Architecture |
|---------|-------|--------|------|-------------|-------------|
| vLLM | 594.8 | 551.0 | **558.5** | **−6.1%** | W4A16 Marlin (fused 1-step) |
| llama.cpp | 352.7 | 362.7 | **360.9** | **+2.3%** | Fused Q4K GEMM (1-step) |
| realizr | 355.5 | 293.6 | **217.3** | **−38.9%** | FP8 cuBLASLt (2-step) |

**c=8 long prompt, all 3 runtimes:**

| Runtime | Aggregate | Decode | TTFT P50 | ITL P50 | Errors |
|---------|-----------|--------|----------|---------|--------|
| **vLLM** | **1,040.7** | **144.5** | 32.0ms | **6.9ms** | **0%** |
| llama.cpp | 421.7 | 53.6 | **30.2ms** | 18.6ms | 2.6% |
| realizr | 324.2 | **67.5** | 329.8ms | **14.8ms** | **0%** |

**Scorecard across prompt profiles (probador llm score, composite):**

| c | Prompt | vLLM | realizr | llama.cpp | realizr TTFT |
|---|--------|------|---------|-----------|-------------|
| 1 | short | 100 A+ | 98 A+ | 97 A+ | 100 |
| 1 | medium | — | 95 A+ | 99 A+ | 96 |
| 4 | short | 98 A+ | 78 B | 70 B | 84 |
| 4 | medium | 99 A+ | 67 C+ | 78 B | 49 |
| **4** | **long** | **85 A-** | **49 D** | **67 C+** | **26** |
| 8 | medium | 97 A+ | 70 B | 69 C+ | 25 |
| **8** | **long** | **83 B+** | **52 C** | **57 C** | **16** |

**realizr drops from 78 B (c=4 short) to 49 D (c=4 long) — nearly 30 points lost from prompt length alone.** llama.cpp drops only 3 points (70→67). vLLM drops 13 points (98→85). realizr's TTFT score degrades from 84 to 16 as prompts grow. **At c=8 long, realizr (52 C) no longer beats llama.cpp (57 C)** — the short-prompt crossover advantage (c≥8) vanishes at long prompts.

**Architectural lesson:** Fused quantized GEMM (1-step) provides prompt-length invariance regardless of the specific quantization format (Q4K, W4A16). The 2-step pipeline (convert + GEMM) is the sole source of realizr's prompt sensitivity. **The fused Q4K→GEMM fix doesn't just improve performance — it eliminates an entire class of workload-dependent regression.**

**Fix path:** Same as Gap 5 — fused Q4K→GEMM kernel. Would eliminate the 1.78× BW overhead, making realizr prompt-length invariant like llama.cpp and vLLM. **Without this fix, realizr is limited to short/medium-prompt workloads at c≥8.**

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

**Updated trajectory (Mar 12, PMAT-097 complete):**
```
Continuous batching baseline:   216.4 aggregate tok/s (0.36x vLLM)
+ PMAT-088a (iter scheduler):   232.7 (+10.4%)
+ PMAT-088c (batch recycling):  256.3 (+24% from baseline)
+ PMAT-088d (multi-prompt):     257.4 (+0.4%, TTFT 60.7ms)
+ PMAT-093 (FP8 threshold):    258.5 (+0.4%, M<=4 DP4A / M>=5 FP8)
+ PMAT-095 (adaptive window):  246.9 (−5%, trades aggregate for zero c=1 TTFT penalty)
+ PMAT-097 (adaptive wait):    259.7 (+5%, tail latency fix, TTFT P99.9 42.8ms)
DP4A ceiling at M=4:            306 tok/s (85% reached)
vLLM AWQ (tensor core GEMM):   598.0 (fundamentally different arch)
```
**Remaining c=4 gap:** 259.7 vs 306 ceiling = 15% headroom. Sources:
(1) recycle stall (~16ms prefill blocks decode), (2) scheduling latency (~12ms reconnect+wait).
Chunked prefill (PMAT-088e) would help for long prompts but short 125-token prompts
already fit in one chunk. Q4K dequant is already at 5.8 insn/value (2.1x better than
llama.cpp) — no kernel optimization headroom remains within DP4A architecture.

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

### Tier Summary (Updated Mar 13 2026 — PMAT-110 corrected, 13 approaches falsified)

| Tier | Items | Status |
|------|-------|--------|
| T0: Decode parity | Fixes 1-6, GH-173/176, PMAT-040 (flash decode) | ✅ 0.94x llama.cpp (c=1) |
| T0: Prefill parity | PMAT-023/024/026, FP8 pipeline (PMAT-053b→086) | ✅ 1.29x llama.cpp (PASS < 2x) |
| T0: Continuous batching | PMAT-072→074, 088a-d, **105** (LmHead FP8) | ✅ **357.2 aggregate c=4 (0.98x PARITY, 78 B > llama.cpp 75 B)** |
| ~~T1: W4A16 tensor core~~ | ~~Marlin-style INT4→FP16 GEMM~~ | **FALSIFIED** (PMAT-091, 054B) — WMMA 87.5% waste at M=4 |
| T1: Chunked prefill | Interleave prefill with decode | Planned — reduces c=4 TTFT |
| ~~T2: GEMV optimization~~ | ~~Q4K dequant instruction reduction~~ | ~~DONE~~ (5.8 insn/value, 2.1x better than llama.cpp) |
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
| PMAT-090 | FP8 cuBLASLt batched decode (M≥2) | +4.3% c=4, **+33% c=8** | ✅ DONE (superseded by PMAT-093 threshold) |
| PMAT-091 | W4A16 coalesced WMMA GEMM | **FALSIFIED** (3.5x slower) | Custom PTX WMMA underperforms FP8/DP4A — needs double-buffer, k-tiling |
| PMAT-092 | Fused batched residual+RMSNorm | **FALSIFIED** (-5% c=4) | Restricts CTA parallelism (1,M) vs (6,M); residual fits L2, extra pass free |
| PMAT-093 | FP8 decode threshold M>=5 | **+10% c=4, +10% c=8** | ✅ DONE (DP4A fused at M<=4, FP8 tensor cores at M>=5) |
| PMAT-094 | Scheduler comparison (short prompt) | Iter sched −4% c=8 | ✅ DONE (cuda_batch_scheduler wins for short prompts) |
| PMAT-095 | Adaptive batch window | +4% c=4, 0ms c=1 TTFT | ✅ DONE (try_recv drain + 3ms adaptive wait for peers) |
| PMAT-096 | M=1 FP32 Q4K GEMV (bypass Q8) | **FALSIFIED** (1.8x slower) | MWV 82.8 vs HwDp4a 148.5 — DP4A compute advantage dominates at M=1 |
| PMAT-097 | Adaptive batch wait (recent_batch_gt1) | **TTFT P99.9: 1889→42.8ms** | ✅ DONE (44x tail reduction, zero c=1 overhead) |
| **PMAT-054B** | **W4A16 WMMA pre-computed scales** | **FALSIFIED** (1.78x slower at M=4) | Pre-computed FP16 scales cut gap from 3.5x (PMAT-091) to 1.78x, but WMMA 32×32 tiles waste 87.5% compute at M=4. DP4A GEMV remains optimal for M<=8. |
| PMAT-098 | Concurrency crossover analysis | realizr **1.08x at c=8** | ✅ DONE (crossover at c≈7.5, FP8 tensor cores vs DP4A ceiling) |
| **PMAT-099** | **Staggered prefill** | **FALSIFIED** (short: +114% TTFT, long: -14%) | Per-slot join overhead > batched prefill for short/medium. Long: +13.6% TTFT, -1.3% aggregate. |
| PMAT-100 | Prompt-profile characterization | TTFT ∝ M×prompt_len confirmed | ✅ DONE (short→long: c=4 aggregate drops 23.5%, ITL +23%) |
| PMAT-101 | Precise crossover characterization | **Crossover at c=7** (was c≈7.5) | ✅ DONE (ITL parity 16.7ms=16.7ms, realizr 1.11x at c=8) |
| **PMAT-102** | **Batched CUDA graph replay (PAR-121)** | **FALSIFIED** (−12% aggregate, +14.5% ITL) | Re-tested PAR-121 after PMAT-075/088b/045 fixes. Graph replay 16.6ms ITL vs eager 14.5ms. |
| **PMAT-103** | **High-concurrency scaling correction** | **Crossover narrowed to c=7-8 only** | PMAT-101 crossover confounded by --parallel 8. With --parallel 16: c=12 llama.cpp 882 vs realizr 605 (1.46x). c=16: 1039 vs 718 (1.45x). |
| **PMAT-104** | **Q4K GEMM kernels at M=12** | **ALL FALSIFIED** (−41% to −83%) | FP8 cuBLASLt confirmed as optimal backend at M>=5. Tested Q4K WMMA (−83%), fused Q4K scalar (−78%), L2 HGEMM (−48%), DP4A GEMM (−41%). Custom kernels can't compete with cuBLASLt. |
| **PMAT-105** | **LmHead FP8 routing** | **+48% c=12, WINS all c>=4** | ✅ DONE. Routed LmHead through batched_gemv_or_gemm instead of batched_gemv_with_fallback. FP8 reads weights once vs Q6K GEMV reads M times. ITL flat 10.4→11.7ms c=4-16. |
| **PMAT-106** | **c=1 decode timing analysis** | **0.5ms gap = GPU kernel BW** | ✅ MEASURED. CPU overhead 9µs (0.13%). Gap is Q6K/attention kernel BW utilization (57% vs 62%). |
| **PMAT-107** | **Defer cuGraphExecDestroy after TTFT** | **FALSIFIED** (+43% P99, 14x P99.9) | Moving graph clear after first token emission worsened tail: P99 42.5→60.9ms, P99.9 43.6→611.6ms. Driver contention between destroy+capture back-to-back. |
| **PMAT-109** | **Graph persistence (remove force_workspace_reinit)** | **TTFT P99: 35→14ms, tail 86→100** | ✅ DONE. CORRECTNESS-015 `force_workspace_reinit` was defeating PAR-200 workspace reuse, forcing graph destruction per request. Removal eliminates cuGraphExecDestroy from steady-state TTFT. P50: 14→13.2ms, P99: ~35→14.2ms. Bimodal TTFT eliminated. |
| **PMAT-110** | **FP8 for all projections at M=4 (BATCHED_DP4A=0)** | **FALSIFIED** (−5.3% c=4) | Disabling batched DP4A to force Q4K through FP8 cuBLASLt at M=4: aggregate 338.4 vs 357.2 (−5.3%), ITL 11.0 vs 10.4ms (+5.8%). DP4A fused gate+up confirmed optimal at M≤4. FP8 reads 1.78× more BW (1 B/elem vs Q4K 0.5625) — tensor core advantage doesn't compensate at M=4. Confirms PMAT-093 with post-PMAT-105 code. |
| **PMAT-111** | **TTFT scaling analysis (c=4 36ms breakdown)** | **TTFT 36→20ms (score 84→96)** | ✅ MEASURED. Structural: 21ms prefill + 11ms decode + 4ms overhead. Fix requires pipeline parallelism or continuous batching. +1.8 composite points (insufficient for grade change). |
| **PMAT-112** | **TTFT P99.9 tail: cold-start, not structural** | **c=4 P99.9: 328→46ms with warmup** | ✅ MEASURED. KV cache allocation (1792MB, PAR-119) causes one-time spike. With 5s warmup: c=4 tail 1.3x, c=8 1.0x, c=16 1.0x. Production-representative. |
| **PMAT-113** | **Prompt-profile sensitivity (medium ~102 tok)** | **c=4: 0.98x→0.81x with medium prompts** | ✅ MEASURED. FP8 prefill 1.78× BW overhead exposed at medium prompts. realizr TTFT doubles (36→76ms), llama.cpp unchanged (19→19ms). Short-prompt parity not representative of production workloads. Fix: fused Q4K→GEMM kernel. |
| **PMAT-117** | **Ollama characterization + Production Workload Guide** | **Best M=1 decode 164.5, serial: c=4 TTFT 602ms** | ✅ MEASURED. Ollama: best M=1 decode (164.5, +3% vs llama.cpp), best ITL (6.1ms), worst TTFT (602ms at c=4, serial queue). 4-runtime comparison at medium prompts. Production workload guide: no single runtime optimal — decision matrix by concurrency, prompt length, output length. |
| **PMAT-118** | **Long prompt sensitivity (~311 tok)** | **realizr LOSES c=8 lead: 0.77x** | ✅ MEASURED. Long prompts: c=4 0.60x (was 1.01x short), c=8 **0.77x** (was 1.47x short). FP8 prefill BW overwhelms decode advantage. llama.cpp prompt-invariant (c=4: 360.9 vs 362.7 medium). llama.cpp --parallel 16 can't serve 311-tok prompts (256/slot limit). realizr only wins at short/medium c≥8. |
| **PMAT-119** | **vLLM long prompt — prompt invariance confirmed** | **c=4 558.5, c=8 1040.7 (long ≈ medium)** | ✅ MEASURED. vLLM is prompt-invariant like llama.cpp: c=4 long 558.5 vs medium 551.0 (+1.4%), c=8 long 1040.7 vs medium 1023.2 (+1.7%). Marlin W4A16 is also fused 1-step quantized GEMM. **realizr is the ONLY runtime with prompt-length sensitivity** — 2-step FP8 pipeline is the sole cause. |
| **PMAT-120** | **vLLM full scaling curve (c=1-16 short)** | **1832 tok/s at c=16, 74% efficiency** | ✅ MEASURED. vLLM c=1→c=16: 153.6→1832.2 (11.9×, 74% efficiency). realizr: 7.6× (47.5%). llama.cpp: 6.5× (40.4%). vLLM ITL +14% c=1→c=16 (vs realizr +75%, llama.cpp +139%). PagedAttention + continuous batching gives dramatically better per-request latency stability under load. |
| **PMAT-114** | **Prompt-profile full matrix + falsification** | **llama.cpp prompt-invariant at ALL c** | ✅ CORRECTED. c=16 504.5 was artifact (GPU contention after vLLM kill). Verified: 1045.3 (+0.8% from short). Complete c=1-16 matrix: realizr only wins c=8 medium (1.08x). c=12: 0.67x, c=16: 0.72x. llama.cpp fused Q4K GEMM is prompt-length invariant (−2.1% to +2.8%). |
| **PMAT-121** | **vLLM complete prompt-profile matrix (c=1-16)** | **±6% invariant all c, all prompts** | ✅ MEASURED. Added c=12 medium (1418.5, −2.9%), c=12 long (1464.7, +0.2%), c=16 long (1717.0, −6.3%). Max deviation −6.3% (c=16 long) from KV cache pressure at 16×311=4976 tokens. vLLM prompt-invariance confirmed at all 5 concurrency levels. |
| **PMAT-122** | **vLLM output length sensitivity (128 vs 32 tok)** | **+6.8-15.2% agg, gap unchanged** | ✅ MEASURED. vLLM aggregate grows +6.8% (c=4) to +15.2% (c=16) with 128 vs 32 output tokens. TTFT dilution grows with concurrency. Decode rates unchanged. realizr/vLLM gap persists (0.50-0.54x at 128 tok vs 0.46-0.53x at 32 tok). Output length does NOT close the architectural gap. |
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
| 2.86.0 | 2026-03-14 | **PMAT-122: vLLM output length sensitivity — 2× gap persists.** vLLM aggregate grows +6.8% (c=4) to +15.2% (c=16) with 128 vs 32 output tokens. TTFT dilution grows with concurrency. Decode rates unchanged across output lengths. realizr/vLLM ratio at 128 tok: 0.54x (c=4), 0.50x (c=8) — virtually unchanged from 32 tok (0.53x, 0.46x). Both runtimes benefit from TTFT dilution at longer output, so net gap unchanged. Output length does NOT close the architectural gap between batch-and-step and continuous batching. vLLM at c=16 128-tok: **2049 tok/s** (new high watermark). |
| 2.85.0 | 2026-03-14 | **PMAT-121: Complete vLLM prompt-profile matrix (c=12/16 × medium/long).** c=12 medium: 1418.5 (−2.9% from short), c=12 long: 1464.7 (+0.2%), c=16 long: 1717.0 (−6.3%). vLLM prompt-invariance confirmed at all 5 concurrency levels (c=1-16). Max deviation ±6.3% vs realizr ±49%. Updated PMAT-119 table to 5×3 matrix, filled c=12 gap in vLLM medium reference table. |
| 2.84.0 | 2026-03-13 | **PMAT-120: vLLM full short-prompt scaling curve c=1-16.** c=8: 1058.5, c=12: 1461.3, c=16: 1832.2. Scaling: 11.9× (74% efficiency) vs realizr 7.6× (47.5%) vs llama.cpp 6.5× (40.4%). ITL stability: vLLM +14% (c=1→c=16), realizr +75%, llama.cpp +139%. Added 4-runtime scaling table to competition baselines section. Updated production workload guide with vLLM scaling data. Also added ollama c=1/c=4 short to baselines tables. |
| 2.83.0 | 2026-03-13 | **PMAT-119: vLLM prompt-length invariance confirmed — realizr is the ONLY prompt-sensitive runtime.** vLLM long prompt: c=4 558.5 (vs medium 551.0, +1.4%), c=8 1040.7 (vs medium 1023.2, +1.7%). Marlin W4A16 is also fused 1-step quantized GEMM. Both vLLM and llama.cpp are prompt-invariant (−2 to +2%). realizr drops 39-49% from short→long. The 2-step FP8 cuBLASLt pipeline is the sole cause. Fused Q4K→GEMM would eliminate an entire class of workload-dependent regression. Full 3-runtime c=4 and c=8 long tables added. |
| 2.82.0 | 2026-03-13 | **PMAT-118: Long prompt sensitivity — realizr loses c=8 lead.** Long prompt (~311 tok): realizr c=4 217.3 vs llama.cpp 360.9 (0.60x), c=8 324.2 vs 421.7 (0.77x). Aggregate degrades 38.9-48.7% vs short. c=8 crossover only holds for short/medium prompts — long prompts overwhelm FP8 decode advantage with prefill BW. llama.cpp `--parallel 16` can't serve 311-tok prompts (256/slot limit, must use --parallel 8). Added Gap 6, full 3-prompt-profile matrix, updated production workload guide. Also: ollama short-prompt baselines (164.6 decode c=1) added to competition tables. |
| 2.81.0 | 2026-03-13 | **PMAT-117: Ollama characterization + Production Workload Guide.** 4-runtime medium prompt comparison: ollama best M=1 decode (164.5 tok/s, +3% vs llama.cpp), best ITL (6.1ms), but worst TTFT (70ms c=1, 602ms c=4 — serial processing). Production-incompatible at c>1. Added production workload guide synthesizing PMAT-113→117: decision matrix by concurrency × prompt length × output length. No single runtime optimal — llama.cpp for c=1-7 medium, realizr for c≥8, vLLM for all concurrencies, ollama for single-user only. Runtime architecture comparison table (batching, quant, TTFT scaling, best regime). |
| 2.80.0 | 2026-03-13 | **PMAT-116b: c=8 medium output length — realizr advantage grows 1.08x→1.25x.** 128 output tokens at c=8 medium: realizr aggregate +17.6% (TTFT diluted over more decode steps), llama.cpp +1.2% (already TTFT-invariant). FP8 decode advantage at M≥5 becomes dominant when TTFT is amortized. For code gen workloads (100-500 output tokens), realizr's c≥8 advantage is much stronger than 32-token benchmarks suggest. |
| 2.79.0 | 2026-03-13 | **PMAT-116: Output length sensitivity.** c=4 medium with 128 vs 32 output tokens: ratio improves 0.81x→0.85x (TTFT share drops 17%→5% of latency). realizr decode drops 3.9% at 128 tokens (KV cache BW). llama.cpp decode stable (+1.2%). At 512+ tokens, ratio converges toward ~0.90x (decode parity). For long-response production workloads, TTFT matters less but decode gap persists (~10% at M=4). |
| 2.78.0 | 2026-03-13 | **PMAT-115: Theoretical fused Q4K→GEMM impact quantified.** If realizr had llama.cpp-equivalent TTFT (fused 1-step dequant): c=8 medium +29.8% (1.40x vs llama.cpp), c=12 +46.3% (0.98x PARITY), c=16 +57.6% (1.13x WINS). c=1/4 remain at 0.93x (decode-limited, not TTFT-limited). Fix simultaneously closes Gap 4 (TTFT scaling) and Gap 5 (prompt sensitivity). Removed stale "c=16 collapse" paragraph left from pre-falsification edit. |
| 2.77.0 | 2026-03-13 | **PMAT-114: c=16 medium "collapse" FALSIFIED + complete c=1-16 matrix.** Initial llama.cpp c=16 medium (504.5 tok/s, "−51.3% collapse") was measurement artifact from GPU contention after killing vLLM. Verification: 1045.3 (+0.8% from short). **llama.cpp is prompt-length invariant at ALL concurrency levels** (fused Q4K GEMM). Added c=12 medium: realizr 627.3 vs llama.cpp 938.1 (0.67x). Complete matrix: realizr only wins c=8 medium (1.08x, narrow). All other concurrencies: llama.cpp advantage grows with medium prompts. Fixed sensitivity matrix, scoring analysis, and executive summary. |
| 2.76.0 | 2026-03-13 | **PMAT-113 complete: Three-way comparison with vLLM medium prompts.** vLLM reference: c=4 551.0, c=8 1023.2, c=16 1778.5 aggregate (medium prompt). vLLM advantage grows with c: 1.88→2.37× vs realizr. All 0% errors. Gap is quantization (W4A16 vs Q4K) + scheduler (PagedAttention vs batch-and-step) + GEMM (Marlin vs FP8 cuBLASLt). c=16 scoring paradox documented: realizr 49% higher aggregate than llama.cpp but scores lower (TTFT penalty). Full 3-way sensitivity matrix added. |
| 2.75.0 | 2026-03-13 | **PMAT-113b complete: Full prompt-profile sensitivity matrix.** Added c=16 medium: realizr 749.7 vs llama.cpp 504.5 (1.49×, up from 1.10× at short). llama.cpp c=16 medium collapses −51.3% (1,632 KV entries overwhelm cuBLAS tiling). realizr advantage INCREASES at high c + medium prompts (1.10×→1.49×). Complete matrix: c=1 0.90×, c=4 0.81×, c=8 1.08×, c=16 1.49×. realizr aggregate drops monotonically with prompt length (−5.8% to −34.2%). llama.cpp aggregate is flat at c=4-8 but collapses at c=16. |
| 2.74.0 | 2026-03-13 | **PMAT-113b: c=8 medium prompt verification — crossover holds.** realizr still wins at c=8 with medium prompts: 474.9 vs 441.5 (1.08×, was 1.47× at short). FP8 decode advantage at M≥5 overcomes prefill BW penalty. Gap narrows (1.47×→1.08×) but does not close. Concurrency crossover is structural (FP8 decode), not prompt-dependent. llama.cpp c=8 medium: 55.9 decode tok/s (vs realizr 79.3), 17.9ms ITL (vs 12.6ms), 2.0% errors (vs 0%). |
| 2.73.0 | 2026-03-13 | **PMAT-113: Prompt-profile sensitivity — FP8 prefill BW overhead exposed at medium prompts.** Benchmark: realizr vs llama.cpp at medium (~102 tokens) vs short (~29 tokens), c=1 and c=4, yoga 4060L 1900MHz, 60s, warmup 5s. Key finding: llama.cpp TTFT is prompt-length invariant (c=4: 19.6→18.8ms, −4%) while realizr TTFT doubles (36.1→75.8ms, +110%). FP8 cuBLASLt reads 1.78× more weight BW than llama.cpp's fused Q4K GEMM. At c=4 medium, llama.cpp retakes aggregate lead (362.7 vs 293.6, 1.24×). Short-prompt parity (1.01×) not representative of production workloads. realizr c=4 decode drops 10.3% with medium prompts (larger KV attention BW). Fix path: fused Q4K dequant→GEMM (same fix needed for Gap 4 TTFT scaling + c=12+ aggregate). Added Gap 5 analysis section. |
| 2.72.0 | 2026-03-13 | **PMAT-112: TTFT P99.9 tail is cold-start, not structural.** With 5s warmup, c=4 TTFT P99.9 drops 7x (328→46ms, tail ratio 1.3x). Root cause: KV cache allocation (1792MB, PAR-119) on first batch. c=8 and c=16 tails perfect (1.0x) with warmup. Added warmup-representative TTFT tail table to scorecard section. Production servers with persistent processes should use warmup data. |
| 2.71.0 | 2026-03-13 | **PMAT-111: TTFT scaling analysis + fresh v2 benchmarks.** TTFT_TRACE reveals c=4 TTFT (36ms) = 21ms multi-prompt FP8 prefill + 11ms first decode + 4ms overhead. Structural: batch-and-step vs continuous batching. TTFT scales 6.6× (c=1→16) vs llama.cpp 3.3×. Fix adds only +1.8 composite points — DP4A ceiling remains the blocker. Fresh v2 benchmarks with both runtimes isolated: realizr c=4 355.5 vs llama.cpp 352.7 (1.01x PARITY). Updated scorecard: realizr 78 B vs llama.cpp 70 B (llama.cpp dropped from 75 B due to 2.8% error rate in fresh run). Added TTFT columns to concurrency scaling table. |
| 2.70.0 | 2026-03-13 | **Scorecard correction with --parallel 16 data.** c=1: 98 A+ (PMAT-109 tail 86→100). c=4: realizr 78 B > llama.cpp 75 B (corrected from 83 B+ vs 70 B with --parallel 8). Tier summary updated: c=4 0.98x PARITY (was 1.05x WINS). Trajectory table updated with corrected llama.cpp baseline (365.8 tok/s). |
| 2.69.0 | 2026-03-13 | **Corrected competitive comparison: llama.cpp --parallel 16.** Previous scaling table used llama.cpp --parallel 8 while realizr had CUDA_MAX_BATCH=16. With matched parallelism: c=4 is parity (357.2 vs 365.8, 0.98x), c=8 realizr dominates (637.8 vs 430.0, 1.48x), c=12 parity (899.3 vs 906.0, 0.99x), c=16 realizr wins (1139.5 vs 1000.4, 1.14x). Scoring: realizr 78 B vs llama.cpp 75 B at c=4 — realizr's 0% error rate (vs 1.3%) more than compensates for 2x TTFT gap. Quantitative scoring analysis: gap to 90 A requires +11.6 points, blocked by DP4A compute ceiling (Aggregate +6.0, Decode +6.0 if perfect). EAGLE speculative or W4A16 tensor core GEMM required for breakthrough. M=4 batch timing: 354.1-354.8ms per 32 tokens (±0.1% variance), 92% of theoretical DP4A ceiling. |
| 2.68.0 | 2026-03-13 | **PMAT-110 FALSIFIED: FP8 for all projections at M=4.** Hypothesis: disable batched DP4A (BATCHED_DP4A=0) to force Q4K projections through FP8 cuBLASLt at M=4, potentially reducing ITL. FP8 reads 1 B/elem vs Q4K 0.5625 B/elem (1.78× more BW) but uses tensor cores. **Result:** c=4 aggregate 338.4 vs 357.2 baseline (−5.3%), ITL 11.0 vs 10.4ms (+5.8%), decode 90.9 vs 96.1 (−5.4%). DP4A fused gate+up at M≤4 confirmed optimal — tensor core advantage doesn't compensate for 1.78× BW overhead. c=8 unaffected (637.5 vs 637.8, FP8 already fires at M≥5). Confirms PMAT-093 with post-PMAT-105 code. Also verified CUDA_MAX_BATCH=8 was bottlenecking c>8 throughput (c=12: 636.8 with batch=8 vs 899.3 with batch=16). Updated forjar-yoga-realizr.yaml to CUDA_MAX_BATCH=16. Full scaling curve verified: c=1 149.5, c=4 357.2, c=8 637.8, c=12 899.3, c=16 1139.5 tok/s (all short prompt, 0% errors). |
| 2.67.0 | 2026-03-13 | **PMAT-109: Graph persistence — c=1 tail 86→100, TTFT bimodal distribution ELIMINATED.** Removed `force_workspace_reinit()` from `run_prefill()` and `clear_decode_graph()` from `generate_gpu_resident_streaming()`. CORRECTNESS-015 was forcing workspace reallocation on every request, which invalidated CUDA decode graphs. PAR-200 in `init_prefill_workspace` already handles graph invalidation when actual reallocation occurs. When workspace capacity is sufficient (same/shorter prompt), buffer addresses are stable → graph persists across requests → no cuGraphExecDestroy. **Before (bimodal):** TTFT P50=14.0ms, P95≈20ms, P99≈35ms, P99.9=43.6ms. **After (uniform):** TTFT P50=13.2ms, P90=13.4ms, P95=13.7ms, P99=14.2ms, P99.9=41.4ms (first request). Decode 149.5 tok/s (unchanged). c=4 aggregate 315.4 (no regression). Tail score 86 A- → 100 A+. c=1 composite: 94 A → 98 A+. |
| 2.66.0 | 2026-03-13 | **Optimization exhaustion analysis.** Both remaining sub-A dimensions investigated to driver-level root causes. Scaling (85 → 90): 12 kernel approaches falsified, only EAGLE speculative remains. Tail c=1 (86 → 90): cuGraphExecDestroy variance is bimodal (95% at 20ms, 5% at 42ms), PMAT-107 falsified. Graph persistence blocked by CORRECTNESS-015. Updated scorecard gap analysis. |
| 2.65.0 | 2026-03-13 | **PMAT-107 FALSIFIED: Deferring cuGraphExecDestroy after first token emission worsens tail latency.** Hypothesis: moving `clear_decode_graph()` from before to after first token emission removes cuGraphExecDestroy from TTFT critical path. Benchmark (120s c=1, probador, streaming): TTFT P50 unchanged (20.0ms), but P99 42.5→60.9ms (+43%), P99.9 43.6→611.6ms (14x worse). Root cause: cuGraphExecDestroy immediately followed by graph capture in the decode loop creates worse CUDA driver contention (destroy+capture back-to-back on same stream) than the original position (destroy happens, then 20ms of prefill/emission buffer before capture). The existing PMAT-085 placement (after prefill, before emission) is optimal. Also established baseline tail characterization: c=1 TTFT distribution is bimodal — 95% at 20-21ms, 5% at 42-44ms. The 2x jump is from cuGraphExecDestroy cost variance. Reverted to original code with falsification notes. |
| 2.64.0 | 2026-03-13 | **PMAT-106: c=1 decode timing analysis — CPU overhead negligible (9µs, 0.13%).** Per-step DECODE_TIMING breakdown: h2d=6µs, graph launch=3µs, GPU execution+argmax+sync=6690µs, total=6700µs (149 tok/s). The 0.5ms gap to llama.cpp (6.7 vs 6.2ms, 0.92x) is entirely GPU kernel BW utilization (57.4% vs 62.1% of 256 GB/s). Root cause: Q6K GEMV kernel (LmHead, 175 MB, ~25% of decode) has ~37% lower BW efficiency than Q4K HwDp4a. No fix planned — 0.92x is diminishing returns. realizr WINS at all c>=4 (PMAT-105), making c=1 gap low priority. Updated Gap 1b analysis with measured data. |
| 2.63.0 | 2026-03-13 | **Scorecard v3.2.0 update (PMAT-105 impact).** Concurrency scaling dimension: 51 C (34.1%) → 85 A- (59.9%). c=4 composite: ~60 C+ → 83 B+. Gap 2 (c=4 vs llama.cpp) RESOLVED: 0.62x → 1.05x (WINS). Gap 3 (vs vLLM) narrowed: 0.35x → 0.60x. Updated five-whys analysis for both gaps. 9/9 dimensions at B+ or above. Remaining to A: scaling (85 → need 90, DP4A compute ceiling at M=4), tail c=1 (86 → need 90, occasional CUDA graph invalidation). |
| 2.62.0 | 2026-03-13 | **PMAT-105: LmHead FP8 routing — realizr WINS at ALL concurrency levels c>=4.** Single biggest optimization since FP8 prefill pipeline. LmHead (Q6K, 151,936×1536 = 175 MB weights) was using `batched_gemv_with_fallback` which always dispatches to Q6K batched GEMV — reads weights M times (once per sequence). One-line fix: route through `batched_gemv_or_gemm` to enable FP8 cuBLASLt dispatch at M>=5, reading FP8 weights ONCE (233 MB). **Results (yoga 4060L, 1900MHz, 60s, short prompt):** c=4: 270→356 aggregate (+31%), 13.9→10.4ms ITL (−25%). c=8: 456→632 (+38%), 16.2→11.3ms (−30%). c=12: 606→899 (+48%), 17.9→11.5ms (−36%). c=16: 718→1140 (+59%), 19.8→11.7ms (−41%). **ITL now nearly flat** from c=4 to c=16 (10.4→11.7ms, +12.5%). Previously 13.9→19.8ms (+42.4%). **Competitive landscape completely changed:** c=4 realizr 1.05x (was 0.80x), c=8 1.52x (was 1.10x), c=12 1.02x (was 0.69x), c=16 1.10x (was 0.69x). Realizr beats llama.cpp at c>=4 while maintaining 0% error rate (vs llama.cpp 0.6-1.6%). Root cause: LmHead is the largest projection (151,936 output dim, 20% of decode time at high M). Q6K batched GEMV at M=12 reads ~660 MB DRAM (3.8× with L2 sharing). FP8 cuBLASLt reads 233 MB once. The FP8 weight cache was already populated from prefill — zero warmup cost. |
| 2.61.0 | 2026-03-13 | **PMAT-104 FALSIFIED: All four Q4K GEMM kernels slower than FP8 cuBLASLt at M=12.** Tested hypothesis that custom Q4K→tensor core kernels could outperform FP8 cuBLASLt at high M (M=12) where tile efficiency improves (75% at M=12 vs 25% at M=4). Benchmarks (yoga 4060L, 1900MHz, 60s, c=12, short prompt, FP8_PREFILL=0): Q4K WMMA tensor cores −83% (93 vs 606 aggregate, 82ms ITL), fused Q4K scalar FMA −78% (123, 68ms ITL), L2-cached HGEMM (Q4K→FP16 dequant + cuBLAS) −48% (289, 39ms ITL), DP4A Q4K×Q8 GEMM −41% (328, 25ms ITL). **Root cause:** cuBLASLt's FP8 GEMM implementation is highly optimized (memory access scheduling, register tiling, tensor core utilization). Custom kernels cannot match this even with 1.78x less weight BW (Q4K 0.5625 B/elem vs FP8 1.0 B/elem). L2 HGEMM fails because FFN projections (27.4 MB FP16) exceed RTX 4060L's 24 MB L2 cache, falling back to DRAM at 3.5x more BW than FP8. **Conclusion:** FP8 cuBLASLt is confirmed as the optimal GEMM backend for realizr at M>=5. The c=12+ gap vs llama.cpp (0.69x) is rooted in llama.cpp's cuBLAS GEMM integration quality (fused Q4K dequant, 1 launch/projection vs realizr's 2-4 launches/projection), not in the choice of GEMM kernel. Incremental FP8 pipeline overhead reduction (fused absmax+convert, direct FP32 output) could save ~0.6ms/step (~3% ITL improvement), but won't close the 0.69x gap. |
| 2.60.0 | 2026-03-13 | **PMAT-103: High-concurrency scaling correction — realizr crossover NARROWED to c=7-8 only.** PMAT-101 found crossover at c=7 with llama.cpp `--parallel 8`. This was **confounded** — `--parallel 8` capped llama.cpp's batch size at M=8, creating an artificial ceiling at ~414 tok/s. Re-tested with `--parallel 16` (matching realizr `CUDA_MAX_BATCH=16`): llama.cpp scales to 882 tok/s at c=12 (1.46x realizr) and 1039 tok/s at c=16 (1.45x realizr). Realizr only wins at c=7-8 (narrow window, 1.10x at c=8). **Root cause:** llama.cpp uses cuBLAS GEMM with fused Q4K→FP16 dequant (1 kernel launch per projection). realizr uses FP8 cuBLASLt (3 launches per projection: absmax + E4M3 convert + GEMM). At M=12-16, cuBLAS FP16 tensor core tile efficiency increases (50% at M=8 → 75% at M=12 → 100% at M=16), causing llama.cpp ITL to *decrease* from 18.9ms (c=8) to 13.2ms (c=12). realizr ITL monotonically increases (16.2→17.9→19.8ms) as expected. Full table: c=1 0.99x, c=4 0.80x, c=8 **1.10x (WINS)**, c=12 0.69x, c=16 0.69x. **Quality advantage:** realizr 0% errors at all c=1-16 vs llama.cpp 0.6-1.6%. Also tested BATCHED_DP4A=0 (pure FP8 for all projections at M>=5): neutral at c=4 (+0.6%), c=5 (+1.0%), c=8 (−0.5%) — confirms hybrid (fused QKV DP4A + FP8 FFN) is already optimal. **Implication:** To compete at c=12+, realizr needs fused Q4K dequant+GEMM (like llama.cpp's mul_mat_q4_K) to eliminate per-projection FP8 conversion overhead. FP8 cuBLASLt's descriptor setup and conversion cost don't amortize well vs llama.cpp's tight CUDA kernel integration. |
| 2.59.0 | 2026-03-13 | **PMAT-102 FALSIFIED: Batched CUDA graph replay (PAR-121) is −12% slower than eager.** Re-tested PAR-121 batched CUDA graphs (previously disabled as "25% slower" since PMAT-056 era) after infrastructure fixes: PMAT-075 (stable buffer addresses across batches), PMAT-088b (async H2D copies, eliminated 2.8ms sync overhead), PMAT-045 (pre-upload device state before capture). Benchmark (yoga 4060L, 1900MHz, 60s, c=4, short prompt, BATCHED_GRAPH=1): 228.8 aggregate (baseline 259.7, −11.9%), 60.2 decode (65.6, −8.2%), 16.6ms ITL (14.5ms, +14.5%), 41.2ms TTFT (39.8ms, +3.5%). Graph capture confirmed successful (single capture, replayed for all subsequent batches). **Root cause analysis:** (1) Per-kernel CPU launch overhead was overestimated at ~7.4µs/kernel (3.6ms for 486 kernels). Actual overhead likely ~2-3µs/kernel, reducing graph savings to ~1-1.5ms. (2) Graph replay path has extra `stream.synchronize()` before argmax (8 kernels not part of captured graph), serializing GPU work that overlaps in eager path. (3) Async H2D copies (4 buffers: input, positions, seq_lens, workspace positions) add ~0.2ms overhead per step. Net effect: graph overhead exceeds savings. **c=4 ITL gap now fully characterized:** 14.5ms (realizr) vs 11.2ms (llama.cpp) = 3.3ms. Not addressable by CUDA graphs. Root cause is DP4A GEMV compute efficiency at M=4: realizr batches 4 independent M=1 GEMVs (4× thread blocks, each reading Q4K weights independently via L2), while llama.cpp uses cuBLAS GEMM with Q4K dequant (reads weights once, tensor core tile computation for all 4 rows). FP8 GEMM at M=4 was also tested (PMAT-093): −3% vs DP4A due to per-projection conversion overhead. HGEMM at M=4 (PMAT-062/088b): −13% vs DP4A due to 3.5× BW (FP16 vs Q4K). **Exhausted optimizations at M<=4:** CUDA graphs (PMAT-102), FP8 (PMAT-093), HGEMM (PMAT-062), W4A16 WMMA (PMAT-054B/091), fused residual+RMSNorm (PMAT-092). **Remaining path to close c=4 gap:** fused FP8 QKV conversion (convert input to E4M3 once, launch 3 FP8 GEMMs — saves 2 conversions/layer = 56 launches), or accept DP4A ceiling at M<=4 and focus on c>=7 where realizr already wins. |
| 2.58.0 | 2026-03-12 | **PMAT-101: Precise crossover at c=7 — ITL parity 16.7ms=16.7ms, realizr wins c>=7.** Full concurrency sweep c=1-8 (short prompt, yoga 4060L, 1900MHz, 30-60s, isolated, both runtimes fresh baselines). Crossover table: c=1 0.92x (148.9/162.0), c=4 0.78x (270.8/349.2), c=5 0.84x (313.9/375.4), c=6 0.91x (367.8/405.0), **c=7 1.01x** (410.2/404.5), **c=8 1.11x** (463.5/416.6). ITL crossover also at c=7: 16.7ms=16.7ms (realizr=llama.cpp). Below c=7: DP4A gap (realizr's fused QKV+gate+up DP4A is ~29% slower per step than llama.cpp at M=4). Above c=7: FP8 tensor cores (M>=5 threshold from PMAT-093) scale linearly while llama.cpp flattens. llama.cpp aggregate saturates at ~405 tok/s for c=6-7 (Q4K DP4A compute ceiling). realizr keeps climbing: c=7→c=8 +13% (410→464), limited only by FP8 tensor core throughput + memory bandwidth. **Updated crossover from c≈7.5 (PMAT-098) to c=7** — narrower sweep reveals exact parity. c=4 gap decomposition: ITL 14.5 vs 11.2ms (3.3ms or 29% overhead). Sources: no CUDA graphs in batched decode (PMAT-075 falsified, +0.8ms), kernel launch overhead in 4-projection GEMV path, attention KV cache scaling. |
| 2.57.0 | 2026-03-12 | **PMAT-100: Prompt-profile characterization — TTFT scales linearly with M×prompt_len, long prompts degrade c=4 by -23%.** First systematic benchmark across 3 prompt profiles (short ~23tok, medium ~125tok, long ~280tok) at c=1 and c=4 (yoga 4060L, 1900MHz, 60s, isolated). **c=1:** short 13.8ms/148.9, medium 19.7ms/148.1, long 41.4ms/146.6 (TTFT/decode). Decode rate barely affected by prompt length (148.9→146.6, -1.5%). TTFT scales ~linearly: 13.8→19.7→41.4ms tracks with token count. Prefill throughput increases with length (1669→5168→6764 tok/s) due to FP8 GEMM amortization. **c=4:** short 39.3ms/270.8, medium 79.5ms/244.6, long 179.1ms/207.2 (TTFT/aggregate). Aggregate drops -23.5% from short→long. Two causes: (1) TTFT increases 4.6x (39→179ms), eating into generation time. (2) ITL increases 23% (14.5→17.9ms) from larger KV caches in attention. **Staggered prefill (PMAT-099) re-tested across profiles:** short +114% worse TTFT, medium +4.8% better, long +13.6% better. Crossover at ~medium length. Even at long prompts, aggregate penalty (-1.3%) partially offsets TTFT gain. Default OFF confirmed correct — staggered only helps TTFT-sensitive long-prompt workloads. **Key insight:** For production API workloads with mixed prompt lengths, aggregate throughput is dominated by decode rate, not TTFT. Chunked prefill (H-CB5) would help by amortizing long prefills across decode steps without blocking the entire batch. |
| 2.56.0 | 2026-03-12 | **PMAT-099: Staggered prefill FALSIFIED for short prompts — default OFF, zero regression confirmed.** Hypothesis: prefilling only the first prompt in Phase 1 and joining remaining slots mid-batch (one-per-decode-step) reduces head-of-line blocking and improves TTFT at c>1. Implementation: `cuda_batch_scheduler.rs` staggered path — Phase 1 calls `batched_setup_and_prefill` with 1 prompt, Phase 2 decode loop joins pending slots via `add_slot_to_batch`. **Short prompt benchmarks (yoga 4060L, 1900MHz, 60s, `--prompt-profile short`):** Staggered c=4: TTFT 84.3ms, aggregate 239.5. Non-staggered c=4: TTFT 39.3ms, aggregate 270.8, 128 requests/0 errors (+4.3% vs PMAT-097 baseline 259.7). c=1: 148.9 decode, 13.8ms TTFT (matches PMAT-097). **Root cause:** Per-slot `add_slot_to_batch` overhead (~14ms each: workspace reinit + prefill + graph clear) exceeds batched multi-prompt prefill cost for short prompts. **Prompt confound resolved:** Earlier "19.9ms c=1 TTFT" was from probador's default prompt (~125 tokens), not the "short" profile (~23 tokens) used in prior baselines. With matching prompts, current binary matches or exceeds all PMAT-097 baselines. **Default changed to OFF** (env `STAGGERED_PREFILL=1` to enable). May win for long prompts (>500 tokens) where batched prefill cost scales with M×prompt_len. Also discovered: `--batch` CLI flag routes through CPU inference (broken) — `--gpu` is the correct CUDA path. NaN corruption not reproduced (0/92 at c=8). |
| 2.55.0 | 2026-03-12 | **PMAT-054B FALSIFIED: W4A16 WMMA pre-computed scales 1.78x SLOWER than DP4A at M=4.** Hypothesis: pre-computing `eff_scale = d * scale_int` and `eff_min = dmin * min_int` as FP16 on CPU would close the WMMA gap (3.5x in PMAT-091) by reducing GPU dequant from ~20 to ~5 insn/elem. Implementation: trueno `repack_q4k_w4a16()` repacks Q4K super-blocks into 2560B tiles (256B eff_scale + 256B eff_min + 2048B interleaved qs). realizr dispatch: W4A16 WMMA for M=2-8 Q4K projections, FP8 for M>=5 fallback, DP4A GEMV for all other. **Result:** Gap reduced from 3.5x to 1.78x (validating format design) but still SLOWER — c=4: 162.1 aggregate (DP4A baseline 259.7, -37.5%), decode 40.8 vs 72.4 (-43.6%), ITL 24.5 vs 13.8ms (+1.78x). **Root cause:** WMMA 32×32 output tiles waste 87.5% compute at M=4 (only 4 of 32 rows valid) + barrier+SHMEM overhead. **Why vLLM Marlin works:** PagedAttention accumulates M=32+ tokens even at low concurrency — Marlin's m16n8k16 MMA operates at high M where tile efficiency is >90%. realizr's batch scheduler produces M=c=4, too small for WMMA. **Remaining paths:** chunked prefill, EAGLE speculative decoding (PMAT-009), higher concurrency (c>=8 where FP8 already wins per PMAT-098). Also fixed: M<=8 upper bound on W4A16 dispatch to prevent intercepting FP8 prefill (TTFT regression 14→168ms). Three realizr commits: d0b358a (trueno format+kernel), 5ee2462 (realizr wiring), ba2b481 (M<=8 guard). |
| 2.54.0 | 2026-03-12 | **PMAT-098: Concurrency crossover analysis — realizr beats llama.cpp at c>=8.** Full 3-runtime comparison at c=1/4/5/6/7/8 (yoga 4060L, 1900MHz, 60s, streaming, short prompt). Fresh baselines: c=1 realizr 148.5 vs llama.cpp 161.7 (0.92x) vs vLLM 153.6. c=4: 259.7 vs 348.7 (0.74x) vs 594.8 (0.44x). **c=8: realizr 447.8 vs llama.cpp 414.3 (1.08x, WINS) vs vLLM 1150.6 (0.39x).** Crossover at c≈7.5. Mechanism: FP8 cuBLASLt tensor core GEMM at M>=5 (PMAT-093) scales throughput with M, while llama.cpp's Q4K GEMV hits compute ceiling at c=6-7 (~404 tok/s). Despite FP8 reading 1.78x more weight data than Q4K (1 vs 0.5625 B/elem), tensor core throughput dominates at M>=8. Key data: c=5 306.1 (0.80x), c=6 346.9 (0.86x), c=7 394.2 (0.97x), c=8 447.8 (1.08x). llama.cpp flat at c=6-7 (404 tok/s). **Implication:** realizr is positioned for production API workloads (c=8-32). The c=4 DP4A gap (0.74x) becomes irrelevant at production concurrency. Also confirmed: Q4K HwDp4a dequant already at 5.8 insn/value (2.1x better than llama.cpp's ~12 insn/value) via PMAT-029/033/039 — no kernel instruction optimization headroom remains. |
| 2.53.0 | 2026-03-12 | **PMAT-097: Adaptive batch wait fixes c=4 TTFT P99.9 tail latency (1889→42.8ms, 44× improvement).** Problem: PMAT-095's Phase 2 adaptive wait only triggers when `batch.len() > 1`, so singleton batches at c>1 start immediately without giving concurrent requests time to arrive. Root cause: at c=4, batch cycle is ~3.9s. All 4 clients complete simultaneously, send new requests near-simultaneously. First request arrives, scheduler recv()s it. Phase 1 `yield_now() + try_recv()` may not catch peers (HTTP latency jitter). Phase 2 skipped (batch.len()==1, old PMAT-095). Starts m=1 batch (1732ms), blocking the channel. Other 3 requests queue for ~1.7s → TTFT P99.9 = 1889ms (45.7× tail ratio). **Fix:** Track `recent_batch_gt1` — when the previous batch had `m > 1`, even singleton batches wait 2ms for peers (Phase 2b). At true c=1, `recent_batch_gt1` stays false → no wait → zero overhead. At sustained c=4, the flag is true → singletons wait 2ms → catch all 4 peers → consistent m=4 batches. **Cold-start note:** First batch at c=4 transition (from c=1) is still m=1 because `recent_batch_gt1` is false. This affects TTFT P99.9 in cold-start benchmarks (1881ms) but not sustained traffic (42.8ms, 1.1× tail ratio). **Benchmarks (yoga 4060L, 1900MHz, 60s, isolated):** c=1: 148.5 decode, 14.0ms TTFT (zero regression). c=4 warm: 259.7 aggregate, 39.8ms TTFT P50, **42.8ms TTFT P99.9** (1.1× tail ratio). c=4 cold-start: 255.7 aggregate, 39.3ms TTFT P50, 1881.8ms P99.9 (one-time transition artifact). realizr commit 591f561. |
| 2.52.0 | 2026-03-12 | **PMAT-096 FALSIFIED: M=1 FP32 Q4K GEMV (bypass Q8 quantization) is 1.8× SLOWER.** Hypothesis: At M=1 where GEMV is memory-bound (PMAT-029), Q8 activation quantization overhead (112 kernels/step) cancels DP4A compute benefit. Bypassing Q8 via FP32 MWV GEMV should improve c=1 decode. **Falsification:** MWV (FP32) = 82.8 tok/s vs HwDp4a = 148.5 tok/s (−44%). TTFT: 41.1ms (MWV) vs 13.8ms (HwDp4a). ITL: 12.1ms vs 6.7ms. **Root cause analysis:** M=1 Q4K GEMV is NOT purely memory-bound on RTX 4060L. Bandwidth floor (515MB weights / 192 GB/s = 2.68ms) vs actual decode (6.7ms HwDp4a, 12.1ms MWV) = 2.5× and 4.5× above floor respectively. DP4A's 4× int8 throughput keeps compute within the memory latency window; FP32 accumulation stalls waiting for memory, creating pipeline bubbles. Additionally, disabling HwDp4a loses fused gate+up+SwiGLU (requires Q4K HwDp4a), compounding the regression. **Key insight:** PMAT-029 "memory-bound" conclusion was about *within-kernel* instruction reduction (93 vs 108 insn/SB). Switching entire GEMV architecture (FP32 vs int8) changes the compute/memory balance — FP32 pushes the kernel partially compute-bound. The 4× instruction advantage of DP4A is NOT "hidden behind DRAM latency" — it's essential for keeping the pipeline full. c=1 decode gap (0.96× llama.cpp) likely comes from attention kernel efficiency or graph structure, not GEMV path. |
| 2.51.0 | 2026-03-12 | **PMAT-095: Adaptive batch window — zero c=1 TTFT penalty, auto-batching c>1.** Problem: Fixed batch window (CUDA_BATCH_WINDOW_MS=5) improves c=4 aggregate by +9% but adds 4ms to c=1 TTFT (14→18ms). Solution: Two-phase drain in cuda_batch_scheduler. Phase 1: zero-latency `yield_now() + try_recv()` (captures requests queued during GPU processing). Phase 2: if try_recv found ≥1 peer AND batch not full, 3ms timed wait for stragglers. At c=1, try_recv finds nothing → Phase 2 skipped → 0ms overhead. At c=4, try_recv finds 1-3 → 3ms wait → consistent M=4 batches. **Benchmarks:** c=1: 14.4ms TTFT (baseline 14.0ms, no penalty). c=4: 246.9 aggregate (+4% vs non-adaptive 237.5, 96% of window=5ms 258.5). c=8: 418.3 aggregate (94% of window=5ms 447.4). Forjar config updated: removed ITERATION_SCHEDULER=1 (PMAT-094 showed it's worse). Default config now auto-adapts without env vars. realizr commit ec56459. |
| 2.50.0 | 2026-03-12 | **PMAT-094: Scheduler comparison — cuda_batch_scheduler wins for short prompts.** Hypothesis: iteration scheduler (decode-maximal, Orca-style) improves aggregate throughput and TTFT via interleaved prefill+decode. Benchmarks (yoga 4060L, 1900MHz, 60s, PMAT-093 binary): **Iteration scheduler**: c=1: 148.5 decode (identical). c=4: 254.0 aggregate, 50.4ms TTFT. c=8: 428.2 aggregate, 61.4ms TTFT. **cuda_batch_scheduler**: c=4: 258.5 aggregate, 41.3ms TTFT. c=8: 447.4 aggregate, 54.9ms TTFT. **Result**: cuda_batch_scheduler +2% c=4, +4% c=8 aggregate; +18% better TTFT at c=4. **Root cause**: For short prompts (23 tokens), multi-prompt batched prefill (20ms for M=4) is cheaper than 3 sequential mid-batch joins (~14ms each = 42ms of decode stalls). Iteration scheduler's decode-maximal policy creates M=1→M=8 transitions with variable batch sizes, preventing consistent FP8 dispatch (M fluctuates through 1-8 during ramp-up). **Conclusion**: Keep cuda_batch_scheduler as default. Iteration scheduler may benefit long prompts (>512 tokens) where chunked prefill avoids generation stalls — needs testing with medium/long prompt profiles. |
| 2.49.0 | 2026-03-12 | **PMAT-093: FP8 decode threshold M>=5 — best-of-both-worlds CONFIRMED.** Hypothesis: FP8 per-projection overhead (absmax + convert + GEMM + rescale = 4 launches) cancels tensor core gains at M<=4 where DP4A fused paths have fewer launches. Threshold raised from M>=2 to M>=5. Code: `cublas_prefill.rs` guard changed `m >= 2` → `m >= 5`, `batched_ffn.rs` fused gate+up guard changed from `!self.gpu_profile.fp8_decode` to `!fp8_will_fire` (where `fp8_will_fire = fp8_decode && m >= 5`). **Benchmarks (yoga 4060L, 1900MHz, 60s, isolated):** c=1: 148.5 decode (no change, M=1 uses neither). c=4 (M=4, DP4A): 258.5 aggregate (+10% vs FP8-always 235.0). c=8 (M=8, FP8): 447.4 aggregate (+10% vs FP8-always 405.6, +36% vs DP4A-only 329.1). Key insight: at M=4, DP4A fused QKV+gate+up saves ~8 launches/layer vs FP8 individual projections (14 vs 22 launches). At M>=5, tensor core BW (256 TOPS FP8 vs ~50 TOPS DP4A equivalent) dominates the launch count difference. realizr commit 8784347. |
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
