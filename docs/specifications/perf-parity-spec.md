# Performance Parity Specification

**Version:** 3.5.0
**Date:** 2026-03-09
**Status:** ACTIVE — single source of truth for all performance parity work

---

## Scope: What This Repo Does and Does Not Do

This repo (`qwen-coder-deploy`) is the **deployment harness and scoreboard**.

**This repo does:**
- Define forjar templates that deploy runtimes to target machines
- Run probador benchmarks against those deployments
- Store benchmark results as JSON
- Report whether realizr matches ollama/llama.cpp

**This repo does NOT:**
- Contain inference engine code (that's `../realizar`)
- Contain GPU kernels (that's `../trueno`)
- Contain the CLI server (that's `../aprender`)
- Debug or fix engine bugs (do that from the engine repo)

If a benchmark reveals a bug (e.g. wrong output at c=2), this repo's job is to
**observe and report** the failure. The fix happens in the engine repo. Then you
re-deploy and re-test from here.

---

## Goal

`apr serve` (via realizr inference engine) must match or beat ollama, llama.cpp,
and vLLM on decode throughput, TTFT, and ITL — across all model formats:

| Format | Runtime | Parity Target |
|--------|---------|---------------|
| GGUF Q4_K_M | apr serve | >= llama.cpp |
| APR native | apr serve | >= llama.cpp |
| SafeTensors | apr serve | >= ollama |
| GGUF Q4_K_M | apr serve | >= vLLM (c>=4) |

**One sentence:** Read weights once, generate tokens at memory-bandwidth speed,
for 1-32 concurrent requests, on consumer NVIDIA GPUs.

**vLLM baseline:** vLLM is the industry-standard serving engine. It excels at
continuous batching (c>=4) via PagedAttention. Compare at c=4+ where its
scheduling advantage matters. Deploy via `vllm serve` on same hardware.

---

## Test Target

**Primary:** `ssh yoga` (RTX 4060 Laptop, sm_89, 24 SMs, 8GB VRAM, x86_64)

All benchmarks run on yoga via SSH. No other test targets unless explicitly added.

---

## Test Method

**Forjar-only, setup/teardown, one component at a time.**

### Principles

1. **Forjar templates only** — every deploy/teardown is a `forjar apply -f <template>.yaml`
2. **One component at a time** — never run realizr + llama.cpp + ollama simultaneously
3. **Setup/teardown isolation** — deploy one runtime, benchmark it, tear it down, deploy next
4. **Deterministic** — locked GPU clocks, warmup period, fixed duration, fixed concurrency

### Workflow

```
# NOTE: Use IP not hostname — yoga only resolves via SSH config, not DNS
forjar apply -f forjar-yoga-realizr.yaml --yes --force
probador llm load --url http://192.168.50.38:8081 --duration 60 --concurrency 1 --warmup 5 --stream true
forjar apply -f forjar-yoga-teardown.yaml --yes

forjar apply -f forjar-yoga-llamacpp.yaml --yes --force
probador llm load --url http://192.168.50.38:8083 --duration 60 --concurrency 1 --warmup 5 --stream true
forjar apply -f forjar-yoga-teardown.yaml --yes

forjar apply -f forjar-yoga-ollama.yaml --yes --force
probador llm load --url http://192.168.50.38:8082 --model qwen2.5-coder:1.5b-instruct --duration 60 --concurrency 1 --warmup 5 --stream true
forjar apply -f forjar-yoga-teardown.yaml --yes
```

### Concurrency Levels

| Level | Description | Pass Criteria |
|-------|-------------|---------------|
| c=1 | Single request, streaming | Decode tok/s >= target |
| c=4 | Continuous batching | Aggregate tok/s >= 3x c=1 |
| c=8 | Stress test | No OOM, no correctness loss |

---

## Metrics

| Metric | Definition | Unit |
|--------|-----------|------|
| Decode tok/s | Tokens generated per second (excluding first token) | tok/s |
| TTFT P50 | Time to first token, median | ms |
| ITL P50 | Inter-token latency, median | ms |
| Prefill tok/s | Prompt processing rate | tok/s |

### Pass/Fail

A format passes parity when ALL of these hold at c=1:

- Decode tok/s >= llama.cpp decode tok/s
- TTFT P50 <= 2x llama.cpp TTFT P50
- ITL P50 <= 1.5x llama.cpp ITL P50

At c=4 (continuous batching):

- Aggregate decode tok/s >= 3x single-request decode tok/s
- No request returns empty or garbage output

---

## Model Under Test

**Qwen2.5-Coder-1.5B-Instruct** (Q4_K_M quantization)

- 28 transformer layers, hidden_dim=1536, num_heads=12, num_kv_heads=2
- head_dim=128, intermediate_dim=8960, vocab_size=151936
- RoPE: NEOX style (rope_type=2), theta=1000000.0
- GGUF file: ~1GB Q4_K_M

This model is small enough to fit in 8GB VRAM with headroom for KV caches
and workspace buffers for up to c=8 concurrent requests.

---

## Components

Each sub-specification covers one vertical slice. Max 500 lines each.

| # | Component | File | Status |
|---|-----------|------|--------|
| 1 | GGUF decode (single request) | [components/gguf-decode.md](components/gguf-decode.md) | Active |
| 2 | GGUF prefill | [components/gguf-prefill.md](components/gguf-prefill.md) | Active |
| 3 | Continuous batching (c=2..32) | [components/continuous-batching.md](components/continuous-batching.md) | Active |
| 4 | APR native format | [components/apr-native.md](components/apr-native.md) | Planned |
| 5 | SafeTensors format | [components/safetensors.md](components/safetensors.md) | Planned |
| 6 | Forjar templates (yoga) | [components/forjar-yoga.md](components/forjar-yoga.md) | Active |
| 7 | Improved LLM load testing | [components/improved-llm-load-testing.md](components/improved-llm-load-testing.md) | Implemented (dogfooded) |

---

## Architecture Context

```
yoga (RTX 4060 Laptop, 192.168.50.38)
├── apr serve :8081  (realizr engine, GGUF/APR/SafeTensors)
├── ollama    :8082  (baseline, GGUF only)
├── llama.cpp :8083  (baseline, GGUF only)
├── vllm      :8084  (baseline, GGUF/SafeTensors, c>=4 batching reference)
└── Benchmarks via probador from dev machine
```

The inference stack:

```
apr-cli (HTTP server, /v1/chat/completions)
  └── realizr (inference engine)
        ├── GGUF loader → OwnedQuantizedModelCuda
        ├── CudaExecutor (GPU kernels via trueno-gpu PTX)
        │     ├── GEMV decode (Q4K/Q6K, DP4A, batched)
        │     ├── KV cache (GPU-resident, per-slot for batching)
        │     ├── Attention (incremental, flash decoding)
        │     └── CUDA graphs (launch overhead elimination)
        └── Batch scheduler (continuous batching, slot-based)
```

---

## Current State (2026-03-09)

### Yoga Baselines — c=1 (single request, 60s, streaming, isolated, ~102 prompt tokens)

| Runtime | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Prefill tok/s |
|---------|-------------|---------------|--------------|---------------|
| **realizr** | **138.3** | 64.5 | **7.2** | 1,580.4 |
| llama.cpp | 134.7 | **17.4** | 7.4 | **5,850.6** |

**Parity check (realizr vs llama.cpp):**
- Decode: 138.3 / 134.7 = **1.03x** (PASS — realizr 3% faster)
- ITL: 7.2 / 7.4 = **0.97x** (PASS — realizr 3% faster)
- TTFT: 64.5 / 17.4 = **3.7x** (FAIL — target <= 2x, improved from 4.5x via PMAT-050)
- Prefill: 1580.4 / 5850.6 = **0.27x** (3.7x gap — drives TTFT)

### Yoga Baselines — c=4 (concurrent, 60s, streaming, isolated, ~102 prompt tokens)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| llama.cpp | **310.0** | **77.3** | **21.5** | **12.9** |
| **realizr** | 186.6 | 51.3 | 260.4 | 19.5 |

**Parity check (realizr vs llama.cpp, c=4):**
- Aggregate: 186.6 / 310.0 = **0.60x** (FAIL — target >= 3x c=1 = 420)
- ITL: 19.5 / 12.9 = **1.51x** (NEAR PASS — target <= 1.5x)
- TTFT: 260.4 / 21.5 = **12.1x** (FAIL — target <= 2x)

**Notes:**
- llama.cpp c=4: true parallel slots, 2.30x aggregate vs c=1
- realizr c=4: 1.34x aggregate vs c=1, improved from 1.19x via PMAT-046 batched bias/rope/Q6K
- Decode and ITL at parity (c=1). Only TTFT and c=4 aggregate remain.

### Known Issues

1. **c=4 ITL 1.51x gap** — 19.5ms vs llama.cpp 12.9ms. Batched KV scatter deployed, 334 launches cut (PMAT-046). Remaining gap: batch scheduler serialization (GH-141) — single `model.write()` lock means most batches are m=1.
2. **c=4 aggregate 0.60x llama.cpp** — 186.6 vs 310.0. Root cause: batch scheduler serialization (GH-141). 10ms accumulation window + exclusive lock. llama.cpp uses per-slot parallelism with independent CUDA streams.
3. **TTFT 3.7x gap** — 64.5ms vs 17.4ms at c=1. PMAT-050 graph capture reduced from 4.5x (78ms). Root cause now **GPU compute time**, not CPU launch overhead. cuBLAS HGEMM reads FP16 (2 B/elem) vs llama.cpp fused Q4K (0.5625 B/elem) = 3.56x bandwidth gap. Graph capture eliminated ~21ms CPU overhead; remaining 64.5ms is GPU-bound.
4. **Fused Q4K GEMM (GH-182) validated** — works correctly but 16x SLOWER than cuBLAS HGEMM at M=125 on RTX 4060 (sm_89). Not viable for TTFT improvement. Needs complete tiling rewrite.
5. **CUDA graph capture blocked for batched decode** — per-layer KV cache pointer updates break static graph capture. Needs flat contiguous KV cache layout. True blocker for c=4 latency parity.
6. **FP16 cold-start eliminated (PMAT-037)** — eagerly warm FP16 weight cache at model init. No first-request penalty.
7. **PMAT-046: Batched bias/rope/Q6K** — Cut 334 kernel launches/step (784→450). ITL 20.0→19.3ms. Aggregate 166.6→186.6 tok/s (+12%).
8. **c=2 zero decode tokens** — batch scheduler bug at c=2. Does not affect c=1 or c=4.
9. **Parity gate false positive** — GPU parity check fails (cosine sim -0.28) but GPU output is correct. CPU reference diverges. Set `SKIP_PARITY_GATE=1` to bypass.
10. **PMAT-050: Prefill CUDA graph capture** — captures 728 kernel launches into 1 graph. TTFT 85.2→64.5ms (24% improvement). First request for each unique S captures graph (~143ms one-time cost). Subsequent requests replay. Disable with `PREFILL_GRAPH=0`.

---

## Historical Lineage

This spec supersedes all previous specifications across both repos.
The journey from 1.2 tok/s (Dec 2025) to decode parity with llama.cpp
on Jetson Orin (Mar 2026) is documented in `../realizar/docs/specifications/`
as historical reference.

Key milestones:
- Coalesced GEMV: 1.2 → 45 tok/s (37x)
- GPU-resident KV cache: eliminated PCIe round-trips
- DP4A Q4K kernel: integer dot product on sm_75+
- CUDA graph capture: 280 kernel launches → 1 graph launch
- Flash Decoding: 2x attention throughput for long sequences
- HGEMM prefill: tensor core prefill via cached FP16 weights
- PMAT-037: Eager FP16 cache warmup (eliminate 303ms cold-start)
- PMAT-037b: cuBLAS HGEMM prefill in batched path (c=4 TTFT 854→150ms)
- Batched GEMV: weight sharing across M concurrent requests
- PMAT-046: Batched bias broadcast, NEOX RoPE, Q6K GEMV — cut 334 launches/step
- GH-182: Fused Q4K GEMM rewrite — fixed 4 bugs (cross-row reduction, scale extraction,
  normalization, qs mapping). Correct serial accumulation, needs tiling for performance.
- GH-182: Tiled fused Q4K GEMM — TILE_M weight reuse, reads Q4K directly (0.5625 B/elem).
  Wired into realizr prefill via FUSED_Q4K_PREFILL=1. Validated correct but 16x slower than
  cuBLAS HGEMM at M=125 on sm_89 (needs complete tiling rewrite).
- PMAT-050: Prefill CUDA graph capture — 728 kernel launches → 1 graph. TTFT 85→64.5ms (24%).
  Root cause shifts from CPU launch overhead to GPU compute (FP16 bandwidth 3.56x vs Q4K).

---

## Five-Whys Root Cause Analysis

### TTFT 3.7x gap (c=1, realizr 64.5ms vs llama.cpp 17ms)

1. **Why is TTFT 3.7x worse?** 64.5ms for ~125-token prefill across 28 layers.
2. **Why 64.5ms?** GPU compute time (~58ms) + graph launch overhead (~6ms).
   PMAT-050 graph capture eliminated 21ms CPU launch overhead (was 85ms without graph).
3. **Why 58ms GPU compute?** cuBLAS HGEMM reads FP16 weights (2 B/elem). Per-layer GEMM
   totals ~1.6ms for 7 projections, plus attention (~0.4ms) = ~2.0ms × 28 layers = 56ms.
4. **Why not use quantized weights directly?** Fused Q4K GEMM (GH-182) is 16x SLOWER than
   cuBLAS HGEMM at M=125 on sm_89. cuBLAS HGEMM leverages tensor cores; fused Q4K uses CUDA
   cores with serial accumulation. Would need complete tiling rewrite with shared memory.
5. **Why is llama.cpp 3.7x faster?** Fused Q4K/Q6K GEMM reads 0.5625/0.66 B/elem (3.56x less
   bandwidth), hand-optimized with CUDA core tiling + warp-level reduction. ~168 launches
   at ~0.1ms each = 17ms.

**Root cause:** FP16 bandwidth overhead (3.56x more data read) + cuBLAS per-call overhead
within the graph (still ~0.08ms per cuBLAS call even with graph replay).
**Fix needed:** Fused Q4K/Q6K GEMM that reads quantized weights directly with proper tiling
(shared memory, warp-level reduction, no serial accumulation). This is a major kernel
engineering effort — effectively building llama.cpp's mul_mat_q4_K equivalent.
**Status:** Graph capture deployed (PMAT-050). CPU overhead eliminated. GPU compute is the bottleneck.

### c=4 aggregate 0.60x gap (realizr 186.6 vs llama.cpp 310.0 tok/s)

1. **Why is c=4 aggregate 0.60x?** Batch scheduler serializes request processing.
2. **Why serialized?** `model.write()` lock held for entire generation (~2.7s for batch of 4).
3. **Why held so long?** Sequential per-slot prefills (4×260ms=1040ms) + decode loop (1660ms).
4. **Why not release between phases?** CudaExecutor owns all GPU state — KV cache, graphs,
   scratch buffers — and requires exclusive access.
5. **Why not per-slot parallelism?** Architectural: weight buffers shared, KV caches per-slot
   but pointers updated per-layer (breaks CUDA graph capture). llama.cpp solves this with
   independent slots, separate CUDA streams, and paged attention.

**Root cause:** Architectural serialization in batch scheduler. Fix requires Phase 1-3 of realizr GH-141.
**Status:** Batched KV scatter deployed, batched bias/rope/Q6K deployed. Lock scope is the bottleneck.
