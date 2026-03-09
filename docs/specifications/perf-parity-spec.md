# Performance Parity Specification

**Version:** 3.2.0
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

### Yoga Baselines — c=1 (single request, 60s, streaming, isolated, short prompt)

| Runtime | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Prefill tok/s | µs/layer |
|---------|-------------|---------------|--------------|---------------|----------|
| ollama | **150.7** | 72.1 | **6.6** | 319.0 | 236.9 |
| llama.cpp | 143.5 | **10.5** | 7.0 | **2,188.3** | 248.9 |
| **realizr** | 140.0 | 51.3 | 7.1 | 448.5 | 255.2 |

**Parity check (realizr vs llama.cpp):**
- Decode: 140.0 / 143.5 = **0.98x** (PASS — within 5%)
- TTFT: 51.3 / 10.5 = **4.9x** (FAIL — target <= 2x)
- ITL: 7.1 / 7.0 = **1.01x** (PASS — within 1.5x)
- Prefill: 448.5 / 2188.3 = **0.20x** (4.9x gap — drives TTFT)

### Yoga Baselines — c=4 (concurrent, 60s, streaming, isolated, short prompt)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| llama.cpp | **291.7** | 74.7 | **23.7** | **13.4** |
| **realizr** | 177.7 | 51.8 | 120.5 | 19.3 |
| ollama | 143.5 | **145.6** | 677.6 | 6.9 |

**Parity check (realizr vs llama.cpp, c=4):**
- Aggregate: 177.7 / 291.7 = **0.61x** (FAIL — target >= 3x c=1 = 420)
- ITL: 19.3 / 13.4 = **1.44x** (NEAR PASS — target <= 1.5x)
- TTFT: 120.5 / 23.7 = **5.1x** (FAIL — target <= 2x)

**Notes:**
- llama.cpp c=4: true parallel slots, 2.07x aggregate vs c=1
- realizr c=4: 1.27x aggregate vs c=1, improved from 1.19x via PMAT-046 batched bias/rope/Q6K
- ollama c=4: serializes requests, decode unchanged, TTFT balloons

### Known Issues

1. **c=4 ITL 1.44x gap** — 19.3ms vs llama.cpp 13.4ms. Batched KV scatter IS deployed (`realizr/src/cuda/executor/batch.rs`), reducing per-step launches from 224→56. Remaining gap: batch scheduler serialization (GH-141) — single `RwLock` blocks concurrent dispatch.
2. **c=4 aggregate 0.61x llama.cpp** — 177.7 vs 291.7. Root cause: batch scheduler serialization (realizr GH-141). The 10ms accumulation window + exclusive `model.write()` lock means most batches are m=1 under load. llama.cpp uses per-slot parallelism with independent KV caches and concurrent CUDA streams. Fix: Phase 1 adaptive window → Phase 2 interleaved prefill/decode → Phase 3 per-slot KV + concurrent streams.
3. **Prefill 4.9x TTFT gap** — realizr reads FP16 (2 B/elem) via HGEMM vs llama.cpp fused Q4K GEMM (0.5625 B/elem). Fused Q4K GEMM kernel exists in trueno (`fused_gemm.rs`) but has **critical scale extraction bug** for sub-blocks 4-7 (trueno GH-182) AND lacks tiling (reads weights M× for M rows). Requires both correctness fix + tiled implementation.
4. **CUDA graph capture blocked for batched decode** — per-layer KV cache pointer updates break static graph capture. Needs flat contiguous KV cache layout refactor. This is the true blocker for c=4 latency parity.
5. **FP16 cold-start eliminated (PMAT-037)** — eagerly warm FP16 weight cache at model init. No first-request penalty.
6. **PMAT-046: Batched bias/rope/Q6K** — Cut 334 kernel launches/step (784→450). ITL 20.0→19.3ms. Aggregate 166.6→177.7 tok/s (+7%).
7. **c=2 zero decode tokens** — realizr returns HTTP 200 with empty body at c=2 (batch scheduler bug). Discovered via `probador llm load --validate basic --concurrency 2`. Does not affect c=1 or c=4. Corrupts server state requiring restart.

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

---

## Five-Whys Root Cause Analysis

### TTFT 4.9x gap (c=1, realizr 51ms vs llama.cpp 10.5ms)

1. **Why is TTFT 4.9x worse?** Prefill reads FP16 weights (2 B/elem) via cuBLAS HGEMM.
2. **Why FP16?** The fused Q4K GEMM kernel in trueno was fundamentally broken (4 bugs — GH-182).
3. **Why was it broken?** Cross-row warp reduction (same as GH-180 NF4 bug), wrong scale extraction
   for sub-blocks 4-7, normalization by 63, and wrong qs byte/nibble mapping.
4. **Why can't the corrected kernel replace HGEMM now?** The corrected kernel uses serial
   accumulation (1-thread-per-output). Without shared-memory tiling, it re-reads Q4K weights
   M× for M output rows — slower than HGEMM for M>3.
5. **What's needed?** Tiled fused Q4K GEMM: load one super-block into shared memory, process
   all M rows from shared memory. Would close the 3.56x bandwidth gap.

**Root cause:** No tiled fused Q4K GEMM kernel exists yet.
**Status:** Correctness bugs fixed (trueno `2648239`). Tiled variant tracked as trueno GH-182.

### c=4 aggregate 0.61x gap (realizr 177.7 vs llama.cpp 291.7 tok/s)

1. **Why is c=4 aggregate 0.61x?** Batch scheduler serializes request processing.
2. **Why serialized?** `model.write()` lock held for entire generation (~330ms for m=4).
3. **Why held so long?** Sequential per-slot prefills (4×50ms=200ms) + decode loop (130ms).
4. **Why not release between phases?** CudaExecutor owns all GPU state — KV cache, graphs,
   scratch buffers — and requires exclusive access.
5. **Why not per-slot parallelism?** Architectural: weight buffers shared, KV caches per-slot
   but pointers updated per-layer (breaks CUDA graph capture). llama.cpp solves this with
   independent slots, separate CUDA streams, and paged attention.

**Root cause:** Architectural serialization in batch scheduler. Fix requires Phase 1-3 of realizr GH-141.
**Status:** Batched KV scatter deployed, batched bias/rope/Q6K deployed. Lock scope is the bottleneck.
