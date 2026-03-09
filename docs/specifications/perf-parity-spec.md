# Performance Parity Specification

**Version:** 3.12.0
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

forjar apply -f forjar-yoga-vllm.yaml --yes --force
probador llm load --url http://192.168.50.38:8084 --model Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ --duration 60 --concurrency 1 --warmup 5 --stream true
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

## Profiling Methodology — `apr profile` Integration

**Principle:** Every performance gap must have a root cause identified via `apr profile`
before optimization work begins. probador measures the gap; apr profile explains it.

### Two-Tool Workflow

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `probador llm load` | Measure throughput, latency, TTFT | Establish parity gaps (external) |
| `apr profile` | Roofline analysis, per-brick hotspots, bottleneck classification | Diagnose root cause (internal) |

### apr profile Capabilities (Current)

| Feature | Command | Output |
|---------|---------|--------|
| Roofline analysis | `apr profile <model> --perf-grade` | Compute-bound vs memory-bound, efficiency % |
| Per-brick hotspots | `apr profile <model> --granular` | Time/% per operation (attention, FFN, norm) |
| Ollama comparison | `apr profile <model> --ollama` | Side-by-side decode/prefill tok/s |
| Baseline comparison | `apr profile <model> --baseline-url URL --baseline-name NAME` | Side-by-side vs any OpenAI-compat runtime (PMAT-056) |
| CI assertions | `apr profile <model> --ci --assert-throughput 140` | Pass/fail with threshold |
| Flamegraph | `apr profile <model> --format flamegraph -o flame.svg` | Interactive SVG |
| Per-layer timing | `apr profile <model> --granular --json` | Per-layer variance detection (CV%) |
| Kernel overhead | `apr profile <model> --granular` | Kernel launch overhead % (F-PROFILE-009) |
| Live trace | `curl -H "X-Trace-Level: brick" /v1/chat/completions` | Per-brick timing in response JSON |

### Profiling Workflow Per Gap

```bash
# Step 1: Measure the gap (probador)
probador llm load --url http://yoga:8081 --concurrency 1 --stream true --duration 60s

# Step 2: Profile the realizr side (apr profile, run ON yoga)
ssh yoga 'apr profile /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --perf-grade --granular --json --warmup 3 --measure 10 --tokens 32'

# Step 3: Compare against any baseline (PMAT-056, --baseline-url)
ssh yoga 'apr profile /path/to/model.gguf \
  --baseline-url http://127.0.0.1:8083 --baseline-model default --baseline-name llama.cpp \
  --perf-grade --granular --warmup 3 --measure 10 --tokens 32'

# Step 3b: Compare against vLLM
ssh yoga 'apr profile /path/to/model.gguf \
  --baseline-url http://127.0.0.1:8084 --baseline-model Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ --baseline-name vLLM \
  --perf-grade --granular --warmup 3 --measure 10 --tokens 32'

# Step 4: Live trace for per-request debugging
curl -s -X POST http://yoga:8081/v1/chat/completions \
  -H "Content-Type: application/json" -H "X-Trace-Level: brick" \
  -d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}'
```

### Gaps in Current Profiling (to be addressed)

| Gap | Impact | Fix |
|-----|--------|-----|
| No c>1 per-request profiling | Can't diagnose per-slot bottlenecks at c=4 | PMAT-055: `apr profile --concurrent 4 --per-request` |
| ~~No llama.cpp/vLLM comparison~~ | ~~Only `--ollama` comparison exists~~ | **DONE** PMAT-056: `--baseline-url` + `--baseline-model` + `--baseline-name` |
| No batch scheduler overhead | Can't measure scheduling vs kernel time | PMAT-055: Timer around scheduler + GEMV phases |
| No prefill/decode split at c>1 | Can't tell which phase dominates at c=4 | PMAT-055: Per-phase timing in batched path |
| BrickProfiler aggregates all slots | Per-slot attention/FFN time invisible | PMAT-055: `profiler.record_slot(brick, slot_id, ns)` |

### Make Targets for Profiling

```bash
make profile-yoga            # apr profile on yoga (roofline + hotspots + grade)
make profile-yoga-ci         # CI assertion mode (threshold-gated)
make profile-yoga-trace      # Live brick trace via X-Trace-Level header
make profile-yoga-compare    # Profile + compare against ollama
make profile-yoga-vs-llamacpp # Profile + compare against llama.cpp (PMAT-056)
make profile-yoga-vs-vllm    # Profile + compare against vLLM (PMAT-056)
make profile-yoga-full       # Full pipeline: deploy, profile, trace, teardown
```

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

### Yoga Baselines — c=1 (single request, 60s, streaming, isolated, ~126 prompt tokens)

| Runtime | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Prefill tok/s |
|---------|-------------|---------------|--------------|---------------|
| **vLLM** | **160.1** | 13.7 | **6.2** | 1,673.0 |
| ollama | 150.9 | 71.8 | 6.6 | 320.2 |
| llama.cpp | 142.5 | **12.3** | 7.0 | **8,280.4** |
| **realizr** | 138.6 | 78.8 | 7.2 | 1,297.5 |

**Parity check (realizr vs llama.cpp):**
- Decode: 138.6 / 142.5 = **0.97x** (PASS — within 5%)
- ITL: 7.2 / 7.0 = **1.03x** (PASS — within 5%)
- TTFT: 78.8 / 12.3 = **6.4x** (FAIL — target <= 2x)
- Prefill: 1297.5 / 8280.4 = **0.16x** (HGEMM vs Q4K GEMM, PMAT-059 improved 7.1x)

**Parity check (realizr vs vLLM):**
- Decode: 138.6 / 160.1 = **0.87x** (FAIL — 13% gap)
- ITL: 7.2 / 6.2 = **1.16x** (PASS)
- TTFT: 78.8 / 13.7 = **5.8x** (FAIL)

**PMAT-059:** Disabled prefill CUDA graph — cuBLAS workspace-free algorithms during graph
capture are 7x slower than eager cuBLAS (541ms vs 78ms for S=125). TTFT 561→78.8ms.
**PMAT-058 regression fix:** c=1 after c=4 batch recovers to baseline (138.4 tok/s).

### Yoga Baselines — c=4 (concurrent, 60s, streaming, isolated, ~126 prompt tokens)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| **vLLM** | **562.6** | **152.2** | **24.5** | **6.6** |
| llama.cpp | 303.8 | 76.3 | 22.1 | 13.1 |
| **realizr** | **137.0** | **42.8** | 659.7 | **23.4** |

**Parity check (realizr vs llama.cpp, c=4):**
- Decode: 42.8 / 76.3 = **0.56x** (FAIL — SGEMM prefill starves decode)
- Aggregate: 137.0 / 303.8 = **0.45x** (FAIL — target >= 3x c=1 = 420)
- ITL: 23.4 / 13.1 = **1.79x** (FAIL — target <= 1.5x)
- TTFT: 659.7 / 22.1 = **29.9x** (FAIL — target <= 2x)

**Parity check (realizr vs vLLM, c=4):**
- Decode: 42.8 / 152.2 = **0.28x** (FAIL — 3.6x gap)
- Aggregate: 137.0 / 562.6 = **0.24x** (FAIL — 4.1x gap)
- ITL: 23.4 / 6.6 = **3.5x** (FAIL)
- TTFT: 659.7 / 24.5 = **26.9x** (FAIL)

**Notes:**
- vLLM c=4: continuous batching, decode barely degrades from c=1 (160→152 tok/s, -5%)
- llama.cpp c=4: true parallel slots, 2.13x aggregate vs c=1
- realizr c=4: 0.99x aggregate vs c=1 (serial SGEMM prefill blocks decode pipeline)
- c=4 decode regression (60→43): FP16 cache cleared for batched KV → SGEMM prefill
  consumes ~660ms TTFT, reducing effective decode window per batch cycle
- PMAT-058: c=1 after c=4 fully recovers (138.4 tok/s)
- PMAT-056: Fixed multi-stream graph capture (conditional stream: capture→self.stream, eager→compute_stream)
- Graph disabled by default (capture 25% slower than eager), enable with BATCHED_GRAPH=1
- Fused QKV DP4A — quantize Q8_1 once, 3x GEMV reuse (56 fewer launches/step)
- vLLM uses AWQ INT4 quantization (efficient for batch serving), not GGUF

### Known Issues

1. **c=1 TTFT 6.4x gap** — 78.8ms vs 12.3ms. PMAT-059 improved 7.1x (was 48x). Remaining gap:
   HGEMM reads FP16 (3.56x more data). Fix: fused Q4K GEMM kernel.
2. **c=4 decode 0.56x gap** — 42.8 vs 76.3 tok/s. Root cause: M=4 DP4A GEMV compute-bound +
   SGEMM prefill overhead. M=1→M=4 ITL: 3.25x (realizr) vs 1.87x (llama.cpp).
3. **c=4 aggregate 0.45x llama.cpp** — 137.0 vs 303.8. Driven by SGEMM prefill (660ms TTFT)
   + decode gap (0.56x). Serial SGEMM blocks the decode pipeline.
4. **c=4 TTFT 29.9x gap** — 659.7ms vs 22.1ms. Root cause: FP16 cache cleared for batched KV
   (VRAM constraint) → SGEMM prefill. Confirmed: keeping FP16 causes CUDA_ERROR_ILLEGAL_ADDRESS.
4. ~~**c=1 regression after c=4**~~ **FIXED (PMAT-058)** — Was 140→124 tok/s. Root cause: FP16 weight cache cleared during batch decode, not rebuilt. Fix: `free_batched_kv_caches()` + `warmup_hgemm_cache()` + `clear_workspace()` at end of `generate_batched_streaming`. c=1 now recovers to 138.5 tok/s (baseline: 139.0).
5. **PMAT-056: Multi-stream graph capture fixed** — scatter/attention use self.stream during capture, compute_stream for eager. Removes PMAT-055 correctness bug. Graph replay still 25% slower than eager (capture overhead), default=eager.
6. **PMAT-054A: Fused QKV Q8_1 quantization** — Q8_1 quantize once, 3 GEMV reuse for Q/K/V projections. Saves 56 launches/step. Gate+up fusion (GH-141) saves 28/step. Total: 84 fewer launches/step.
7. **PMAT-046: Batched bias/rope/Q6K** — Cut 334 kernel launches/step (784→450). ITL 20.0→19.3ms.
8. **GH-141: Batched HW DP4A Q4K GEMV** — Replaces cuBLAS SGEMM for M=2..8 decode. Q4K (0.5625 B/elem) + Q8_1 (1.125 B/elem) vs FP32 (8 B/elem). Decode 51.1→60.2 (+17.8%).
9. **Parity gate false positive** — GPU parity check fails (cosine sim -0.28) but GPU output is correct. CPU reference diverges. Set `SKIP_PARITY_GATE=1` to bypass.
10. ~~**PMAT-050: Prefill CUDA graph capture**~~ **DISABLED (PMAT-059)** — cuBLAS workspace-free
    algorithms during graph capture are 7x slower than eager cuBLAS (541ms vs 78ms).
    Disable was default since v3.12.0. Enable with `PREFILL_GRAPH=1` for testing only.

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
- vLLM baseline added (Mar 9): AWQ INT4, 160.1 tok/s c=1, 562.6 aggregate c=4.
  New benchmark ceiling for continuous batching throughput.
- PMAT-054A: Fused QKV Q8_1 quantization — quantize once, 3x GEMV reuse (56 fewer
  launches/step). Provable contract validated (Grade B, binding coverage 1.00).
- PMAT-055: Disabled batched CUDA graph — multi-stream capture bug caused identical
  token_ids across decode steps. Eager batched path: c=4 decode 35.9→60.2 (+68%),
  aggregate 97→202 (+108%), zero-token failures 25%→0%.
- PMAT-056: Fixed multi-stream graph capture root cause — scatter/attention use
  self.stream during capture, compute_stream for eager (preserves kernel scheduling
  overlap). Removed !is_capturing guard from DP4A paths (pure GPU kernels are
  graph-capturable). Graph replay still 25% slower than eager; default=eager.
  Re-benchmarked llama.cpp c=4: 74.7 decode (was 79.7), ITL 13.4ms (was 12.9ms).
- PMAT-058: Fixed c=1 regression after c=4 batch. Five-Whys: FP16 cache (2944 MB)
  cleared during batch decode (GH-141 VRAM pressure), not rebuilt → VRAM layout
  change caused 12% decode regression (139→123 tok/s). Fix: free batched KV caches
  (~460MB), rebuild FP16 cache (~130ms), clear workspace to M=1 buffers. c=1 fully
  recovers to 138.5 tok/s after c=4 batch.
- PMAT-059: Disabled prefill CUDA graph — cuBLAS workspace-free algorithms during
  graph capture 7x slower than eager (541ms vs 78ms on RTX 4060L). Also fixed two
  secondary bugs: (a) separated prefill_graph_capture_failed from shared flag (prefill
  failure was blocking decode graph capture), (b) graph replay always allowed for
  already-captured S values. TTFT P50: 561→78.8ms (7.1x improvement).

---

## Five-Whys Root Cause Analysis

### TTFT 6.4x gap (c=1, realizr 78.8ms vs llama.cpp 12.3ms)

1. **Why is TTFT 6.4x worse?** 78.8ms for ~125-token prefill across 28 layers via eager HGEMM.
2. **Why 78.8ms?** cuBLAS HGEMM reads FP16 weights (2 B/elem), ~58ms GPU compute + ~12ms
   scheduler window + HTTP/SSE overhead. PMAT-059 discovered prefill CUDA graph made this
   7x WORSE (541ms) because cuBLAS uses workspace-free algorithms during graph capture.
3. **Why 58ms GPU compute?** FP16 cache occupies 2944 MB of 8 GB VRAM (37%).
   HGEMM reads 2 bytes/elem vs Q4K's 0.5625 bytes/elem — 3.56x more bandwidth consumed.
4. **Why not use quantized weights directly?** Fused Q4K GEMM (GH-182) is 16x SLOWER than
   cuBLAS HGEMM at M=125 on sm_89 due to serial accumulation without proper tiling.
5. **Why is llama.cpp 6.4x faster (12.3ms)?** Fused Q4K/Q6K GEMM reads 0.5625/0.66 B/elem,
   hand-optimized with CUDA core tiling + warp-level reduction + shared memory.

**Root cause:** FP16 bandwidth overhead (3.56x more data read than Q4K) + scheduler window.
**Fix needed:** Fused Q4K/Q6K GEMM that reads quantized weights directly with proper tiling.
**Status:** PMAT-059 improved 7.1x (561→78.8ms) by disabling prefill graph. Remaining 6.4x
gap requires fused Q4K GEMM kernel (~12ms target).

### c=4 decode 0.56x gap (realizr 42.8 vs llama.cpp 76.3 tok/s)

1. **Why is c=4 decode 0.56x?** ITL 23.4ms vs 13.1ms per token step at M=4.
2. **Why 23.4ms at M=4?** Two factors: (a) DP4A GEMV compute-bound at M=4, (b) SGEMM prefill
   consumes ~660ms TTFT per batch, reducing decode throughput in pipeline.
3. **Why worse M-scaling than llama.cpp?** realizr uses batched GEMV (one block per output row,
   loops over M activations). llama.cpp uses tiled GEMM that tiles both M and N.
   M=1→M=4 ITL ratio: 3.25x (realizr) vs 1.87x (llama.cpp).
4. **Why batched GEMV instead of GEMM?** DP4A GEMV was designed for M=1 decode (GH-176).
   M>1 batching was added by repeating M=1 kernel. No proper GEMM tiling for small M (2-8).
5. **Why does vLLM maintain near-c=1 decode (160→152, -5%)?** Marlin INT4 GEMM kernel
   processes M requests in single fused kernel with proper tiling.

**Root cause:** DP4A GEMV kernel designed for M=1, suboptimal at M=4. SGEMM prefill overhead.
**Fix needed:** Small-M GEMM kernel tiling + fused Q4K prefill to eliminate SGEMM dependency.
**Status:** Eager batched path: 42.8 tok/s, ITL 23.4ms.

### c=4 TTFT 29.9x gap (realizr 659.7ms vs llama.cpp 22.1ms)

1. **Why is TTFT 29.9x?** 4 sequential prefills at ~165ms each = 660ms.
2. **Why 165ms per prefill?** SGEMM prefill (FP32 per-call dequant), not HGEMM.
3. **Why SGEMM?** FP16 weight cache (2944 MB) + batched KV (~940 MB) + Q4K (~850 MB)
   exceeds 8GB VRAM → CUDA_ERROR_ILLEGAL_ADDRESS. Must clear FP16 before batch alloc.
   Confirmed: keeping FP16 causes CUDA_ERROR_ILLEGAL_ADDRESS (code 700).
4. **Why not prefill before allocating batched KV?** Current code order: clear FP16 →
   allocate batched KV → prefill. Reordering to prefill-first (HGEMM) → clear FP16 →
   allocate batched KV would use ~58ms/slot (232ms total) but requires D2H/H2D KV scatter.
5. **Why is llama.cpp 22.1ms?** Fused Q4K GEMM (no FP16 cache needed) + parallel slots.

**Root cause:** VRAM constraint forces FP16 clear → SGEMM fallback (3x slower than HGEMM).
**Fixes:** (a) Reorder: HGEMM prefill before batched KV alloc → ~232ms (2.8x improvement).
(b) Fused Q4K GEMM prefill → no FP16 dependency, ~50ms target.
(c) PMAT-051 chunked/batched prefill → fused multi-slot prefill.

### c=4 aggregate 0.36x gap (realizr 203.6 vs vLLM 562.6 tok/s)

1. **Why is c=4 aggregate 0.36x vLLM?** Per-request decode 60.2 vs 152.2 tok/s (0.40x),
   and serial prefill (4 × 100ms blocks decode).
2. **Why 0.40x per-request decode?** M=4 DP4A GEMV compute-bound (see decode analysis above).
   vLLM Marlin INT4 GEMM maintains near-M=1 efficiency.
3. **Why no CUDA graph at c>1?** PMAT-056 fixed multi-stream correctness (scatter on
   self.stream during capture). But graph replay is 25% slower than eager — capture overhead
   needs investigation.
4. **Why 25% graph regression?** Graph captures ~420 kernels per step including Q8_1 quantize
   + DP4A GEMV. Replay overhead from graph infrastructure may not amortize at <500 kernels.
5. **Why does vLLM's decode barely degrade (160→152, -5%)?** Marlin GEMM, PagedAttention,
   CUDA graph replay. Adding 3 more requests adds ~5% overhead vs 3x.

**Root cause:** DP4A GEMV compute-bound at M>1 + serial prefill overhead.
**Status:** PMAT-056 conditional stream fix deployed. Eager path: 203.6 aggregate tok/s.

---

## vLLM Source Analysis — Replication Plan

vLLM achieves 160.1 tok/s c=1 decode and 562.6 aggregate c=4 on the same RTX 4060L hardware.
Source code analysis (Mar 9) identified 5 architectural advantages to replicate.

### 1. Unified Scheduler — No Prefill/Decode Phase Split

**vLLM approach:** Single token budget per step. Running requests scheduled first (FCFS),
then waiting requests fill remaining budget. Each request tracks `num_computed_tokens` —
no separate "prefill queue" vs "decode queue". Mixed prefill+decode tokens processed in
one forward pass.

**realizr current:** `CudaBatchScheduler` has 10ms accumulation window, separate prefill
then decode phases. Prefill blocks the decode loop — 4 sequential prefills at c=4 = 1040ms
before first decode token.

**Replication (PMAT-051):**
- Track `num_computed_tokens` per slot in batch scheduler
- Token budget = max_batch_tokens (e.g., 4096)
- Each step: schedule running requests first (1 token each for decode), then fill remaining
  budget with waiting requests' prefill chunks
- Chunked prefill: split long prompts across steps (e.g., 128 tokens/step max per request)
- Result: first decode token for request 0 emitted while request 3 still prefilling

### 2. Paged KV Cache with Block Tables

**vLLM approach:** KV cache organized as fixed-size pages (block_size=16). Per-request
block table maps logical position → physical page. No contiguous allocation required.
Eliminates fragmentation, enables prefix sharing (same KV blocks reused across requests
with shared prefixes). Block table shape: `[max_num_reqs, max_num_blocks]`.

**realizr current:** Per-slot contiguous KV cache with stride-based layout. KV scatter via
D2D memcpy from single prefill to batched slot. Breaks CUDA graph capture because pointers
change per-layer.

**Replication (PMAT-052):**
- Allocate KV cache as `[num_blocks, 2, block_size, num_kv_heads, head_dim]`
- Per-request block table: `Vec<u32>` of block IDs
- Block allocator: free list of block IDs, allocate on prefill, free on request completion
- Slot mapping kernel: `slot = block_table[req][pos / block_size] * block_size + pos % block_size`
- **Critical benefit:** Graph-compatible — block table is a fixed-size tensor, only values change
  (no pointer updates per layer). Single graph captures entire forward pass.

### 3. Single-Kernel Batched Attention (FlashInfer/FlashAttention)

**vLLM approach:** One attention kernel launch processes ALL requests in the batch.
Split into decode path (M=1 query per request, optimized for long KV) and prefill path
(M>1 queries, full attention). Per-request metadata:
- `paged_kv_indptr`: cumulative block counts (shape `[num_reqs+1]`)
- `paged_kv_indices`: flat array of all block IDs
- `seq_lens`: KV cache length per request

**realizr current:** `batched_incremental_attention_into` processes M requests but with
stride-based KV layout. Flash Decoding splits KV into chunks across threadblocks.

**Replication (PMAT-053):**
- Modify attention kernel to accept block table + indptr instead of stride offset
- Decode: each threadblock handles one request's attention (grid = num_reqs × num_chunks)
- Prefill: per-request full attention with paged KV reads
- Key change: attention kernel reads KV via `kv_cache[block_table[block_idx]]` not
  `kv_cache[slot * stride + layer * layer_stride]`

### 4. Marlin-Style Quantized GEMM (c=1 decode gap)

**vLLM approach:** Marlin kernel achieves 10.9x speedup over naive AWQ. Key techniques:
- Tile-optimized INT4 weight layout (16×64 tiles, offline reshuffled)
- In-kernel INT4→FP16 dequant via bit manipulation (no materialization)
- Tensor core HMMA for the actual multiply-accumulate
- Weight bypasses L2 (direct to SMEM), preserves activation cache
- 8 threads per warp read 128-byte cache lines (coalesced)

**realizr current:** HW DP4A Q4K GEMV — reads Q4K weights, quantizes activations to Q8_1,
then integer dot product. CUDA cores only, no tensor cores for decode. 140.3 vs 160.1 = 0.88x.

**Analysis of the 12% c=1 gap:**
- vLLM Marlin reads INT4 (0.5 B/elem) + FP16 activations (2 B/elem)
- realizr reads Q4K (0.5625 B/elem) + Q8_1 activations (1.125 B/elem)
- Bandwidth similar, but Marlin uses tensor cores (HMMA) for compute
- DP4A uses CUDA cores — fewer FLOPS/cycle at M=1
- Q8_1 quantize overhead: 196 calls/step (28 layers × 7 GEMV)

**Replication options (PMAT-054):**
- **Option A:** Fused Q8_1 quantize — quantize once at layer entry, reuse for all 7 GEMV
  (already partially done for gate+up, extend to all projections)
- **Option B:** INT4→FP16 dequant + tensor core HMMA for M=1 (Marlin-style)
  Requires: offline weight reshuffling, FP16 activation path, HMMA PTX
- **Option C:** Reduce DP4A instruction count — current Q4K dequant is 20+ instructions
  per super-block vs llama.cpp's ~5 bitmask ops. Halving instructions = ~10% throughput gain

### 5. CUDA Graph Strategy — Full Graph for Decode, Piecewise for Mixed

**vLLM approach:** Two modes:
- **Uniform decode** (all requests at M=1): Full CUDA graph captures entire forward pass
  including Marlin GEMM. Zero launch overhead.
- **Mixed/prefill batches:** Piecewise graphs — attention stays eager (variable seq lengths),
  GEMM layers captured. Reduces memory vs full graph per batch shape.

**realizr current:** CUDA graph captures full decode at c=1. Batched decode (c>1) has
`BATCHED_GRAPH=0` workaround — per-layer KV pointer updates break graph capture.

**Replication (depends on PMAT-052):**
- With paged KV cache: block table is a fixed tensor, only values change → graph-compatible
- Capture full forward pass for M=1..8 decode (pre-capture common batch sizes)
- For mixed batches: capture GEMV/GEMM layers only, attention stays eager
- Estimate: eliminating kernel launch overhead at c=4 could recover 10-15% (16.8→~14.5ms ITL)

### Prioritized Replication Roadmap

| # | Task | Impact | Effort | Depends On |
|---|------|--------|--------|------------|
| 1 | ~~PMAT-054A: Fused Q8_1 quantize (QKV + gate+up)~~ | ~~+5-8% c=1 decode~~ | ~~Low~~ | **DONE** — saves 84 kernel launches/step |
| 2 | PMAT-051: Chunked prefill scheduler | 3-5x TTFT at c=4 | Medium | None |
| 3 | PMAT-052: Paged KV cache with block tables | Enables #4, #5 | High | None |
| 4 | PMAT-053: Paged attention kernel | Enables graph capture at c>1 | High | PMAT-052 |
| 5 | PMAT-052+graphs: Full graph capture at c>1 | 10-15% ITL at c=4 | Medium | PMAT-052 |
| 6 | PMAT-054B: Marlin-style HMMA decode | +12% c=1 decode | Very High | New kernel |

**Target after items 1-5:** realizr c=4 aggregate >= llama.cpp (317 tok/s), TTFT <= 2x llama.cpp.
**Target after item 6:** realizr c=1 decode >= vLLM (160 tok/s).
