# Performance Parity Specification

**Version:** 3.18.0
**Date:** 2026-03-10
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
| llama.cpp | 142.6 | **12.3** | 7.0 | **8,309.5** |
| **realizr** | 138.4 | 40.5 | 7.2 | 2,519.5 |

**Parity check (realizr vs llama.cpp):**
- Decode: 138.4 / 142.6 = **0.97x** (PASS — within 5%)
- ITL: 7.2 / 7.0 = **1.03x** (PASS — within 5%)
- TTFT: 40.5 / 12.3 = **3.3x** (FAIL — target <= 2x, was 4.0x pre-PMAT-063)
- Prefill: 2519.5 / 8309.5 = **0.30x** (BW-limited: FP16 2B/elem vs Q4K 0.56B/elem)

**Parity check (realizr vs vLLM):**
- Decode: 138.4 / 160.1 = **0.86x** (FAIL — 14% gap)
- ITL: 7.2 / 6.2 = **1.16x** (PASS)
- TTFT: 40.5 / 13.7 = **3.0x** (FAIL)

**PMAT-063:** cuBLAS workspace pre-allocation + multi-M JIT warmup + batch window 10→1ms.
Fixed `cublasSetWorkspace_v2` symbol name (was `cublasSetWorkspace`, not in libcublas.so.12).
Pre-warms 32 (M,N,K) shapes at model load (8 M values × 4 weight shapes). Batch scheduler
window reduced 10ms→1ms default, eliminates unnecessary latency for c=1.
TTFT 48.9→40.5ms (17%). Prefill graph capture works now but 11x slower than eager (cuBLAS
chooses slow algorithms during capture regardless of workspace). Abandoned prefill graph.
**Root cause profiling (PREFILL_DETAIL_TRACE):** Layer 0 GPU time: 1.12ms (S=125), all 28
layers: 27ms. Non-GPU overhead: 22ms (tokenization, lm_head, SSE). cuBLAS first-call JIT:
42ms per new (M,N,K) shape. Steady-state HGEMM: 16.7ms for 196 calls (0.085ms/call avg).
**PMAT-053:** FP8 E4M3 prefill via cuBLASLt — 1 B/elem weights (2x less than HGEMM FP16).
TTFT 48.9→39.0ms (1.25x). Prefill 2086→2618 tok/s (1.25x). Decode unchanged.
**PMAT-052:** Zero-copy c=1 prefill attention — read K/V from packed buffer (lda=kv_dim),
scatter via PTX kernel. Eliminated 14,000 D2D copies. TTFT 78.8→48.9ms (1.6x).
**PMAT-059:** Disabled prefill CUDA graph — cuBLAS workspace-free algorithms during graph
capture are 7x slower than eager cuBLAS (541ms vs 78ms for S=125). TTFT 561→78.8ms.
**PMAT-058 regression fix:** c=1 after c=4 batch recovers to baseline (138.4 tok/s).

### Yoga Baselines — c=4 (concurrent, 60s, streaming, isolated, ~126 prompt tokens)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|----------------|-------------|---------------|-------------|
| **vLLM** | **562.6** | **152.2** | **24.5** | **6.6** |
| llama.cpp | 305.4 | 76.4 | 22.0 | 13.1 |
| **realizr** | **193.5** | **51.1** | **128.3** | **19.6** |

**Parity check (realizr vs llama.cpp, c=4):**
- Decode: 51.1 / 76.4 = **0.67x** (FAIL — HGEMM FP16 BW 3.5x > Q4K)
- Aggregate: 193.5 / 305.4 = **0.63x** (FAIL — target >= 3x c=1 = 420)
- ITL: 19.6 / 13.1 = **1.50x** (PASS — target <= 1.5x)
- TTFT: 128.3 / 22.0 = **5.8x** (FAIL — was 10.7x, PMAT-051 v2 improved 1.9x)

**Parity check (realizr vs vLLM, c=4):**
- Decode: 51.2 / 152.2 = **0.34x** (FAIL — 3.0x gap)
- Aggregate: 193.8 / 562.6 = **0.34x** (FAIL — 2.9x gap)
- ITL: 19.5 / 6.6 = **3.0x** (FAIL)
- TTFT: 128.6 / 24.5 = **5.2x** (FAIL — was 14.6x, PMAT-051 v2 improved 2.8x)

**Notes:**
- vLLM c=4: continuous batching, decode barely degrades from c=1 (160→152 tok/s, -5%)
- llama.cpp c=4: true parallel slots, 2.08x aggregate vs c=1
- realizr c=4: 1.40x aggregate vs c=1 (PMAT-051 v2 improved 0.62x→0.65x vs llama.cpp)
- PMAT-051 v2: Multi-prompt prefill + zero-copy attention + PTX scatter kernel.
  Attn+Scatter 128.7→12.2ms (10.5x). TTFT 256→128.6ms (2.0x). FFN is now bottleneck (80%).
- PMAT-058: c=1 after c=4 fully recovers (138.8 tok/s)
- PMAT-056: Fixed multi-stream graph capture (conditional stream: capture→self.stream, eager→compute_stream)
- Graph disabled by default (capture 25% slower than eager), enable with BATCHED_GRAPH=1
- Fused QKV DP4A — quantize Q8_1 once, 3x GEMV reuse (56 fewer launches/step)
- vLLM uses AWQ INT4 quantization (efficient for batch serving), not GGUF

### Known Issues

1. **c=1 TTFT 3.3x gap** — 40.5ms vs 12.3ms. PMAT-063 improved from 48.9ms (batch window 10→1ms).
   **Root cause (profiled):** GPU prefill = 27ms (28 layers), non-GPU overhead = 13ms.
   GPU breakdown: HGEMM = 16.7ms (BW-limited by FP16 2B/elem), attention+norms = 10.3ms.
   llama.cpp reads Q4K directly (0.56 B/elem, 3.56x less BW) + has flash attention (fused ops).
   **Fix path:** Tiled Q4K GEMM with WMMA tensor cores (dequant Q4K→FP16 in SHMEM, tensor core
   GEMM). Theoretical TTFT: ~12ms (3.3ms GEMM + 5ms attention + 4ms overhead). This IS what
   llama.cpp does internally.
2. **c=4 decode 0.56x gap** — 42.8 vs 76.3 tok/s. Root cause: M=4 DP4A GEMV compute-bound +
   SGEMM prefill overhead. M=1→M=4 ITL: 3.25x (realizr) vs 1.87x (llama.cpp).
3. **c=4 aggregate 0.49x llama.cpp** — 149.2 vs 303.9. Driven by decode gap (0.56x) +
   sequential HGEMM prefill (237ms). PMAT-061 improved 0.49x→0.62x via HGEMM decode.
4. **c=4 TTFT 5.4x gap** — 128.6ms vs 24.0ms. PMAT-051 v2 eliminated D2D copy bottleneck
   (Attn+Scatter 128.7→12.2ms, 10.5x). Remaining gap: FFN HGEMM 82.7ms (80% of 103.5ms total).
   Fix: fused Q4K GEMM for prefill (reads Q4K directly, 3.56x less BW than HGEMM FP16).
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
  Wired into realizr prefill via FUSED_Q4K_PREFILL=1. K-offset bug fixed: pair*64+nibble*32
  (was pair*32+nibble*128). Kernel is now CORRECT (max_rel_err < 0.25) but architecturally
  20x slower than cuBLAS HGEMM on tensor-core GPUs — scalar dequant cannot use tensor cores.
  Useful only on Jetson Orin (BW-constrained, 67 GB/s). Not default on sm_89 GPUs.
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
- PMAT-061: cuBLAS HGEMM for M>1 decode — tensor cores instead of compute-bound DP4A GEMV.
  Five-Whys: Q4K GEMV 3.25x instruction scaling at M=4 (compute-bound). HGEMM reads FP16
  (3.5x more data) but tensor cores handle M×compute effortlessly (memory-bound).
  Keeps FP16 during decode (4690 MB fits in 7.5 GB). Disables fused QKV/gate+up DP4A
  so individual GEMVs route through cuBLAS. c=4 decode: 42.8→51.2 tok/s (+19.6%).
  ITL: 23.4→19.6ms (-16.2%). TTFT: 357.5→256.1ms (-28.3%). Aggregate: 149→184 (+23.5%).
  c=1 decode: no regression (138.3 vs 138.8). Env: HGEMM_BATCHED_DECODE=0 to disable.
- PMAT-060: HGEMM prefill during batched decode — disproved GH-141 VRAM assumption.
  FP16 (2944 MB) + batched KV (896 MB) + Q4K (850 MB) = 4690 MB, fits in 8GB VRAM
  with 2.8 GB headroom. The GH-141 ILLEGAL_ADDRESS was from stale decode graph pointers,
  not VRAM pressure. Also fixed init_prefill_workspace arg order bug (PMAT-045).
  c=4 TTFT: 660→358ms (1.85x). Prefill 145→285 tok/s (2x). Aggregate 137→149 (+9%).
- PMAT-051 v2: Multi-prompt batched prefill with zero-copy attention. Three optimizations:
  (a) prefill_multi_prompt: concatenate M prompts, run GEMM once per layer (weight sharing).
  (b) prefill_attention_from_packed: cuBLAS attention reads K/V directly from packed QKV
  buffer (lda=kv_dim instead of head_dim). Eliminates bulk_scatter_kv (56,000 D2D copies).
  (c) PTX scatter_packed_kv kernel: single launch scatters K/V from packed buffer to batched
  KV cache (replaces scatter_single_kv_to_batched_layer). Total: Attn+Scatter 128.7→12.2ms
  (10.5x), TTFT 256→128.6ms (2.0x). FFN now 80% of prefill (82.7ms / 103.5ms).
- PMAT-052: Zero-copy c=1 prefill attention. Same approach as PMAT-051 v2 applied to
  single-prompt path: read K/V directly from packed buffer (lda=kv_dim), scatter to single
  KV cache via PTX kernel (28 launches replaces 14,000 D2D copies from bulk_scatter_kv).
  c=1 TTFT: 78.8→48.9ms (1.6x). Prefill tok/s: 452→2086 (4.6x). Decode unchanged (138.2).
- PMAT-053: FP8 E4M3 prefill via cuBLASLt — cached FP8 weights (1 B/elem = 50% of FP16).
  Pipeline: f32→fp8 activations → cuBLASLt FP8 GEMM → fp16 output → f32. Weight cache
  1472 MB (vs 2944 MB FP16). c=1 TTFT: 48.9→39.0ms (1.25x). Prefill: 2086→2618 tok/s.
  GH-182: Fixed fused Q4K GEMM K-offset bug (pair*64+nibble*32). Kernel is correct but
  20x slower than cuBLAS HGEMM on tensor-core GPUs — scalar dequant can't use tensor cores.
  Fused Q4K useful only on Jetson Orin (BW-constrained). FP8 is better approach for sm_89+.
  Bottleneck: f32↔fp8 conversion overhead (~12ms per prefill = 196 GEMMs × 0.06ms/convert).
  cuBLASLt first-call heuristic: 3.1ms (cached internally after first shape).
- PMAT-063: cuBLAS workspace pre-allocation + multi-M JIT warmup + batch window 10→1ms.
  Three fixes: (a) Fixed cublasSetWorkspace_v2 symbol name (was cublasSetWorkspace — not
  exported by libcublas.so.12, only cublasSetWorkspace_v2 exists). (b) Pre-warm 32 (M,N,K)
  shapes at model load (8 M values × 4 weight shapes). Each new shape has 42ms first-call
  JIT. (c) Batch scheduler default window 10ms→1ms — eliminates unnecessary TTFT latency.
  c=1 TTFT: 48.9→40.5ms (17%). Prefill graph with workspace works but 11x slower than eager
  (cuBLAS chooses slow algorithms during capture). Profiling: HGEMM steady-state 16.7ms (196
  calls, 0.085ms avg), GPU total 27ms (28 layers), non-GPU 13ms. Root cause confirmed:
  FP16 BW overhead. Need tiled Q4K WMMA GEMM for parity.

---

## Five-Whys Root Cause Analysis

### TTFT 3.3x gap (c=1, realizr 40.5ms vs llama.cpp 12.3ms)

1. **Why is TTFT 3.3x worse?** 40.5ms for ~125-token prefill across 28 layers via eager HGEMM.
2. **Why 40.5ms?** GPU prefill = 27ms (measured via PREFILL_TRACE), non-GPU = 13ms.
   GPU breakdown: HGEMM 16.7ms (196 calls × 0.085ms avg) + attention/norms 10.3ms.
   Non-GPU: tokenization + embed H2D + lm_head + argmax + SSE = ~13ms.
   Batch scheduler window (10ms→1ms) saved ~9ms. PMAT-063.
3. **Why 16.7ms HGEMM?** Reads 2944 MB FP16 weights at ~177 GB/s effective = 16.7ms.
   llama.cpp reads 850 MB Q4K at ~177 GB/s = 4.8ms. 3.56x more data is the gap.
4. **Why not use quantized weights directly?** Fused Q4K GEMM (GH-182) correct but 20x slower
   on sm_89 (scalar dequant can't use tensor cores). Need tiled Q4K GEMM with WMMA.
5. **Why is llama.cpp 3.3x faster?** Tiled Q4K GEMM: load Q4K from DRAM, dequant→FP16 in
   shared memory, WMMA tensor core compute. Reads 3.56x less data + flash attention.

**Root cause:** FP16 bandwidth overhead (3.56x more data read than Q4K).
**Fix path:** Tiled Q4K WMMA GEMM — dequant Q4K super-blocks per-tile into SHMEM,
use WMMA FP16 tensor cores for compute. Target: TTFT ~12ms (matching llama.cpp).
**PMAT-063 profiling results (PREFILL_DETAIL_TRACE):**
- cuBLAS first-call JIT: 42ms per new (M,N,K) shape. Fixed by multi-M warmup (8 M values).
- Prefill graph capture with workspace: works but 11x slower than eager (slow algorithms).
- Steady-state HGEMM: 16.7ms / 196 calls = 0.085ms/call (near-optimal for cuBLAS).
- cublasSetWorkspace_v2 symbol name fixed (was cublasSetWorkspace → undefined symbol).
**History:** PMAT-052 1.6x (78.8→48.9ms), PMAT-059 7.1x (561→78.8ms), PMAT-063 1.2x (48.9→40.5ms).

### c=4 decode 0.56x gap (realizr 42.8 vs llama.cpp 76.4 tok/s)

1. **Why is c=4 decode 0.56x?** ITL 23.4ms vs 13.1ms per token step at M=4.
2. **Why 19.6ms at M=4?** PMAT-061 routes M>1 decode through cuBLAS HGEMM (tensor cores).
   FP16 weight reads = 3 GB vs Q4K 850 MB → 3.5x more data, but tensor cores hide compute.
   HGEMM is memory-bound (~15.6ms theoretical) vs DP4A GEMV compute-bound (~23.4ms measured).
3. **Why still 1.46x vs llama.cpp?** llama.cpp reads Q4K directly (850 MB at 192 GB/s = 4.4ms
   theoretical). HGEMM reads FP16 (3 GB at 192 GB/s = 15.6ms). 3.5x BW gap.
4. **Why not read Q4K directly for M=4?** Fused Q4K GEMM (FUSED_Q4K_PREFILL=1) was tested —
   53.5ms ITL at M=4 (3x worse). Kernel designed for large M (prefill), not small M decode.
5. **Why does vLLM maintain near-c=1 decode (160→152, -5%)?** Marlin INT4 GEMM kernel
   processes M requests in single fused kernel with proper M×N tiling.

**Root cause:** FP16 HGEMM reads 3.5x more data than Q4K. Need fused Q4K GEMM for small M.
**Fix needed:** Fused Q4K GEMM kernel optimized for small M=2..8 (tile_m=4, shared memory).
**Status:** Eager batched path: 42.8 tok/s, ITL 23.4ms.

### c=4 TTFT 5.4x gap (realizr 128.6ms vs llama.cpp 24.0ms)

1. **Why is TTFT 5.4x?** Multi-prompt HGEMM prefill: QKV 8.6ms + Attn 12.2ms + FFN 82.7ms = 103.5ms + overhead.
2. **Why 82.7ms FFN?** 6 HGEMM calls (gate, up, down + output) × 28 layers at M_total=500.
   HGEMM reads FP16 weights (2 B/elem) — 3.56x more bandwidth than Q4K (0.5625 B/elem).
3. **Why not fused Q4K GEMM?** GH-182 fused Q4K GEMM K-offset fixed, now CORRECT. But 20x
   SLOWER than cuBLAS HGEMM on sm_89 — scalar dequant can't use tensor cores.
4. **Why is llama.cpp 24.0ms?** Fused Q4K GEMM (0.5625 B/elem) + parallel slots.
5. **Why not parallel slots?** PMAT-051 v2 already reads weights ONCE for all 4 prompts.
   Remaining gap is pure HGEMM bandwidth overhead.

**Root cause:** HGEMM reads 3.56x more data than Q4K (FP16 vs Q4_K_M).
**PMAT-051 v2:** Multi-prompt batched prefill + zero-copy attention (lda=kv_dim) + PTX
  scatter kernel. Eliminated 56,000 D2D copies. Attn+Scatter: 128.7→12.2ms (10.5x).
  TTFT: 256→128.6ms (2.0x). FFN is now 80% of total (82.7ms / 103.5ms).
**Fixes remaining:** FP8 prefill (1 B/elem, 2x BW reduction) → ~64ms target. Fused Q4K GEMM
  (0.5625 B/elem, 3.56x BW reduction) → ~23ms target but needs llama.cpp-level optimization.

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
