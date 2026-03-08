# Performance Parity Specification

**Version:** 3.0.0
**Date:** 2026-03-08
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

`apr serve` (via realizr inference engine) must match or beat ollama and llama.cpp
on decode throughput, TTFT, and ITL — across all three model formats:

| Format | Runtime | Parity Target |
|--------|---------|---------------|
| GGUF Q4_K_M | apr serve | >= llama.cpp |
| APR native | apr serve | >= llama.cpp |
| SafeTensors | apr serve | >= ollama |

**One sentence:** Read weights once, generate tokens at memory-bandwidth speed,
for 1-32 concurrent requests, on consumer NVIDIA GPUs.

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

---

## Architecture Context

```
yoga (RTX 4060 Laptop, 192.168.50.38)
├── apr serve :8081  (realizr engine, GGUF/APR/SafeTensors)
├── llama.cpp :8083  (baseline, GGUF only)
├── ollama    :8082  (baseline, GGUF only)
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

## Current State (2026-03-08)

### Yoga Baselines — c=1 (single request, 60s, streaming, locked clocks)

| Runtime | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Prefill tok/s |
|---------|-------------|---------------|--------------|---------------|
| realizr | 138.7 | 114.4 | 7.2 | 891.7 |
| llama.cpp | **143.4** | **12.1** | **7.0** | **8399.9** |
| ollama | 145.4 | 71.5 | 6.9 | 1427.1 |

**Parity check (realizr vs llama.cpp):**
- Decode: 138.7 / 143.4 = **0.97x** (PASS — within 5%)
- TTFT: 114.4 / 12.1 = **9.5x** (FAIL — target <= 2x)
- ITL: 7.2 / 7.0 = **1.03x** (PASS — within 1.5x)
- Prefill: 891.7 / 8399.9 = **0.11x** (9.4x gap — drives TTFT)

### Yoga Baselines — c=4 (continuous batching, 60s, streaming)

| Runtime | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Throughput req/s |
|---------|-------------|---------------|--------------|-----------------|
| realizr | **0.0** | 42.0 | — | 96.4 |
| llama.cpp | 74.0 | 22.0 | 13.5 | 2.3 |
| ollama | 144.8 | 2676.4 | 6.9 | 1.1 |

**Notes:**
- realizr c=4: 0 decode tokens — PMAT-044 batched decode bug (prefill works, decode broken)
- ollama c=4: serializes requests (no real batching), same decode as c=1, huge TTFT
- llama.cpp c=4: true batching, 2.1x aggregate throughput vs c=1

### Known Issues

1. **PMAT-044: Batched decode correctness** — c>1 produces 0 output tokens on yoga.
   Prefill completes but decode generates nothing. Must debug from `../realizar`.
2. **Prefill 9.4x gap** — realizr reads FP16 (2 B/elem) vs llama.cpp fused Q4K GEMM
   (0.5625 B/elem). Requires fused quantized GEMM kernel in `../trueno`.

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
- Batched GEMV: weight sharing across M concurrent requests
