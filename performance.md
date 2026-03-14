# LLM Inference Performance

## GPU (RTX 4090, Qwen2.5-Coder-1.5B Q4_K_M)

### Competition Benchmarks (2026-03-04, c=4, 60s, 5s warmup)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Tok/s | Decode tok/s | Error Rate |
|------|---------|-------------|-----|----------|----------|----------|-------|-------------|------------|
| 2026-03-04 | llama.cpp | 4 | 7.4 | 537.8 | 565.4 | 588.7 | 948.2 | 238.0 | 0% |
| 2026-03-04 | ollama | 4 | 4.4 | 899.5 | 938.7 | 947.2 | 568.9 | 142.3 | 0% |
| 2026-03-04 | realizar-safetensors | 4 | 1.4 | 2,643.7 | 4,428.2 | 4,469.5 | 167.1 | 43.3 | 0% |
| 2026-03-04 | realizar-gguf | 4 | 1.3 | 3,259.5 | 4,078.2 | 4,162.8 | 150.7 | 39.1 | 0% |
| 2026-03-04 | realizar-apr | 4 | 1.4 | 2,728.3 | 3,560.9 | 4,057.3 | 143.3 | 39.9 | 0% |

### Previous (2026-03-03, c=4, 60s, 3 runs)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | Tok/s | Error Rate |
|------|---------|-------------|-----|----------|----------|----------|-------|------------|
| 2026-03-03 | llama.cpp | 4 | 7.92 | 504 | 521 | 528 | 1,013.6 | 0% |
| 2026-03-03 | ollama | 4 | 4.75 | 839 | 872 | 887 | 607.9 | 0% |
| 2026-03-03 | realizar-safetensors | 4 | 0.75 | 5,274 | 7,480 | 9,013 | 96.5 | 0% |
| 2026-03-03 | realizar-gguf | 4 | 0.20 | 18,989 | 24,229 | 24,258 | 25.8 | 0% |
| 2026-03-03 | realizar-apr | 4 | 0.00 | N/A | N/A | N/A | 0.0 | 100% |

### Historical (2026-03-02)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Requests |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|----------|
| 2026-03-02 | realizar-gpu | 4 | 10.2 | 392.6 | 599.6 | 705.2 | 392.6 | 10.2 | 609 |
| 2026-03-02 | ollama-gpu | 4 | 120.3 | 30.8 | 48.8 | 72.0 | 30.8 | 240.5 | 7216 |
| 2026-03-02 | llamacpp-gpu | 4 | 328.2 | 11.4 | 15.6 | 18.5 | 11.4 | 656.4 | 19692 |

## CPU (Intel EPYC, 192.168.50.100, Qwen2.5-Coder-1.5B Q4_K_M)

### Competition Benchmarks (2026-03-03, c=4, 60s, 3 runs)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Error Rate |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|------------|
| 2026-03-03 | llama.cpp | 4 | 1.71 | 2,340 | 2,381 | 2,389 | 2,340 | 218.5 | 0% |
| 2026-03-03 | ollama | 4 | 1.17 | 3,356 | 3,782 | 3,817 | 3,356 | 149.5 | 0% |
| 2026-03-03 | realizar-safetensors | 4 | 0.22 | 18,110 | 18,293 | 18,317 | 18,110 | 28.3 | 0% |
| 2026-03-03 | realizar-gguf | 4 | 0.18 | 20,007 | 30,699 | 31,408 | 20,007 | 23.0 | 0% |
| 2026-03-03 | realizar-apr | 4 | 0.07 | 53,263 | 54,537 | 54,537 | 53,263 | 9.5 | 0% |

### Historical (2026-03-01)

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Requests |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|----------|
| 2026-03-01 | realizar-apr | 4 | 0.4 | 12807.2 | 12950.4 | 12963.4 | 12807.2 | 6.9 | 13 |
| 2026-03-01 | realizar-gguf | 4 | 1.5 | 2510.7 | 3839.4 | 3876.5 | 2510.6 | 1.5 | 45 |

## Isolated Streaming (c=1, 60s, 5s warmup, stream=true) — 2026-03-13 (PMAT-109)

### RTX 4060 Laptop — yoga (24 SMs, sm_89, 8GB VRAM, locked 1900MHz)

**Short prompt (23 tokens):**

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | TTFT P99 (ms) | ITL P50 (ms) |
|---------|-------------|--------------|---------------|---------------|-------------|
| ollama | **164.6** | — | 69.8 | — | **6.1** |
| llama.cpp | 161.7 | **2,280** | **10.2** | — | 6.2 |
| vLLM | 153.6 | 2,016 | 12.6 | — | 6.5 |
| realizr | 149.5 | 1,718 | 13.2 | **14.2** | 6.7 |

**Decode: 4-way near-parity.** ollama leads M=1 at 164.6 but serial processing (c>1 incompatible).
realizr vs llama.cpp = **0.92x** (DP4A BW ceiling: 57% vs 62% roofline utilization).
**TTFT: 1.29x** (13.2ms vs 10.2ms) — FP8 prefill (1 B/elem) vs Q4K fused GEMM (0.56 B/elem).
**PMAT-109:** Graph persistence fix — bimodal TTFT tail ELIMINATED. P99 14.2ms (was 35ms). Tail score 86→100 A+.
**PMAT-106/107/110:** 13 kernel approaches falsified for M=1 decode improvement. 92% of DP4A ceiling reached.

### RTX 4090 (128 SMs, sm_89)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) | Latency P50 (ms) |
|---------|-------------|--------------|---------------|-------------|-----------------|
| realizr | **411.7** | 1,734 | 58.8 | 2.4 | 368.1 |
| llama.cpp | 436.9 | 17,620 | 5.8 | 2.3 | — |

**Decode gap: 1.06x (near parity).** Improvement: 1.55x (266→412 tok/s) via Flash Decode chunk\_size 128→32.
Prefill gap: 10.2x (HGEMM FP16 reads vs llama.cpp fused Q4K GEMM).

### Jetson Orin Nano Super (8 SMs, sm_87, MAXN_SUPER 1020MHz)

| Runtime | Decode tok/s | Prefill tok/s | TTFT P50 (ms) | ITL P50 (ms) | Latency P50 (ms) |
|---------|-------------|--------------|---------------|-------------|-----------------|
| **realizr** | **40.8** | 481.4 | 47.8 | **24.5** | 807.4 |
| llama.cpp | 36.1 | **676.0** | **34.0** | 27.7 | 893.7 |

**Decode: realizr 13% FASTER than llama.cpp (0.88x).** Improvement: +12.4% from MAXN_SUPER (1020MHz) + PMAT-078 Q6K SMEM cache.
Prefill gap: 1.4x (HGEMM FP16 on-demand vs fused Q4K GEMM). TTFT narrowed from 5.5x to 1.4x.

### Config: GpuProfile auto-detect (no env vars), fused\_gate\_up=true, FLASH\_DECODE\_CHUNK\_SIZE=32

### Optimization History (GH-131/173/174/176, PMAT-033→087)

| Step | Jetson Decode | 4090 Decode | 4060L Decode | Date |
|------|--------------|-------------|-------------|------|
| Baseline (MWV DP4A) | 16.7 | 128.4 | — | Mar 5 |
| +locked clocks | 21.4 | — | — | Mar 6 |
| +HW DP4A Q4K (GH-176) | 27.8 | 162.8 | — | Mar 6 |
| +grid 16 blocks/SM | 33.7 | — | — | Mar 7 |
| +HGEMM prefill + graph | 32.7 | 266.3 | — | Mar 7 |
| +Flash Decode chunk=32 | 36.3 | **411.7** | — | Mar 8 |
| +PMAT-044 PTX parity | — | — | 140.3 | Mar 8 |
| +PMAT-086 FP8+batch0 | — | — | 139.0 (TTFT: 46→15.5ms) | Mar 11 |
| +PMAT-087 1900MHz | — | — | 154.8 (TTFT: 13.4ms) | Mar 12 |
| +PMAT-109 Graph persist | — | — | **149.5** (TTFT: 13.2ms, P99: **14.2ms**) | Mar 13 |
| +PMAT-105 FP8 LmHead | — | — | c=4: **357.2** (was 210.8, +69%) | Mar 13 |
| +MAXN_SUPER 1020MHz | **40.8** | — | — | Mar 12 |

### Cross-Platform Decode Summary (c=1, isolated, streaming)

| Platform | ollama | llama.cpp | vLLM | realizr |
|----------|--------|-----------|------|---------|
| **RTX 4060 Laptop** (24 SMs, 1900MHz) | **164.6** | 161.7 | 153.6 | 149.5 |
| RTX 4090 (128 SMs) | — | 436.9 | — | 411.7 |
| Jetson Orin (8 SMs, MAXN_SUPER 1020MHz) | — | 36.1 | — | **40.8** |

### Bandwidth Utilization (corrected: 67 GB/s peak for Orin Nano Super)

| Runtime | BW (GB/s) | % of Peak |
|---------|----------|-----------|
| realizr Q4K GEMV | 20.5 | 30.6% |
| realizr total decode | 18.2 | 27.1% |
| llama.cpp total decode | 27.4 | 40.9% |

Tracking: [GH-131](https://github.com/paiml/realizar/issues/131)

## Concurrent Streaming (c=4, 60s, 5s warmup, stream=true) — 2026-03-13 (PMAT-111)

### RTX 4060 Laptop — yoga (24 SMs, sm_89, 8GB VRAM, locked 1900MHz, short prompt)

| Runtime | Aggregate tok/s | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Errors |
|---------|----------------|-------------|---------------|-------------|--------|
| **vLLM** | **594.8** | **150.4** | 25.3 | **6.7** | 0% |
| llama.cpp | 365.8 | — | **19.0** | 10.7 | 1.3% |
| **realizr** | **357.2** | 96.1 | 36.2 | 10.4 | **0%** |
| ollama | 159.1 | 161.5 | 612.3 | 6.2 | 0% |

**vLLM dominates c=4** via continuous batching + PagedAttention (1.67x over realizr).
realizr vs llama.cpp: **0.98x PARITY** (short prompt) — 0% errors vs 1.3%.
**PMAT-105:** FP8 cuBLASLt LmHead at M>=5 — reads weights once instead of M times. Single biggest c≥4 breakthrough.
Ollama: serial prefill — TTFT 612ms at c=4, aggregate flat vs c=1. Production-incompatible at c>1.

## Concurrency Scaling (c=1→16, 60s, 5s warmup, short prompt) — 2026-03-13 (PMAT-120)

### RTX 4060 Laptop — yoga (all runtimes --parallel/batch 16, isolated)

| c | realizr | llama.cpp | vLLM | ollama | realizr/llama.cpp |
|---|---------|-----------|------|--------|-------------------|
| 1 | 149.5 | 158.9 | 153.6 | **164.6** | 0.94x |
| 4 | 357.2 | 365.8 | **594.8** | 159.1 | **0.98x** PARITY |
| **8** | **637.8** | 430.0 | **1,058.5** | — | **1.48x** WINS |
| 12 | 899.3 | 906.0 | **1,461.3** | — | 0.99x PARITY |
| **16** | **1,139.5** | 1,000.4 | **1,832.2** | — | **1.14x** WINS |

**Scaling efficiency (c=1→c=16):** vLLM 11.9× (74%) > realizr 7.6× (47.5%) > llama.cpp 6.5× (40.4%).
**ITL stability (c=1→c=16):** vLLM +14% (5.9→6.7ms) > realizr +75% (6.7→11.7ms) > llama.cpp +139% (6.2→15.4ms).
**Quality:** realizr 0% errors at all c. llama.cpp 1.3-3.2% errors. vLLM 0%.
**PMAT-105 breakthrough:** LmHead FP8 dispatch at M≥5. ITL nearly flat c=4→c=16 (10.4→11.7ms).

## Prompt-Profile Sensitivity (short/medium/long) — 2026-03-13 (PMAT-113→118)

### RTX 4060 Laptop — realizr vs llama.cpp competitive ratio (aggregate tok/s)

| c | Short (~23 tok) | Medium (~102 tok) | Long (~311 tok) | Direction |
|---|----------------|-------------------|-----------------|-----------|
| 1 | 0.93x | 0.90x | 0.82x | realizr worse with length |
| 4 | **1.01x** PARITY | **0.81x** LOSES | **0.60x** LOSES | Grade inversion: B→D |
| **8** | **1.47x** WINS | **1.08x** wins | **0.77x** LOSES | Crossover disappears |

**Root cause:** realizr's 2-step FP8 pipeline (Q4K→FP8 convert + FP8 GEMM) reads **1.78× more weight bandwidth** per prefill than llama.cpp's fused 1-step Q4K GEMM. This is the SOLE cause of prompt-length sensitivity.
**llama.cpp and vLLM are prompt-length invariant** (−2% to +3% across all profiles and concurrency levels).
**realizr is the ONLY runtime with prompt-length sensitivity** — drops 39-49% from short→long prompts.
**Fused Q4K→GEMM kernel** is the single highest-value optimization — would close this gap entirely.

### vLLM Reference (Marlin W4A16 + PagedAttention, PMAT-119/120)

| c | Short | Medium | Long | Sensitivity |
|---|-------|--------|------|-------------|
| 1 | 153.6 | — | 149.1 | −2.9% (invariant) |
| 4 | 594.8 | 551.0 | 558.5 | −6.1% (invariant) |
| 8 | 1,058.5 | 1,023.2 | 1,040.7 | −1.7% (invariant) |

### Cross-Prompt Scorecards (probador llm score)

| Runtime | c=4 Short | c=4 Medium | c=4 Long | Drop |
|---------|-----------|------------|----------|------|
| realizr | 78 B | 67 C+ | **49 D** | −30 points |
| llama.cpp | 70 B | 71 B | 67 C+ | −3 points |

### Jetson Orin Nano Super (8 SMs, sm_87, MAXN_SUPER 1020MHz)

**c=4 provides zero batching benefit on 8 SMs:**

| Runtime | c=1 Decode tok/s | c=4 Aggregate tok/s | TTFT P50 (ms) | ITL P50 (ms) |
|---------|-----------------|---------------------|---------------|-------------|
| realizr | 40.8 | 39.6 | 2,469 | 24.5 |
| llama.cpp | 36.1 | — | — | — |

**Root cause:** 8 SMs fully saturated at M=1 — no headroom for batched compute.
Serial prefill at c=4: ~617ms each (4 × sequential). Decode unchanged from c=1.

## GPU Profiling — BrickProfiler (2026-03-06, C-GDP-001 Contract)

### Corrected Brick Breakdown (RTX 4090, Immediate Sync, CUDA\_GRAPH\_DISABLE=1)

After fixing 18 hardcoded values in the cbtop pipeline ([aprender#426](https://github.com/paiml/aprender/pull/426)),
BrickProfiler now reports real per-kernel GPU timing:

| Brick | Per-Call (µs) | Per-Decoded-Token (µs) | % of Decode |
|-------|--------------|----------------------|-------------|
| AttentionScore | 67.5 | 1,891 | 17.7% |
| GateProjection | 53.2 | 1,489 | 13.9% |
| RmsNorm | 25.2 | 1,434 | 13.4% |
| DownProjection | 42.1 | 1,178 | 11.0% |
| QkvProjection | 35.1 | 982 | 9.2% |
| Activation | 30.6 | 856 | 8.0% |
| Residual2 | 24.2 | 678 | 6.3% |
| LmHead | 594.2 | 594 | 5.6% |
| OutputProjection | 21.0 | 587 | 5.5% |
| RopeEmbedding | 19.6 | 549 | 5.1% |
| Residual1 | 15.9 | 446 | 4.2% |

Contract: `gpu-decode-profiling-v1` v2.0.0 — 15 falsification tests, all PASS.

Fixes:
- [realizar#137](https://github.com/paiml/realizar/pull/137): Force eager decode when profiler active (CUDA graphs hide brick timing)
- [aprender#426](https://github.com/paiml/aprender/pull/426): 18 hardcoded BrickScore values replaced with real profiler data

### Serial Baseline (2026-03-06, c=1, isolated, CUDA\_GRAPH\_DISABLE=1)

| Runtime | Decode tok/s |
|---------|-------------|
| realizar | 162.7 |
| llama.cpp | 262.7 |
| **Gap** | **1.61x** |

**Note:** This was before Flash Decode chunk\_size=32 (PMAT-040). Current graph-mode decode: 411.7 tok/s (1.06x gap).

## GPU Profiling — Nsight Systems (2026-03-04)

### CUDA Kernel Time Distribution

| Kernel | Time (%) | Instances | Avg (µs) | Med (µs) |
|--------|----------|-----------|----------|----------|
| mwv_q4k_gemv | 46.0% | 53,592 | 9.9 | 4.5 |
| q6k_gemv_warp_reduce | 31.9% | 9,251 | 39.7 | 39.2 |
| multi_warp_attention_indirect | 9.3% | 8,932 | 12.0 | 11.2 |
| rmsnorm_vectorized | 5.3% | 18,183 | 3.3 | 3.1 |
| residual_add | 3.8% | 44,660 | 1.0 | 1.0 |
| rope_neox_indirect | 1.7% | 17,864 | 1.1 | 1.0 |
| kv_cache_scatter_indirect | 1.3% | 17,864 | 0.9 | 0.9 |
| fused_swiglu | 0.7% | 8,932 | 0.9 | 0.9 |

Source: `results/nsys-apr-gpu-kernels-20260304.txt`

### Per-Operation Telemetry (2026-03-02)

| Operation | Time (µs) | % of Decode | Bottleneck |
|-----------|-----------|-------------|------------|
| AttentionScore | 88,390 | 76.0% | MEMORY |
| RmsNorm | 17,118 | 14.7% | MEMORY |
| QkvProjection | 2,755 | 2.4% | MEMORY |
| GateProjection | 1,838 | 1.6% | MEMORY |
| RopeEmbedding | 1,637 | 1.4% | COMPUTE |
| OutputProjection | 965 | 0.8% | MEMORY |
| DownProjection | 938 | 0.8% | MEMORY |

**Kernel launch overhead:** 128,484µs (52.5% of decode time)
**Memory efficiency:** 8.4% (Grade D)
**Decode throughput (profile run):** 130.7 tok/s

Source: `results/profile-gpu-20260302.txt`

## Performance Results

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Avg tok/req | ITL P50 (ms) | Decode tok/s | Prefill tok/s | TPOT P50 (ms) | Err% | Requests |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|-------------|--------------|--------------|---------------|---------------|------|----------|
| 2026-03-01 | realizar-apr | 4 | 0.4 | 12807.2 | 12950.4 | 12963.4 | 12807.2 | 6.9 | 0.0 | - | - | - | - | 0% | 13 |
| 2026-03-01 | realizar-gguf-1 | 4 | 1.5 | 2586.0 | 4155.6 | 4179.1 | 2586.0 | 1.5 | 0.0 | - | - | - | - | 0% | 45 |
| 2026-03-01 | realizar-gguf-2 | 4 | 1.5 | 2510.7 | 3839.4 | 3876.5 | 2510.6 | 1.5 | 0.0 | - | - | - | - | 0% | 45 |
| 2026-03-02 | realizar-gpu | 4 | 10.2 | 392.6 | 599.6 | 705.2 | 392.6 | 10.2 | 0.0 | - | - | - | - | 0% | 609 |
| 2026-03-02 | ollama-gpu | 4 | 120.3 | 30.8 | 48.8 | 72.0 | 30.8 | 240.5 | 0.0 | - | - | - | - | 0% | 7216 |
| 2026-03-02 | llamacpp-gpu | 4 | 328.2 | 11.4 | 15.6 | 18.5 | 11.4 | 656.4 | 0.0 | - | - | - | - | 0% | 19692 |
| 2026-03-02 | realizar-gpu | 4 | 4.5 | 743.6 | 1647.6 | 2154.1 | 743.6 | 4.5 | 0.0 | - | - | - | - | 0% | 267 |
| 2026-03-02 | ollama-gpu | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 0% | 400038 |
| 2026-03-02 | llamacpp-gpu | 4 | 97.5 | 33.1 | 91.3 | 95.1 | 33.1 | 195.1 | 0.0 | - | - | - | - | 0% | 5853 |
| 2026-03-02 | realizar-gpu | 4 | 5.1 | 608.6 | 1611.8 | 1664.8 | 608.5 | 5.1 | 0.0 | - | - | - | - | 0% | 306 |
| 2026-03-02 | ollama-gpu | 4 | 91.9 | 35.0 | 75.4 | 85.6 | 35.0 | 183.8 | 0.0 | - | - | - | - | 0% | 5514 |
| 2026-03-02 | llamacpp-gpu | 4 | 161.0 | 11.6 | 67.8 | 70.7 | 11.6 | 322.0 | 0.0 | - | - | - | - | 0% | 9660 |
| 2026-03-04 | realizar-gpu | 4 | 1.2 | 2529.5 | 5013.3 | 5794.3 | 2529.4 | 151.4 | 121.9 | 24.5 | 40.8 | - | 0.0 | 0% | 78 |
| 2026-03-04 | ollama-gpu | 4 | 4.4 | 914.7 | 951.5 | 957.8 | 914.6 | 561.6 | 128.0 | 7.1 | 139.9 | - | 0.0 | 0% | 264 |
| 2026-03-04 | llamacpp-gpu | 4 | 7.3 | 548.2 | 576.4 | 586.9 | 548.2 | 931.5 | 128.0 | 4.3 | 233.5 | - | 0.0 | 0% | 440 |
| 2026-03-04 | ollama-jetson | 1 | 0.1 | 11313.3 | 11600.5 | 11600.5 | 11313.3 | 11.3 | 128.0 | 88.4 | 11.3 | - | 0.0 | 0% | 6 |
| 2026-03-04 | llamacpp-jetson | 1 | 0.3 | 3986.4 | 3990.2 | 3991.6 | 3986.4 | 32.1 | 128.0 | 31.1 | 32.1 | - | 0.0 | 0% | 16 |
| 2026-03-04 | llamacpp-jetson-c4 | 4 | 0.5 | 7484.2 | 7505.4 | 7507.4 | 7484.2 | 68.5 | 128.0 | 58.5 | 17.1 | - | 0.0 | 0% | 36 |
| 2026-03-04 | ollama-jetson-c4 | 4 | 0.1 | 41578.9 | 42946.3 | 42946.3 | 41578.9 | 12.3 | 128.0 | 324.8 | 3.1 | - | 0.0 | 0% | 9 |
| 2026-03-04 | realizr-jetson-cpu | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 100.0% | 377086 |
| 2026-03-05 | realizar-jetson | 1 | 0.1 | 11581.4 | 11584.8 | 11584.9 | 3960.3 | 11.1 | 128.0 | 60.0 | 16.7 | 25.8 | 60.0 | 0% | 6 |
| 2026-03-05 | llamacpp-jetson | 1 | 0.3 | 3981.9 | 3986.1 | 3986.7 | 48.5 | 32.1 | 128.0 | 31.0 | 32.3 | 2101.8 | 31.0 | 0% | 16 |
| 2026-03-05 | ollama-jetson | 1 | 0.2 | 4258.4 | 4282.1 | 4283.4 | 426.4 | 30.1 | 128.0 | 30.2 | 33.2 | 239.2 | 30.2 | 0% | 15 |
| 2026-03-05 | realizar-jetson | 1 | 0.1 | 11583.0 | 11587.2 | 11587.9 | 3962.0 | 11.1 | 128.0 | 60.0 | 16.7 | 25.7 | 60.0 | 0% | 6 |
| 2026-03-05 | llamacpp-jetson | 1 | 0.3 | 3981.8 | 3982.9 | 3983.2 | 48.8 | 32.1 | 128.0 | 31.0 | 32.3 | 2090.6 | 31.0 | 5.9% | 17 |
| 2026-03-05 | realizar-jetson-nodp4a | 1 | 0.1 | 14016.7 | 14020.7 | 14021.0 | 3976.0 | 9.1 | 128.0 | 79.1 | 12.6 | 25.7 | 79.1 | 0% | 5 |
| 2026-03-05 | realizar-jetson-4warp | 1 | 0.1 | 12293.0 | 12299.4 | 12299.7 | 3963.6 | 10.4 | 128.0 | 65.6 | 15.2 | 25.7 | 65.6 | 0% | 5 |
| 2026-03-05 | realizar-jetson-2warp | 1 | 0.1 | 12721.0 | 12727.7 | 12728.8 | 3970.3 | 10.1 | 128.0 | 68.9 | 14.5 | 25.7 | 68.9 | 0% | 5 |
| 2026-03-05 | realizar-jetson-nodp4aq6k | 1 | 0.1 | 12601.1 | 12603.2 | 12603.4 | 3817.5 | 10.2 | 128.0 | 69.1 | 14.5 | 26.7 | 69.1 | 0% | 5 |
| 2026-03-05 | realizar-jetson | 1 | 0.1 | 11574.8 | 11585.2 | 11587.0 | 3956.3 | 11.1 | 128.0 | 60.0 | 16.7 | 25.8 | 60.0 | 0% | 6 |
| 2026-03-05 | llamacpp-jetson | 1 | 0.3 | 3980.9 | 3985.6 | 3991.3 | 48.8 | 32.1 | 128.0 | 31.0 | 32.3 | 2092.0 | 31.0 | 0% | 16 |
| 2026-03-05 | ollama-jetson | 1 | 0.1 | 10781.8 | 18705.0 | 20272.3 | 448.9 | 10.1 | 128.0 | 80.0 | 12.5 | 227.2 | 80.0 | 0% | 5 |
| 2026-03-08 | realizar-cpu | 4 | 0.1 | 37029.1 | 40395.4 | 40435.8 | 18846.8 | 13.8 | 128.0 | 143.7 | 7.0 | 5.5 | 143.7 | 0% | 8 |
| 2026-03-08 | ollama-cpu | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 100.0% | 517540 |
| 2026-03-08 | llamacpp-cpu | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 100.0% | 4 |
