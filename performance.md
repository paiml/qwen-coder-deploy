# LLM Inference Performance

## Production Methodology — RTX 4060 Laptop (PMAT-177/228, 2026-03-17)

Medium prompt (~102 tok), uniform:16,256 output, streaming, 5s warmup, 60s duration, locked 1900MHz.
Each runtime deployed in isolation via forjar (serial benchmarks).
realizr uses CUDA_MAX_BATCH=32 ITERATION_SCHEDULER=1 (PMAT-258, quality bug eliminated). Prior B16 data preserved below for reference.

### 4-Runtime Aggregate Throughput (tok/s)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 147.2 | 158.1 | 152.4 | 151.8 |
| 4 | 217.6 | 354.4 | 587.4 | 160.1 |
| 8 | 351.7 | 420.1 | 1,115.2 | 159.4 |
| 16 | 571.3 | 896.6 | 1,982.9 | 161.0 |
| 32 | 867.3 | 943.2 | 2,757.6 | 159.0 |
| 64 | 887.4 | — | 3,036.1 | — |
| 128 | 857.1 | — | 3,049.4 | — |

c=1-16: PMAT-177 (BATCH=32, confirmed ±2.6% by PMAT-224). c=32-128 realizr: PMAT-228 (BATCH=16, heterogeneous output). realizr asymptote ~880 tok/s at BATCH=16 (−41% vs BATCH=32 1500 tok/s, but correct output at all c).

### Scorecards (probador llm score)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 94 A | 97 A+ | 97 A+ | 78 B |
| 4 | 58 C | 73 B | 97 A+ | 58 C |
| 8 | 64 C+ | 65 C+ | 96 A+ | 58 C |
| 16 | 70 B | 72 B | 94 A | 58 C |
| 32 | 66 C+ | 51 C | 86 A- | 57 C |
| 64 | 68 C+ | — | 73 B | — |
| 128 | **67 C+** | — | 63 C+ | — |

PMAT-229 definitive combined scoring (4-runtime, best-in-class bonuses applied). realizr BATCH=16 at c=32-128, BATCH=32-equivalent at c=1-16. **Quality crossover preserved at c=128** (67 vs 63 C+) — realizr's stable decode (41 tok/s) beats vLLM's degraded decode (15 tok/s).

### Asymptotes (PMAT-192/195/197)

| Runtime | Asymptote | Saturation c | Architecture |
|---------|-----------|-------------|-------------|
| vLLM | 3,050 tok/s | c=64 | PagedAttention, continuous batching, CUTLASS GEMM |
| realizr (BATCH=16) | 880 tok/s (hetero) / 1,010 (fixed) | c=32 | Batch-and-step, queue+batch=16 (workaround) |
| realizr (BATCH=32) | 1,500 tok/s (⚠️ bug at c≥20 medium) | c=64 | Batch-and-step, queue+batch=32 |
| llama.cpp | 943 tok/s | c=32 | Fixed 16 slots, ncols-templated GEMV |
| ollama | 160 tok/s | c=1 (serial) | Serial FIFO |

### Quality Crossover (PMAT-192/229/233)

realizr **beats** vLLM at c=128: 67 C+ vs 63 C+.
BATCH=16 decode floor 57 tok/s (vs BATCH=32 49 tok/s); vLLM degrades to 15 tok/s.
Same-session c=64 decode: realizr 57.5 vs vLLM 50.5 — **realizr wins per-request decode 1.14×** despite 3.5× aggregate deficit. Crossover mechanism: BATCH=16 caps KV scan growth, preserving per-request quality at high concurrency.

### BATCH=16 vs BATCH=32 Decode Preservation (PMAT-231/232)

| c | BATCH=32 dec | BATCH=16 dec | B32 pres | B16 pres | Note |
|---|-------------|-------------|----------|----------|------|
| 1 | 148.3 | 149.2 | 100% | 100% | same |
| 4 | 81.7 | 82.4 | 55% | 55% | same |
| 16 | 72.2 | 71.3 | 49% | 48% | same |
| 32 | 69.7† | 56.5 | 47% | 38% | B32 bug-inflated |
| 64 | 48.9 | **57.5** | 33% | **39%** | B16 wins +17.6% |
| 128 | 48.8 | **57.1** | 33% | **38%** | B16 wins +17% |

†BATCH=32 c=32 decode inflated by quality bug (avg_tok=67 vs expected ~136, PMAT-232 confirmed).
c=64 confirmed clean same-session (avg ~135 tokens both). Tradeoff: −40% aggregate for +17% per-request decode at c≥64.

### Iso-Quality Gap (PMAT-187/193/230, updated PMAT-263)

**B32 iter sched vs B16 batch-and-step:**

| Constraint | B16 B&S gap | B32 iter gap | Δ |
|------------|-------------|-------------|---|
| ITL≤12ms | 18.7× | 18.7× | same (c=1 for both) |
| ITL≤15ms | 4.8× (r c=16) | 9.5× (r c=4) | **worse** (ITL +8.7% at c=4-16) |
| ITL≤21ms | 3.5× (r c=128) | **2.0×** (r c=128) | **−43%** (aggregate +71%) |
| Score≥70 | 5.3× (r c=16 571) | **2.1×** (r c=32 1464) | **−60%** |
| Score≥75 | — | **2.1×** (r c=32 1464) | new (B16 never reached 75 at c≥4) |

**Trade-off:** Iteration scheduler increases ITL (+9-26% at c=4-16), widening the strict-ITL gap. But +33-69% aggregate throughput more than compensates at relaxed constraints. Score-based gap: **60% improvement** (5.3× → 2.1×). After Phase 1+CB: projected ~1.0-1.4× across all constraints.

### Scaling Efficiency (PMAT-235, updated PMAT-262)

Scaling efficiency = (agg_c / agg_1) / c. Perfect = 1.0.

| c | vLLM | realizr B32 iter | realizr B16 B&S | llama.cpp | ollama |
|---|------|-----------------|-----------------|-----------|--------|
| 4 | **0.96** | 0.49 | 0.37 | 0.56 | 0.26 |
| 8 | **0.91** | 0.42 | 0.30 | 0.33 | 0.13 |
| 16 | **0.81** | 0.37 | 0.24 | 0.35 | 0.07 |
| 32 | **0.57** | 0.31 | 0.18 | 0.19 | 0.03 |

B32 iter sched: +33% (c=4), +41% (c=8), +54% (c=16), +69% (c=32) scaling efficiency vs B16 B&S. realizr now matches llama.cpp at c=8 (0.42 vs 0.33) and c=16 (0.37 vs 0.35). vLLM still 2.0× more efficient at c=4 (0.96 vs 0.49).

### Tail Latency & Jitter (PMAT-234)

| Runtime | Jitter (max) | Error rate | TTFT tail (P99/P50) |
|---------|-------------|------------|---------------------|
| ollama | **1.01×** (perfect) | 0% | 1.10-1.14× |
| vLLM | 1.10× | 0% | 1.06-2.49× |
| realizr (B16 B&S) | 1.18× (c≤64), 1.49× (c=128) | 0% | **1.01-1.12×** (tightest) |
| realizr (B32 iter) | **1.09-1.17×** (all c) | 0% | — |
| llama.cpp | **1.38×** (worst) | **1-2%** | 1.10-1.57× |

Jitter = TPOT P99 / ITL P50. llama.cpp is the only runtime with errors (avg_tok ~92 vs ~136, ctx_size constraint).

### Three-Way Kernel Architecture (PMAT-209→217, nsys/ncu profiling)

| Dimension | realizr | llama.cpp | vLLM |
|-----------|---------|-----------|------|
| Weight format | Q4K GGUF + DP4A INT8 | Q4K/Q6K GGUF | AWQ INT4 → FP16 GEMM |
| Decode dispatch | CUDA graph (M=1 only) | ncols-templated GEMV (M=1-4) | CUTLASS GEMM (M=batch) |
| Dominant kernel | fused_gate_up DP4A 49.8% | Q4K ncols=4 GEMV 21% | CUTLASS GEMM 95.7% |
| Per-call time | 84µs (fused) | 37.6µs (Q4K ncols=4) | 2,165µs |
| Kernel types | 44+ | ~35 | ~15 |
| Launches/step | 771 (no graph), 1 (graph) | ~350 | ~113 |

### CUDA API Root Cause (PMAT-217, updated PMAT-266)

At c=4, realizr CPU spends **80-82%** of time blocked in cuStreamSynchronize (10.4-10.7ms median).
M=1 graph is invalid for M>1 → 771 individual kernel launches per decode step.
**PMAT-266: This profile is IDENTICAL between batch-and-step and iteration scheduler** — scheduling changes don't affect CUDA dispatch.
llama.cpp: 3,579 graph launches, dynamic re-capture, 0.46µs median sync.
vLLM: 11,467 pre-captured graph launches, event-based sync, 18.9µs median sync.

**Per-step c=4 budget**: 1.2ms launch + 10ms GPU kernels + 0.3ms H2D = ~12ms → 290 tok/s.
**PMAT-266 correction**: Graph capture saves ~1.2ms launch overhead (17% improvement), but GPU kernel compute (10ms, 44 kernels) is the floor. vs vLLM: 2.17ms (1 CUTLASS GEMM) = **4.6× GPU kernel gap.**
**Revised projection**: per-M graph → ~340 tok/s at c=4 (was ~400). Reaching 0.80×+ vLLM requires **kernel fusion** (44 → ~15 kernels).

### Prompt-Length Sensitivity (PMAT-219/220)

Competitive ratios (realizr/llama.cpp) across prompt lengths with heterogeneous output:

| Workload | c=1 | c=4 | c=8 | c=16 |
|----------|-----|-----|-----|------|
| short + hetero | 0.95 | 0.70 | 0.95 | 0.78 |
| medium + hetero | 0.93 | 0.61 | 0.85 | 0.65 |
| long + hetero | 0.91 | 0.55 | 0.75 | — † |

realizr/vLLM ratios:

| Workload | c=1 | c=4 | c=8 | c=16 |
|----------|-----|-----|-----|------|
| short + hetero | 0.97 | 0.41 | 0.34 | 0.32 |
| medium + hetero | 0.96 | 0.37 | 0.32 | 0.30 |
| long + hetero | 0.94 | 0.33 | 0.28 | 0.24 |

† realizr c=16 long: output degradation (p50=10 tokens).
Prompt length monotonically hurts realizr. FP8 prefill BW overhead scales linearly.

### Reproducibility (PMAT-216)

Fresh benchmarks on 2026-03-16 confirm <1% delta vs PMAT-177 across all runtimes and concurrency levels.

---

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

### Measurement Stability — Serial c=1 Same-Session (PMAT-236, Mar 17)

| Runtime | PMAT-177 | Serial c=1 | Δ |
|---------|----------|-----------|---|
| realizr | 147.2 | 149.2 | +1.4% |
| vLLM | 152.4 | 153.5 | +0.7% |
| llama.cpp | 158.1 | 158.9 | +0.5% |
| ollama | 151.8 | 160.1 | +5.5% |

7.3% total spread (149.2-160.1). 3/4 runtimes within 1.4% of PMAT-177 baseline. Ollama variance largest — M=1 exclusive decode is thermal-sensitive. Ranking stable: ollama > llama.cpp > vLLM > realizr.

### Measurement Stability — Serial c=4 Same-Session (PMAT-237, Mar 18)

| Runtime | PMAT-177 | Serial c=4 | Δ | Scaling (c=4/c=1) |
|---------|----------|-----------|---|-------------------|
| vLLM | 587.4 | 587.1 | −0.1% | 3.86× |
| llama.cpp | 354.4 | 355.0 | +0.2% | 2.24× |
| realizr | 217.6 | 217.6 | 0.0% | 1.46× |
| ollama | 160.1 | 159.7 | −0.3% | 1.00× |

All 4 within 0.3% of PMAT-177 (tighter than c=1). Batching architecture divergence: vLLM 3.86× > llama.cpp 2.24× > realizr 1.46× > ollama 1.00×.

### Measurement Stability — Serial c=8 Same-Session (PMAT-238, Mar 18)

| Runtime | PMAT-177 | Serial c=8 | Δ | Scaling (c=8/c=1) | Decode |
|---------|----------|-----------|---|-------------------|--------|
| vLLM | 1,115.2 | 1,114.7 | −0.05% | 7.33× | 142.8 |
| llama.cpp | 420.1 | 406.0 | −3.4% | 2.56× | 51.9 |
| realizr | 351.7 | 355.5 | +1.1% | 2.38× | 75.1 |
| ollama | 159.4 | 157.3 | −1.3% | 0.98× | 158.4 |

**Per-request decode crossover at c=8:** realizr 75.1 vs llama.cpp 51.9 = **1.45× realizr advantage** (FP8 tensor core at M≥5). Aggregate still 12.4% below llama.cpp (scheduling overhead).

### Measurement Stability — Serial c=16 Same-Session (PMAT-240, Mar 18)

| Runtime | PMAT-177 | Serial c=16 | Δ | Decode | r/lc dec |
|---------|----------|------------|---|--------|----------|
| vLLM | 1,982.9 | 1,980.1 | −0.1% | 127.4 | — |
| llama.cpp | 896.6 | 853.5 | −4.8% | 56.3 | — |
| realizr | 571.3 | 583.6 | +2.2% | 72.1 | 1.28× |
| ollama | 161.0 | 156.8 | −2.6% | 157.9 | — |

realizr decode still beats llama.cpp at c=16: 72.1 vs 56.3 = 1.28× (narrowing from 1.45× at c=8). llama.cpp variance grows with c (−4.8%). vLLM ≤0.1% at all c levels.

### Same-Session Serial Scoring (PMAT-241, Mar 18)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 95 A+ | 97 A+ | 98 A+ | 74 B |
| 4 | 58 C | 73 B | 98 A+ | 58 C |
| 8 | **65 C+** | 62 C+ | 97 A+ | 57 C |
| 16 | **71 B** | **71 B** | 94 A | 57 C |

Matches PMAT-229 production scoring ±2 points. realizr ties llama.cpp at c=16 (71 B) despite 46% aggregate deficit — decode, errors, and tail compensate. realizr overtakes at c=8 (65 vs 62).

### TTFT Scaling Curve (PMAT-242, Mar 18, same-session serial)

| c | realizr | llama.cpp | vLLM | ollama | r/vLLM |
|---|---------|-----------|------|--------|--------|
| 1 | 18.7 | 12.0 | 13.9 | 87.0 | 1.3× |
| 4 | 76.5 | 24.7 | 21.7 | 3,108 | 3.5× |
| 8 | 148.0 | 44.5 | 23.2 | 6,634 | 6.4× |
| 16 | 279.0 | 60.4 | 26.2 | 14,952 | **10.6×** |

TTFT growth c=1→16: vLLM 1.9× (continuous batching) < llama.cpp 5.0× < realizr 14.9× < ollama 172×. realizr TTFT tail **tightest** at c=4,16 (P99/P50 = 1.02×) vs vLLM **worst** (2.33× at c=16).

### ITL Jitter Scaling (PMAT-243, Mar 18, same-session serial, TPOT P99 / ITL P50)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 1.00× | 1.01× | 1.00× | 1.01× |
| 4 | 1.05× | 1.03× | 1.01× | 1.02× |
| 8 | 1.05× | 1.08× | 1.02× | 1.01× |
| 16 | 1.09× | **1.49×** | 1.04× | 1.01× |

llama.cpp jitter worst (1.49× at c=16). realizr tight (≤1.09×). llama.cpp only runtime with errors (1-2.9%).

### Competitive Advantage Matrix (PMAT-244, Mar 18)

| Metric | c=1 | c=4 | c=8 | c=16 |
|--------|-----|-----|-----|------|
| Aggregate | llama.cpp | **vLLM** | **vLLM** | **vLLM** |
| Decode | llama.cpp | **vLLM** | **vLLM** | **vLLM** |
| TTFT tail | vLLM | **realizr** | realizr | **realizr** |
| Jitter | tie | vLLM | vLLM | **realizr** |
| Errors | tie | tie | tie | tie |
| Score | **vLLM** | **vLLM** | **realizr** | tie (r/lc) |

realizr profile: most predictable, error-free. Quality ≥ llama.cpp at c≥8 despite aggregate deficit.

### Serial c=32 Same-Session (PMAT-245/246, Mar 18)

| Runtime | PMAT-177 | Serial c=32 | Δ | Decode |
|---------|----------|------------|---|--------|
| vLLM | 2,757.6 | 2,900.6 | +5.2% | 93.5 |
| llama.cpp | 943.2 | 888.5 | −5.8% | 57.7 |
| realizr | 867.3 | 868.6 | +0.2% | 57.1 |

**PMAT-246 regression falsified**: PMAT-245 showed llama.cpp at 426.7 tok/s (−54.7%), but re-verification on same deploy (where c=16 = 863.1 tok/s) yields **888.5 tok/s** (−5.8%, within normal variance). Effective slots: 15.4/16 (near-max utilization). The PMAT-245 anomaly was transient — likely a llama.cpp HEAD build regression that resolved on subsequent deploy. 0.9% error rate (6/635). Per-request decode: realizr 57.1 ≈ llama.cpp 57.7 (near-parity at c=32).

### Serial c=64/128 Same-Session (PMAT-247, Mar 18)

| Runtime | c | PMAT-177 | Serial | Δ | Decode |
|---------|---|----------|--------|---|--------|
| vLLM | 64 | 3,036.1 | 3,151.0 | +3.8% | 50.4 |
| vLLM | 128 | 3,049.4 | 3,086.3 | +1.2% | 24.4 |
| realizr | 64 | 887.4 | 891.4 | +0.5% | 57.7 |
| realizr | 128 | 857.1 | 885.4 | +3.3% | 57.2 |

**Both runtimes at asymptote.** realizr: 885-891 tok/s (BATCH=16 ceiling, decode 57.2-57.7 stable). vLLM: 3,050-3,150 tok/s (continuous batching saturated). Per-request decode: realizr 57.2 > vLLM 24.4 at c=128 — **realizr wins per-request decode 2.34×** at highest concurrency. vLLM per-request decode halves each doubling (50.4→24.4) while realizr holds constant (BATCH=16 cap prevents further decode degradation). 0% errors both runtimes. TTFT: realizr 16.6s vs vLLM 133ms at c=128 (125× gap). Completes serial isolated curve c=1→128 for realizr+vLLM.

### BATCH=32 + Iteration Scheduler (PMAT-258, Mar 18)

PMAT-221 quality bug **eliminated** by iteration scheduler. Slot-level recycling avoids KV corruption.

| c | Iter B16 | Iter B32 | Δ | avg_tok | errors |
|---|---------|---------|---|---------|--------|
| 4 | 291.3 | 290.1 | −0.4% | 132 | 0% |
| 8 | 494.8 | 494.4 | −0.1% | 129 | 0% |
| 16 | 884.8 | 880.4 | −0.5% | 126 | 0% |
| 32 | 873.7 | **1,463.8** | **+67.5%** | 126 | 0% |
| 64 | 882.0 | **1,494.1** | **+69.4%** | 135 | 0% |
| 128 | 885.6 | **1,514.7** | **+71.0%** | 134 | 0% |

**Asymptote raised from 885 to 1,515 tok/s** (+71%). realizr now 1.55× llama.cpp at c=32 and 0.50× vLLM at c=128 (was 0.28× with B&S B16). B32 identical to B16 at c≤16 (only fills min(c,BATCH) slots). PMAT-221 was a scheduling bug, not a kernel bug.

### Phase 1 Projections (PMAT-265, corrected by PMAT-266, Mar 18)

Updated projections from B32 iter sched baseline using PMAT-264 decomposition. **PMAT-266 nsys correction: original 0.81× projection was overly optimistic.** The 82% cuStreamSync time is mostly GPU kernel compute (10ms), not CPU launch overhead (1.2ms). Graph capture saves ~2ms/step (17%), not the full sync duration.

**Scenario A: Per-M graph capture only** (decode_rate +17%, launch overhead eliminated):

| c | Current | ~~PMAT-265~~ | **PMAT-266 revised** | vs vLLM |
|---|---------|-------------|---------------------|---------|
| 4 | 290 | ~~474~~ | **340** | **0.58×** |
| 8 | 494 | ~~914~~ | **578** | **0.52×** |
| 16 | 880 | ~~1,627~~ | **1,030** | **0.52×** |
| 32 | 1,464 | ~~2,222~~ | **1,713** | **0.62×** |

**Scenario B: Per-M graph + CB** (decode_rate +17%, scheduling → 0.97×):

| c | Current | ~~PMAT-265~~ | **PMAT-266 revised** | vs vLLM |
|---|---------|-------------|---------------------|---------|
| 4 | 290 | ~~541~~ | **388** | **0.66×** |
| 8 | 494 | ~~1,043~~ | **660** | **0.59×** |
| 16 | 880 | ~~1,856~~ | **1,174** | **0.59×** |
| 32 | 1,464 | ~~2,536~~ | **1,954** | **0.71×** |

**Scenario C: Per-M graph + kernel fusion + CB** (decode_rate → 0.85×, requires fusing 44 kernels):

| c | Current | Projected | vs vLLM |
|---|---------|-----------|---------|
| 4 | 290 | **474** | **0.81×** |
| 8 | 494 | **914** | **0.82×** |
| 16 | 880 | **1,627** | **0.82×** |
| 32 | 1,464 | **2,222** | **0.81×** |

**PMAT-266 key insight:** Per-M graph capture alone gives **0.52-0.62× vLLM** (not 0.81×). Reaching 0.80×+ requires **kernel fusion** — reducing 44+ small kernels (10ms total) toward vLLM's single CUTLASS GEMM (2.17ms). The 4.6× GPU compute gap is the binding constraint, not launch overhead.

### nsys CUDA API Trace — Iteration Scheduler (PMAT-266, Mar 18)

nsys trace of B32 iteration scheduler at c=4 (90s capture, yoga RTX 4060L):

**CUDA API Summary:**

| API | Time% | Total | Calls | Median |
|-----|-------|-------|-------|--------|
| cuStreamSynchronize | **80.5%** | 44.9s | 4,510 | 10.7ms |
| cuLaunchKernel | 10.7% | 6.0s | 2,984,497 | 1.8µs |
| cuMemcpyHtoD | 5.2% | 2.9s | 544,921 | 2.8µs |
| cuGraphLaunch | 0.0% | 11ms | 351 | 29µs |

**GPU Kernel Summary (top 5):**

| Kernel | Time% | Total | Instances | Avg |
|--------|-------|-------|-----------|-----|
| batched_hw_dp4a_q4k_gemv | **35.0%** | 18.0s | 514,284 | 35µs |
| batched_incremental_attention | **24.8%** | 12.7s | 107,404 | 119µs |
| sm89_xmma_gemm (FP8, large) | 17.0% | 8.7s | 61,667 | 142µs |
| sm89_xmma_gemm (FP8, small) | 11.5% | 5.9s | 193,032 | 31µs |
| batched_rmsnorm_vectorized | 1.6% | 0.8s | 225,419 | 3.6µs |

**Per-step budget (M=4):** ~12ms total — GPU kernels ~10ms (DP4A 4ms + attention 2.8ms + FP8 GEMM 3.3ms), launch 1.2ms (661 launches), H2D 0.3ms (120 copies). vs vLLM: single CUTLASS GEMM 2.17ms → **4.6× GPU kernel compute gap.**

**Critical finding:** cuStreamSynchronize profile is **identical to PMAT-217** (batch-and-step: 82.4%, 10.4ms median). The iteration scheduler does NOT change the CUDA dispatch pattern — scheduling improvement is purely CPU-side slot management. The GPU kernel pipeline is the same 44+ kernel sequence regardless of scheduler.

### Gap Decomposition Update (PMAT-264, Mar 18)

The PMAT-179 2-factor model (gap = decode_rate × scheduling_utilization) recomputed with B32 iter sched:

| c | r/v agg | decode_rate | sched_util | sched_util (B16 B&S) |
|---|---------|------------|-----------|---------------------|
| 4 | 0.49× | 0.52× | **0.96×** | 0.55× |
| 8 | 0.44× | 0.46× | **0.96×** | 0.52× |
| 16 | 0.44× | 0.46× | **0.96×** | 0.52× |
| 32 | 0.53× | 0.56× | **0.94×** | 0.67× |
| 64 | 0.49× | 1.04× | **0.47×** | — |
| 128 | 0.50× | 2.08× | **0.24×** | — |

**The iteration scheduler closes the scheduling gap.** At c≤32, scheduling utilization is **94-96%** (near-vLLM). The remaining 0.44-0.53× ratio is **almost purely decode kernel efficiency** — per-M CUDA graph capture is the primary fix. At c≥64, realizr's decode advantage (1.04-2.08×) is offset by queueing penalty (0.47-0.24×) from the BATCH=32 cap.

**Phase 1 priority implications:** (1) Per-M graph capture → fixes decode_rate at c≤32 (0.46→~0.85× projected). (2) Paged KV → removes BATCH cap for c>32. CB confirmed as definitively higher-value.

### B32 Crossover Precision (PMAT-261, Mar 18)

B32 iteration scheduler vs vLLM decode/ITL at high concurrency:

| c | realizr B32 dec | vLLM dec | r/v dec | realizr B32 ITL | vLLM ITL | r/v ITL |
|---|----------------|---------|---------|----------------|---------|---------|
| 64 | 49.2 | 50.4 | **0.98×** | 20.3 | 19.8 | **1.03×** |
| 80 | 49.0 | 39.3 | **1.25×** | 20.4 | 25.5 | **0.80×** |
| 128 | 49.5 | 24.4 | **2.03×** | 20.2 | 41.0 | **0.49×** |

**Crossover shifted c=64 → c≈66** (minimal, 2 units). B32 decode constant ~49 tok/s (vs B16 57, −14%). vLLM decay unchanged. BATCH=32 trades 14% per-request decode for 71% aggregate throughput — crossover barely moves because vLLM's linear decay dominates. Advantage at c=128: 2.03× (was 2.35× at B16).

Compare B16 (PMAT-255): crossover at c=64, advantage 1.14→2.35×. B32 compresses the advantage window by ~14% but shifts entry point only 2 c-units.

### Iteration Scheduler Heterogeneity (PMAT-260, Mar 18)

B32 iteration scheduler, fixed:128 vs uniform:16,256 output:

| c | uniform:16,256 | fixed:128 | Penalty | PMAT-254 (B16 b&s) |
|---|----------------|-----------|---------|---------------------|
| 4 | 290.1 | 312.6 | **7.2%** | 31% |
| 8 | 494.4 | 553.2 | **10.6%** | 36% |
| 16 | 880.4 | 980.1 | **10.2%** | 42% |
| 32 | 1,463.8 | 1,621.8 | **9.7%** | 14% |

**Iteration scheduler reduces heterogeneity penalty from 31-42% to 7-11%** (4× improvement). Per-slot recycling reclaims most of the waste from variable output lengths — when a short request finishes, its slot is immediately available. Remaining 7-11% penalty is from KV memory fragmentation (fixed-size KV slots still pre-allocate max capacity).

**Paged KV ROI revision**: At c=16, paged KV now recovers +100 tok/s (880→980 tok/s, 1.11×) vs +423 tok/s (584→1,006 tok/s, 1.72×) with B16 batch-and-step. Marginal ROI decreased 4.2× because iteration scheduler already captures most scheduling waste. **CB (mid-batch joins + per-M graphs) is now definitively the higher-value Phase 1 target** — the scheduling utilization gap (0.45-0.50× vs projected 0.97×) dominates over the residual heterogeneity gap (7-11%).

### Iteration Scheduler Benchmark (PMAT-257, Mar 18)

`ITERATION_SCHEDULER=1` (existing framework, zero code changes, BATCH=16):

| c | Batch-and-step | Iteration sched | Δ agg | TTFT b&s | TTFT iter | Δ TTFT | Score b&s | Score iter |
|---|---------------|----------------|-------|----------|-----------|--------|-----------|------------|
| 1 | 147.2 | 147.2 | 0.0% | 18.7 | 18.7 | 0.0% | 94 A | 95 A+ |
| 4 | 217.6 | **291.3** | **+33.8%** | 82.0 | **42.9** | **−47.7%** | 58 C | **70 B** |
| 8 | 351.7 | **494.8** | **+40.7%** | 178.0 | **46.3** | **−74.0%** | 64 C+ | **75 B** |
| 16 | 571.3 | **884.8** | **+54.9%** | 279.0 | **47.8** | **−82.9%** | 70 B | **78 B** |
| 32 | 867.3 | 873.7 | +0.7% | 2,235 | 2,246 | +0.5% | 66 C+ | 67 C+ |
| 64 | 891.4 | 882.0 | −1.1% | — | 7,245 | — | 68 C+ | 68 C+ |

Hits BATCH=16 asymptote at c=16 (was c=32). TTFT collapses because requests join mid-decode instead of waiting for batch-wide prefill. ITL trade-off: +9-26% at c=4-16. At c≥32 both schedulers equivalent. Revised r/v ratio: 0.45-0.50× (was 0.29-0.37×). **Single highest-value zero-implementation-cost improvement.**

### Phase 1 Readiness Audit (PMAT-256, Mar 18)

Codebase review of realizr for continuous batching readiness:

| Area | Status | Files | LOC Change |
|------|--------|-------|-----------|
| KV cache (paged_kv/) | **READY** | mod_paged.rs (444 LOC) | 0 |
| Memory allocator | **READY** | core.rs (per-layer HashMap) | 0 |
| Batch scheduler | **BLOCKER** | cuda_batch_scheduler.rs (430), iteration_scheduler.rs (459) | +500-700 |
| CUDA graphs | **RISK** | core.rs (decode_graph, batched_decode_graphs) | +200-300 |

**Total Phase 1: ~1,000-1,400 LOC** across 4-5 files. PagedKvCache already implements dynamic page allocation, CoW, and defragmentation — no changes needed. Batch scheduler is the binding blocker: legacy `process_cuda_batch()` is synchronous batch-and-step; `iteration_scheduler.rs` framework exists (opt-in via `ITERATION_SCHEDULER=1`) but incomplete. CUDA graph invalidation strategy unclear — PMAT-042 workspace realloc risk of silent data corruption. Critical path: graph safety → async iteration loop → mid-batch slot addition → prefill chunking.

### Crossover Precision (PMAT-255, Mar 18)

| c | realizr agg | vLLM agg | realizr dec | vLLM dec | r/v dec | realizr ITL | vLLM ITL | r/v ITL |
|---|------------|---------|------------|---------|---------|------------|---------|---------|
| 64 | 891.4 | 3,151.0 | 57.7 | 50.4 | **1.14×** | 17.3 | 19.8 | **0.87×** |
| 80 | 883.9 | 3,074.3 | 57.3 | 39.3 | **1.46×** | 17.4 | 25.5 | **0.68×** |
| 96 | 877.9 | 3,120.2 | 57.4 | 33.0 | **1.74×** | 17.4 | 30.3 | **0.58×** |
| 112 | 885.9 | 3,137.0 | 57.5 | 28.5 | **2.02×** | 17.4 | 35.1 | **0.50×** |
| 128 | 885.4 | 3,086.3 | 57.2 | 24.4 | **2.35×** | 17.5 | 41.0 | **0.43×** |

**Crossover at c=64** (not c≈96 as previously interpolated). realizr decode constant 57.2-57.7 (BATCH=16 floor); vLLM decays linearly 50.4→24.4. realizr ITL constant 17.3-17.5ms; vLLM grows 19.8→41.0ms. Aggregate: realizr ~880-890 (BATCH=16 asymptote), vLLM ~3,070-3,150 (CB saturated). 0% errors both runtimes at all c levels.

### Output-Length Sensitivity (PMAT-254, Mar 18)

**realizr aggregate (tok/s):**

| Output | c=4 | c=8 | c=16 | c=32 |
|--------|-----|-----|------|------|
| fixed:32 | 210.6 | 473.6 | 748.9 | 751.0 |
| fixed:128 | **316.3** | **553.7** | **1,006.3** | **1,008.0** |
| fixed:256 | 304.3 | 536.7 | 1,011.6 | 1,011.5 |
| uniform:16,256 | 217.6 | 355.5 | 583.6 | 868.6 |

**vLLM aggregate (tok/s):**

| Output | c=4 | c=8 | c=16 | c=32 |
|--------|-----|-----|------|------|
| fixed:32 | 553.4 | 1,027.3 | 1,788.2 | 2,713.2 |
| fixed:128 | 589.2 | 1,115.8 | 2,030.9 | 3,205.5 |
| fixed:256 | 593.1 | 1,130.9 | 2,004.1 | 3,242.9 |
| uniform:16,256 | 587.1 | 1,114.7 | 1,980.1 | 2,900.6 |

**Heterogeneity penalty** (uniform vs fixed:128): realizr **31-42%** at c=4-16, **14% at c=32** (queuing dominates). vLLM **0-2.5%** at c=4-16, **9.5% at c=32**. PagedAttention eliminates heterogeneity cost.

**Paged KV ROI**: At c=16, paged KV recovers +423 tok/s (1.72×) from 584→1,006. But realizr fixed:128 is still 0.51× vLLM — **CB is still needed after paged KV.** Combined Phase 1 projected 0.97× vLLM (PMAT-180).

### Prompt-Length Sensitivity (PMAT-253, Mar 18)

**realizr aggregate (tok/s):**

| Profile | c=1 | c=4 | c=8 | c=16 |
|---------|-----|-----|-----|------|
| short (23 tok) | 148.8 | 239.8 | 387.9 | 655.4 |
| medium (102 tok) | 147.2 | 217.6 | 355.5 | 583.6 |
| long (~500 tok) | 142.2 | 190.8 | 305.4 | 501.5 |

**vLLM aggregate (tok/s):**

| Profile | c=1 | c=4 | c=8 | c=16 |
|---------|-----|-----|-----|------|
| short (23 tok) | 152.6 | 588.6 | 1,137.4 | 2,049.7 |
| medium (102 tok) | 152.3 | 587.1 | 1,114.7 | 1,980.1 |
| long (~500 tok) | 152.1 | 569.8 | 1,095.7 | 1,808.0 |

**Long-prompt penalty (vs medium)**: realizr −3.4% (c=1), −12.3% (c=4), −14.1% (c=8), −14.1% (c=16). vLLM −0.1% (c=1), −2.9% (c=4), −1.7% (c=8), −8.7% (c=16). TTFT long/short ratio: realizr 3.0× (c=1) → 7.7× (c=16); vLLM 1.0-1.1×. **Decision gate BORDERLINE**: 12-14% at c≥4 between 10% skip and 15% required thresholds. Phase 0 (fused Q4K GEMM) is optional — not mandatory for Phase 1. Short-prompt boost (vs medium): realizr +9-12% at c≥4.

### Extended Competitive Advantage Matrix (PMAT-252, updated PMAT-261 for B32)

| Metric | c=1 | c=4 | c=8 | c=16 | c=32 | c=64 | c=128 |
|--------|-----|-----|-----|------|------|------|-------|
| Aggregate | llama.cpp | **vLLM** | **vLLM** | **vLLM** | **vLLM** | **vLLM** | **vLLM** |
| Decode (B32) | ollama | ollama | ollama | ollama | vLLM | vLLM† | **realizr** |
| TTFT | llama.cpp | **vLLM** | **vLLM** | **vLLM** | **vLLM** | **vLLM** | **vLLM** |
| ITL (B32) | ollama | ollama | ollama | ollama | vLLM | vLLM† | **realizr** |
| Errors | r/v/o | r/v/o | r/v/o | r/v/o | r/v | r/v | r/v |
| Score | vLLM | **vLLM** | **vLLM** | **vLLM** | **vLLM** | vLLM | **realizr** |

†B32 crossover at c≈66 (was c=64 at B16). At c=64: realizr 49.2 vs vLLM 50.4 (0.98×); at c=80: realizr 49.0 vs vLLM 39.3 (1.25×).
**Four phase boundaries:** (1) c=1-4 parity, (2) c=5-7 FP8 crossover, (3) c=8-64 vLLM dominance, (4) c≈66-128 quality crossover — realizr wins decode, ITL, errors, AND score at c=128.

### ITL Crossover Analysis (PMAT-251, Mar 18)

| c | realizr | llama.cpp | vLLM | ollama | r/v | realizr growth | vLLM growth |
|---|---------|-----------|------|--------|-----|----------------|-------------|
| 1 | 6.7 | 6.3 | 6.5 | 6.2 | 1.03× | 1.0× | 1.0× |
| 4 | 12.1 | 11.1 | 6.7 | 6.2 | 1.81× | 1.8× | 1.0× |
| 8 | 13.3 | 19.3 | 7.0 | 6.3 | 1.90× | 2.0× | 1.1× |
| 16 | 13.9 | 17.7 | 7.9 | 6.3 | 1.76× | 2.1× | 1.2× |
| 32 | 17.5 | 17.3 | 10.7 | — | 1.64× | 2.6× | 1.6× |
| 64 | **17.3** | — | 19.8 | — | **0.87×** | 2.6× | 3.0× |
| 128 | **17.5** | — | 41.0 | — | **0.43×** | 2.6× | 6.3× |

**ITL crossover at c=64:** realizr 17.3ms < vLLM 19.8ms. At c=128: 17.5ms vs 41.0ms — **realizr ITL 2.3× better.** realizr ITL stabilizes at 17.3-17.5ms for c≥32 (BATCH=16 floor). vLLM ITL grows 6.3× from c=1→128 (no floor). This mirrors the decode crossover and is the fundamental quality advantage that drives the scoring crossover at c=128. Error rates: llama.cpp 1-3% at all c, all others 0%.

### TTFT Scaling Full Curve (PMAT-250, Mar 18)

| c | realizr | llama.cpp | vLLM | ollama | r/v |
|---|---------|-----------|------|--------|-----|
| 1 | 18.7 | 12.0 | 13.9 | 87.0 | 1.3× |
| 4 | 76.5 | 24.7 | 21.7 | 3,108 | 3.5× |
| 8 | 148.0 | 44.5 | 23.2 | 6,634 | 6.4× |
| 16 | 279.0 | 60.4 | 26.2 | 14,952 | 10.6× |
| 32 | **2,234.8** | **1,646.3** | 36.4 | — | **61.4×** |
| 64 | 7,180.0 | — | 70.8 | — | 101.4× |
| 128 | 16,593.8 | — | 133.3 | — | **124.5×** |

**Phase transition at c=32:** Both realizr and llama.cpp have 16-slot architectures (BATCH=16 / --parallel 16). At c>16, queuing creates a TTFT cliff: realizr 279→2,235ms (8.0× per doubling), llama.cpp 60→1,646ms (27.3×). vLLM: smooth 1.4-1.9× per doubling (continuous batching). TTFT growth factor c=1→128: realizr 887× vs vLLM 9.6×. The gap widens from 1.3× (c=1) to 124.5× (c=128). Despite this, realizr maintains tightest TTFT tail (P99/P50 ≤1.1×) — deterministic batch scheduling.

### Per-Request Decode Decay Curve (PMAT-249, Mar 18)

| c | realizr | llama.cpp | vLLM | ollama | r/l | r/v |
|---|---------|-----------|------|--------|-----|-----|
| 1 | 149.2 | 158.9 | 153.5 | 160.1 | 0.94× | 0.97× |
| 4 | 82.4 | 89.7 | 149.7 | 160.8 | 0.92× | 0.55× |
| 8 | **75.1** | 51.9 | 142.8 | 158.4 | **1.45×** | 0.53× |
| 16 | **72.1** | 56.3 | 127.4 | 157.9 | **1.28×** | 0.57× |
| 32 | 57.1 | 57.7 | 93.5 | — | 0.99× | 0.61× |
| 64 | **57.7** | — | 50.4 | — | — | **1.14×** |
| 128 | **57.2** | — | 24.4 | — | — | **2.34×** |

**Decode preservation** (decode_c / decode_1):
- ollama: 99-100% (serial, no degradation)
- vLLM: 98%→93%→83%→61%→33%→**16%** (no floor — per-request quality collapses)
- realizr: 55%→50%→48%→38%→39%→**38%** (stabilizes at BATCH=16 floor)
- llama.cpp: 56%→33%→35%→36% (notch at c=8, then recovers)

**Three crossover points:** (1) realizr beats llama.cpp decode at c=8 (1.45×), (2) realizr approaches parity at c=32 (0.99×), (3) realizr beats vLLM decode at c=64 (1.14×, widens to 2.34× at c=128). realizr's BATCH=16 cap is both its ceiling AND its floor — it prevents further decode degradation that vLLM suffers.

### Definitive Serial Scoring Curve (PMAT-248, Mar 18)

| c | vLLM | realizr | llama.cpp | ollama |
|---|------|---------|-----------|--------|
| 1 | 98 A+ | 95 A+ | 97 A+ | 74 B |
| 4 | 98 A+ | 58 C | 73 B | 58 C |
| 8 | 97 A+ | 65 C+ | 62 C+ | 57 C |
| 16 | 94 A | 71 B | 71 B | 57 C |
| 32 | 89 A- | 66 C+ | 63 C+ | — |
| 64 | 73 B | 68 C+ | — | — |
| 128 | 63 C+ | **68 C+** | — | — |

**Quality crossover at c=128**: realizr 68 > vLLM 63 (BATCH=16 caps decode at 57 tok/s; vLLM collapses to 24.4). **realizr overtakes llama.cpp at c=8** (65 vs 62). Scores match PMAT-229 production scoring within ±2 points at c=1-16. probador 1.0.3.

### Scaling Curve Synthesis (PMAT-239, Mar 18)

**Per-request decode crossover at c=5-7** (at c=4: realizr 0.92× llama.cpp, at c=8: 1.45×).

**Marginal throughput (Δ agg / Δ c):**
- vLLM: +145 (c=1→4), +132 (c=4→8) — nearly constant (continuous batching)
- llama.cpp: +65 → +13 — **collapses 80%** (fixed-slot saturation)
- realizr: +23 → +35 — increasing as batch fills
- ollama: ~0 (serial)

**Decode preservation (decode_c / decode_1):**
- vLLM: 97.5% (c=4), 93.0% (c=8) — near-perfect
- ollama: 100% / 99% (serial)
- realizr: 55.2% / 50.3% — stabilizes
- llama.cpp: 56.5% / **32.7%** — fastest degradation

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

### vLLM Reference (Marlin W4A16 + PagedAttention, PMAT-119/120/121)

| c | Short | Medium | Long | Short→Long Δ |
|---|-------|--------|------|-------------|
| 1 | 153.6 | — | 149.1 | −2.9% |
| 4 | 594.8 | 551.0 | 558.5 | −6.1% |
| 8 | 1,058.5 | 1,023.2 | 1,040.7 | −1.7% |
| 12 | 1,461.3 | 1,418.5 | 1,464.7 | +0.2% |
| 16 | 1,832.2 | 1,778.5 | 1,717.0 | −6.3% |

**PMAT-121:** vLLM near-invariant at all c (max ±6.3%). Compare realizr: −34% to −49%.

### Cross-Prompt Scorecards (probador llm score)

| Runtime | c=4 Short | c=4 Medium | c=4 Long | Drop |
|---------|-----------|------------|----------|------|
| realizr | 78 B | 67 C+ | **49 D** | −30 points |
| llama.cpp | 70 B | 71 B | 67 C+ | −3 points |

### ⚠️ PMAT-221/222/223: Batched Prefill Quality Bug & Workaround

realizr produces 0 tokens when too many slots prefill simultaneously with non-short prompts (CUDA_MAX_BATCH=32):

| Profile | c_max OK | c_min BROKEN | Per-batch tokens OK | Per-batch tokens BROKEN |
|---------|----------|--------------|---------------------|-------------------------|
| short (23 tok) | 32+ | never | 736 | never |
| medium (102 tok) | 19 | 20 | 1,938 | 2,040 |
| long (311 tok) | 8 | 9 | 2,488 | 2,799 |

**Total-token hypothesis FALSIFIED (PMAT-222):** long 2,488 works but medium 2,040 breaks — bug is prompt-length-dependent, not total-token.

**CUDA_MAX_BATCH=16 workaround (PMAT-223):** Eliminates bug at ALL concurrency levels and prompt lengths:

| c | BATCH=32 (tok/s) | BATCH=16 (tok/s) | BATCH=16 correct? |
|---|-------------------|-------------------|-------------------|
| 1 | 146.4 | 147.2 | ✅ |
| 4 | 216.1 | 316.1 | ✅ |
| 8 | 355.1 | 560.2 | ✅ |
| 16 | 586.5 | 1,008.1 | ✅ |
| 32 | 944.7 ⚠️ | 1,009.9 | ✅ |
| 128 | 1,505.6 | 1,010.3 | ✅ |

⚠️ BATCH=32 c=32 data was bug-affected (avg_tok=67.2). BATCH=16 asymptote is 1,010 tok/s (−33% from 1,500), but output is always correct. **Recommended for production until batch_prefill.rs is fixed.**

### PMAT-227: Long-Prompt Production Refresh (2026-03-17, BATCH=16, correct flags)

3-runtime sweep with long prompt (~311 tok) + uniform:16,256 output + streaming + 60s. BATCH=16 workaround. llama.cpp ctx-size 8192 (required for long prompts at p=16).

| c | realizr | llama.cpp | vLLM | realizr/lcpp | realizr/vLLM |
|---|---------|-----------|------|-------------|-------------|
| 1 | 142.3 | 155.6 | 152.1 | 0.91× | 0.94× |
| 4 | 182.4 | 342.2 | 569.6 | 0.53× | 0.32× |
| 8 | 306.3 | 399.3 | 1,089.4 | 0.77× | 0.28× |
| 16 | 484.2 | 814.0 | 1,788.8 | 0.59× | 0.27× |
| 32 | 692.1 | — | 2,797.6 | — | 0.25× |
| 64 | 705.1 | — | 3,252.1 | — | **0.22×** |

**Prompt-length sensitivity (long vs medium % change):**

| Runtime | c=1 | c=4 | c=8 | c=16 | c=32 | c=64 |
|---------|-----|-----|-----|------|------|------|
| realizr | −3% | −16% | −13% | −15% | — | — |
| llama.cpp | −2% | −3% | −5% | −9% | — | — |
| vLLM | 0% | −3% | −2% | −10% | +1% | +7% |
| ollama | 0% | 0% | — | — | — | — |

**realizr has the largest long-prompt penalty** (FP8 2-step prefill). TTFT gap grows: 8.7× (c=4), 14.4× (c=8), 24× (c=16), 77× (c=32), **144× (c=64)**. vLLM confirmed prompt-invariant (±10% noise, no systematic trend at c=32/64). realizr long asymptote 705 tok/s at BATCH=16 (−52% vs medium BATCH=32). Supersedes PMAT-220/225 data.

**Long-prompt scorecards:**

| c | realizr | llama.cpp | vLLM |
|---|---------|-----------|------|
| 1 | 88 A- | 95 A+ | 98 A+ |
| 4 | 47 D | 75 B | 99 A+ |
| 8 | 59 C | 61 C+ | 98 A+ |
| 16 | 64 C+ | 71 B | 94 A |

realizr drops 5-11 scoring points vs medium (worst: c=4 −11, grade C→D). TTFT penalty dominates at c=4 (score=20 vs 99 vLLM).

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
