# LLM Inference Performance

## Production Methodology — RTX 4060 Laptop (PMAT-177/228, 2026-03-17)

Medium prompt (~102 tok), uniform:16,256 output, streaming, 5s warmup, 60s duration, locked 1900MHz.
Each runtime deployed in isolation via forjar (serial benchmarks).
realizr uses CUDA_MAX_BATCH=32 ITERATION_SCHEDULER=1 (PMAT-258, quality bug eliminated). Prior B16 data preserved below for reference.

### 4-Runtime Aggregate Throughput (tok/s)

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 148.6 | 159.5 | 153.5 | 162.5 |
| 4 | 325.2 | 351.2 | 598.4 | 635.2 |
| 8 | **524.8** | 419.2 | 1,142.4 | — |
| 16 | **931.2** | 912.0 | 2,036.8 | — |
| 32 | 1,600.0 | **1,948.8** | 2,998.4 | — |

Mar 21, 60s duration, medium prompt, same-session serial isolated. realizr: PMAT-291 graph dispatch + PMAT-294 Q8 cache (cumulative +8.5% at c=4). realizr overtakes llama.cpp at c=8 (+25%). llama.cpp reclaims lead at c=32 (fused GEMV scales better at high parallelism).

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

**Per-step c=4 budget (PMAT-267 corrected)**: GPU kernels 7.4ms + launch 0.9ms + H2D 0.4ms + serving 5.5ms = 13.8ms → 290 tok/s.
**PMAT-267 correction of PMAT-266**: GPU kernel time is **7.4ms** (not 10ms). Serving overhead is **5.5ms** (40% of step, not captured in nsys). Graph + event-based sync enables CPU-GPU overlap: serving runs concurrently with GPU.
**Revised projection**: per-M graph + event sync → **390-466 tok/s** at c=4 (0.66-0.79× vLLM) depending on achievable overlap (50-80%). Wall-time gap is **2.0×** (not 4.6×).

### Prompt-Length Sensitivity (PMAT-219/220, updated PMAT-268)

Competitive ratios (realizr/llama.cpp) across prompt lengths with heterogeneous output (B16 B&S):

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

### Iteration Scheduler Prompt-Length Sensitivity (PMAT-268, Mar 18)

B32 iter sched prompt-length penalty vs medium baseline (uniform:16,256 output, streaming, 60s):

| c | Short agg | Medium agg | Long agg | Short boost | Long penalty | PMAT-253 (B16 B&S) |
|---|-----------|------------|----------|------------|-------------|---------------------|
| 4 | 317.0 | 290.1 | 241.0 | **+9.3%** | **−16.9%** | −12.3% |
| 8 | 551.2 | 494.4 | 412.6 | **+11.5%** | **−16.6%** | −14.1% |
| 16 | 990.9 | 880.4 | 691.4 | **+12.6%** | **−21.5%** | −14.1% |
| 32 | 1,705.6 | 1,463.8 | 1,079.5 | **+16.5%** | **−26.3%** | — |

Per-request decode by prompt profile:

| c | Short dec | Medium dec* | Long dec | Short/Med | Long/Med |
|---|-----------|------------|----------|-----------|----------|
| 4 | 82.7 | ~82 | 63.2 | ~1.01× | **0.77×** |
| 8 | 71.4 | ~65 | 53.5 | ~1.10× | **0.82×** |
| 16 | 65.3 | ~57 | 46.0 | ~1.15× | **0.81×** |
| 32 | 57.0 | ~49 | 36.6 | ~1.16× | **0.75×** |

*Medium decode estimated from aggregate/active_slots.

TTFT (P50): short 35-42ms (flat across c), long 67-75ms (flat). Ratio: 1.8×. The iteration scheduler's per-slot prefill makes TTFT **constant with concurrency** (unlike B&S where TTFT grew linearly).

**Key finding: The iteration scheduler INCREASES prompt-length sensitivity** vs B16 B&S. Long penalty grows from −12-14% (B&S) to −17-26% (iter sched). Cause: per-slot prefill concentrates FP8 2-step overhead on each request without batch amortization. Short prompts benefit (+9-17%) because less prefill per slot. Sensitivity grows with concurrency (more active KV to scan per decode step with longer sequences). **vLLM cross-validated same-session (PMAT-269): ±3% at c≤8, +3-12% short boost at c=16-32, long penalty −9% max (c=16) then reverses.** The FP8 2-step pipeline remains realizr's prompt-length Achilles heel.

### vLLM Prompt-Length Cross-Validation (PMAT-269, Mar 18)

Same-session cross-validation of PMAT-268. vLLM isolated on yoga RTX 4060L (realizr stopped):

| c | Short agg | Medium agg* | Long agg | Short boost | Long penalty |
|---|-----------|------------|----------|------------|-------------|
| 4 | 588.8 | 587.4 | 569.7 | +0.2% | −3.0% |
| 8 | 1,133.0 | 1,115.2 | 1,095.6 | +1.6% | −1.8% |
| 16 | 2,053.5 | 1,982.9 | 1,809.3 | +3.6% | **−8.8%** |
| 32 | **3,095.4** | 2,757.6 | 2,839.7 | **+12.3%** | +3.0% |

*Medium from PMAT-258 (same B32 iter sched session, identical GPU clock lock).

Per-request decode:

| c | Short dec | Long dec | Delta |
|---|-----------|----------|-------|
| 4 | 149.9 | 146.0 | −2.6% |
| 8 | 145.5 | 140.5 | −3.4% |
| 16 | 132.1 | 116.2 | −12.0% |
| 32 | 99.9 | 91.4 | −8.5% |
| 64 | 55.6 | 54.7 | −1.6% |
| 128 | 28.7 | 29.0 | +1.0% |

### vLLM Full Prompt-Sensitivity Characterization (PMAT-270, Mar 18)

Extended PMAT-269 to c=64/128. Same-session, vLLM isolated:

| c | Short agg | Medium agg* | Long agg | Short boost | Long penalty |
|---|-----------|------------|----------|------------|-------------|
| 4 | 588.8 | 587.4 | 569.7 | +0.2% | −3.0% |
| 8 | 1,133.0 | 1,115.2 | 1,095.6 | +1.6% | −1.8% |
| 16 | 2,053.5 | 1,982.9 | 1,809.3 | +3.6% | **−8.8%** |
| 32 | 3,095.4 | 2,757.6 | 2,839.7 | **+12.3%** | +3.0% |
| 64 | **3,460.6** | 3,036.1 | **3,407.6** | **+14.0%** | **+12.2%** |
| 128 | **3,605.9** | 3,049.4 | **3,609.8** | **+18.2%** | **+18.4%** |

*Medium from PMAT-258 (same clock lock session).

**Key finding: vLLM prompt-sensitivity is a CONCAVE function of concurrency.** Penalty peaks at c=16 (−8.8%) then reverses. At c≥32, long prompts are **FASTER** than medium (+3% to +18%). At c=128, short and long converge to identical throughput (3,606 vs 3,610 tok/s).

**Root cause analysis:**
1. **c≤8 (noise zone):** Prefill overhead small relative to decode. ±3%.
2. **c=16 (peak penalty):** Enough concurrency to interrupt decode batching, not enough to amortize. −8.8%.
3. **c≥32 (reversal):** Continuous batching + PagedAttention fully amortizes prefill. Long prompts produce larger KV caches that improve attention compute density at high c. +3→18%.

### realizr Full Prompt-Sensitivity Characterization (PMAT-271, Mar 18)

Extended PMAT-268 to c=64/128. Same B32 iter sched session:

| c | Short agg | Medium agg | Long agg | Short boost | Long penalty |
|---|-----------|------------|----------|------------|-------------|
| 4 | 317.0 | 290.1 | 241.0 | +9.3% | −16.9% |
| 8 | 551.2 | 494.4 | 412.6 | +11.5% | −16.6% |
| 16 | 990.9 | 880.4 | 691.4 | +12.6% | −21.5% |
| 32 | 1,705.6 | 1,463.8 | 1,079.5 | +16.5% | −26.3% |
| 64 | **1,747.6** | 1,494.1 | **1,131.0** | **+17.0%** | **−24.3%** |
| 128 | **1,770.9** | 1,514.7 | **1,124.6** | **+16.9%** | **−25.7%** |

Per-request decode at c=64/128: short 57.5/57.9 (constant at M=32 ceiling), long 37.5/37.5 (constant). Long decode penalty: −34.8% (c=64/128) — flat at asymptote.

**Key finding: realizr long penalty PLATEAUS at c≥32 (−24 to −26%).** Does not grow indefinitely — once at BATCH=32 asymptote, both medium and long hit the ceiling and the penalty ratio stabilizes. Short boost also plateaus at +17%.

**Asymptotes by prompt profile:**
| Profile | Asymptote | vs Medium |
|---------|-----------|-----------|
| Short (23 tok) | ~1,771 tok/s | +17% |
| Medium (102 tok) | ~1,515 tok/s | baseline |
| Long (~311 tok) | ~1,125 tok/s | −26% |

**Structural contrast with vLLM:**
- **realizr:** penalty plateaus at −24-26% (fixed-slot KV scan cost proportional to sequence length at every decode step → constant % penalty at asymptote)
- **vLLM:** penalty reverses at c≥32 (+3→18%, concave — continuous batching amortizes prefill, long KV improves attention density)
- realizr has no amortization mechanism without continuous batching + fused Q4K GEMM

### llama.cpp Prompt-Sensitivity Characterization (PMAT-272, Mar 18)

Isolated on yoga, ctx=8192, --parallel 16:

| c | Short agg | Long agg | Short/Long |
|---|-----------|----------|------------|
| 4 | 343.9 | 343.9 | 1.00× |
| 8 | 411.6 | 406.7 | 1.01× |
| 16 | 834.4 | 850.6 | 0.98× |
| 32 | 929.6 | 893.7 | 1.04× |

**Key finding: llama.cpp is GENUINELY prompt-invariant.** Short ≈ long within ±4% at all concurrency levels. The fused Q4K GEMM processes prompts in a single kernel pass with no dequantization overhead — confirming that PMAT-054 (fused Q4K GEMM for realizr) would eliminate realizr's 24-26% long-prompt penalty.

**⚠️ Note:** Medium baselines (354.4/420.1/896.6/943.2) were measured with ctx=4096 (256 tok/slot). These runs used ctx=8192 (512 tok/slot) to accommodate long prompts. The ~3% drop vs medium at c≤8 may be the larger KV allocation overhead, not a prompt effect.

### Complete 3-Runtime Prompt-Sensitivity Summary (PMAT-268→272)

| Runtime | Pattern | Long penalty range | Mechanism |
|---------|---------|-------------------|-----------|
| **realizr** | PLATEAU | −17→−26%, stabilizes at asymptote | Fixed-slot KV scan: penalty proportional to sequence length |
| **vLLM** | CONCAVE | −9% peak (c=16), reverses to +18% (c=128) | CB + PagedAttention amortizes; long KV improves density |
| **llama.cpp** | INVARIANT | ±4% (noise) | Fused Q4K GEMM: single-pass, no dequantization overhead |

realizr is the ONLY runtime with a structural prompt-length penalty. The fused Q4K GEMM (PMAT-054) is the definitive fix — llama.cpp proves the architecture eliminates the penalty entirely.

### c=1 Prompt-Profile Comparison (PMAT-273, Mar 18)

| Runtime | Profile | Aggregate | Decode | TTFT P50 | ITL P50 |
|---------|---------|-----------|--------|----------|---------|
| realizr | short | 148.8 | 150.0 | 13.1 | 6.7 |
| realizr | long | 142.6 | 147.7 | **39.7** | 6.8 |
| vLLM | short | 152.6 | 153.6 | 12.2 | 6.5 |
| vLLM | long | 152.3 | 153.4 | 12.9 | 6.5 |
| llama.cpp | short | 159.0 | 161.4 | 10.0 | 6.2 |
| llama.cpp | long | 156.8 | 157.8 | 10.8 | 6.3 |

**Key finding: FP8 prefill overhead visible even at c=1.** realizr TTFT: 13.1→39.7ms (3.0× long/short ratio). vLLM: 12.2→12.9ms (1.06×). llama.cpp: 10.0→10.8ms (1.08×). Decode and ITL are prompt-invariant at c=1 for all runtimes — the penalty is purely TTFT at low concurrency, growing to aggregate throughput at high concurrency (PMAT-268→271).

### Competitive Ratio × Prompt Profile (PMAT-274, Mar 18)

Competitive ratios shift dramatically with prompt length:

**realizr/vLLM ratio:**

| c | Short | Medium | Long | Short−Long gap |
|---|-------|--------|------|---------------|
| 1 | 0.98× | 0.97× | 0.94× | +0.04× (+4%) |
| 4 | 0.54× | 0.49× | 0.42× | +0.12× (+23%) |
| 8 | 0.49× | 0.44× | 0.38× | +0.11× (+25%) |
| 16 | 0.48× | 0.44× | 0.38× | +0.10× (+23%) |
| 32 | 0.55× | 0.53× | 0.38× | +0.17× (+32%) |
| 64 | 0.50× | 0.49× | 0.33× | +0.17× (+35%) |
| 128 | 0.49× | 0.50× | 0.31× | **+0.18× (+36%)** |

**realizr/llama.cpp ratio:**

| c | Short | Medium | Long | Short−Long gap |
|---|-------|--------|------|---------------|
| 1 | 0.94× | 0.93× | 0.91× | +0.03× (+3%) |
| 4 | 0.92× | 0.82× | 0.70× | +0.22× (+27%) |
| 8 | **1.34×** | **1.18×** | 1.01× | +0.32× (+28%) |
| 16 | **1.19×** | 0.98× | 0.81× | **+0.37× (+38%)** |
| 32 | **1.83×** | **1.55×** | **1.21×** | **+0.63× (+40%)** |

**Key findings:**
1. **realizr/llama.cpp crossover shifts with prompt length:** short → realizr wins at c≥8. Medium → realizr wins at c=8,32 (loses c=16 at 0.98×). Long → realizr barely wins c=8 (1.01×), loses c=16 (0.81×)
2. **realizr/vLLM gap widens 32-36% with long prompts** at c≥32: 0.50× (medium) → 0.31× (long) at c=128
3. **Prompt-profile impact grows with concurrency:** +3% at c=1 → +40% at c=32
4. **PMAT-054 ROI quantified:** fused Q4K GEMM would recover the 0.18× gap at c=128 (from 0.31× to ~0.49×), effectively making realizr prompt-invariant like llama.cpp

Medium c=16 re-verified: 877.6 tok/s (−0.3% vs PMAT-258 880.4). All ratios are stable.

### TTFT Scaling Architecture (PMAT-275, Mar 18)

TTFT P50 (short prompt) across 3 runtimes:

| c | realizr | vLLM | llama.cpp |
|---|---------|------|-----------|
| 1 | 13.1 | **12.2** | **10.0** |
| 4 | 35.0 | **21.4** | 26.6 |
| 8 | 38.4 | **22.3** | 44.1 |
| 16 | 40.2 | **24.7** | 54.1 |
| 32 | 42.2 | **33.2** | 1,546.2 |
| 64 | 2,414.3 | **62.4** | — |
| 128 | 7,101.8 | **110.9** | — |

Three distinct TTFT scaling patterns:
1. **realizr (iter sched):** FLAT at c≤32 (35-42ms), then CLIFF at c>32 (BATCH=32 queue). Per-slot prefill is non-blocking but FP8 overhead makes absolute TTFT higher than competitors
2. **vLLM:** GRADUAL growth (12→111ms at c=128) — continuous batching interleaves prefill with decode. Best absolute TTFT at all c≥4
3. **llama.cpp:** LINEAR growth (10→54ms at c≤16), then CLIFF at c=32 (--parallel 16 queue). Best c=1 TTFT (10ms) from fused Q4K GEMM

**Key insight:** realizr TTFT is flat at c≤32 (Δ<8ms from c=4→32) — the iteration scheduler's per-slot prefill makes TTFT **independent of concurrency** below the batch cap. However, absolute TTFT is 1.5-1.7× higher than vLLM due to FP8 2-step overhead. PMAT-054 (fused Q4K GEMM) would reduce realizr TTFT to ~10-15ms (matching llama.cpp), maintaining the flat scaling while closing the absolute gap.

### Per-Request Decode Gap Decomposition (PMAT-277, Mar 19)

Same-session serial isolated (PMAT-276 data). Per-request decode rate:

| c | realizr | vLLM | llama.cpp | realizr/vLLM |
|---|---------|------|-----------|-------------|
| 1 | 149.2 | 153.6 | 159.1 | 0.97× |
| 4 | 75.7 | 149.6 | 89.1 | 0.51× |
| 8 | 64.4 | 142.8 | 53.3 | 0.45× |
| 16 | 57.2 | 127.4 | 57.8 | 0.45× |
| 32 | 48.8 | 93.6 | 60.0 | 0.52× |
| 64 | 49.4 | 50.3 | — | 0.98× |
| 128 | 49.4 | 25.0 | — | **1.98×** |

**realizr decode is CONSTANT at c≥32** (48.8-49.4 tok/s) — BATCH=32 ceiling. vLLM decode HALVES each doubling above c=32 (93.6→50.3→25.0). **Per-request decode crossover at c≈64** (0.98× ≈ parity), widening to **1.98× at c=128**.

**Gap decomposition update (PMAT-264→277):**

| c | decode_rate | sched_util | product | measured r/v |
|---|------------|-----------|---------|-------------|
| 4 | 0.51× | 0.98× | 0.50× | 0.50× |
| 8 | 0.45× | 0.98× | 0.44× | 0.44× |
| 16 | 0.45× | 0.98× | 0.44× | 0.44× |
| 32 | 0.52× | 0.98× | 0.51× | 0.51× |
| 64 | 0.98× | 0.48× | 0.47× | 0.47× |
| 128 | 1.98× | 0.24× | 0.48× | 0.48× |

**2-factor model validated:** gap = decode_rate × sched_util matches measured r/v within 1%. At c≤32: decode_rate is binding (0.45-0.52×), scheduling near-optimal (0.94-0.98×). At c≥64: decode_rate EXCEEDS 1.0× (realizr wins per-token) but sched_util collapses (0.24-0.48×) from BATCH=32 queue saturation.

**ITL prompt-profile dependency (PMAT-277 addendum):**

| c | Short ITL | Medium ITL | Long ITL | Long/Short |
|---|-----------|-----------|----------|------------|
| 1 | 6.7ms | 6.7ms | 6.8ms | 1.01× |
| 4 | 12.1ms | 13.2ms | 15.8ms | 1.31× |
| 16 | 15.3ms | 17.5ms | 21.8ms | 1.42× |
| 32 | 17.5ms | 20.5ms | 27.3ms | 1.56× |
| 128 | 17.3ms | 20.2ms | 26.7ms | 1.54× |

ITL degrades with prompt length because longer sequences have more KV entries to scan during attention. The long/short ratio grows from 1.01× (c=1, single KV) to 1.56× (c=32, 32 KV slots × longer sequences). Plateaus at c≥32 (BATCH ceiling). **Interactive quality impact:** long-prompt users see 56% slower token delivery at c=32, affecting perceived typing speed.

### CUDA Graph Overhead Isolation (PMAT-279, Mar 19)

Same-session A/B: SKIP_CUDA_GRAPH=1 vs default (graph enabled), B32 iter sched:

| c | Graph | No-graph | Delta |
|---|-------|----------|-------|
| 1 | 146.5 | 130.6 | **+12.2%** |
| 4 | 285.5 | 285.1 | +0.1% |
| 8 | 482.4 | 482.5 | −0.0% |
| 16 | 856.3 | 862.8 | −0.8% |
| 32 | 1,440.9 | 1,450.3 | −0.6% |

**Key finding: The current M=1 CUDA graph is ONLY beneficial at c=1 (+12.2%).** At c≥4, it provides zero benefit (−0.8% at c=16 from graph state overhead). This validates PMAT-267: the value of per-M graph is **100% CPU-GPU pipelining enablement**, not launch overhead savings. At production concurrency (c≥4), kernel launches are already amortized across M tokens within the graph — the 771-kernel graph replays fast enough. The bottleneck is the synchronous `cuStreamSynchronize` blocking (5.5ms serving overhead) that prevents CPU-GPU overlap.

**c=1 per-token time decomposition** (cross-validated with `apr profile`):
- GPU forward pass: 6.29ms (82% of step — from `apr profile` 159.1 tok/s)
- Serving overhead: 0.54ms (7% — HTTP, tokenizer, scheduling)
- Graph launch savings: 0.83ms (11% — 771 kernels → 1 graph replay)
- Graph-enabled total: 6.83ms (146.5 tok/s)
- No-graph total: 7.66ms (130.6 tok/s)

**Serving overhead scaling:** 0.54ms (c=1) → 6.3ms (c=4) → ~8.4ms (c=16). At c=4, serving is **46% of step time** (6.3/13.7ms). This is the target for event-based pipelining — overlap serving with GPU execution.

**Pipelining projection (PMAT-280, Mar 19):**

Event-based sync: step time = max(GPU, serving) + ~0.3ms sync overhead (replaces sequential GPU+serving):

| c | Current | GPU | Serving | Pipelined step | Projected | vs vLLM |
|---|---------|-----|---------|---------------|-----------|---------|
| 1 | 146.5 | 6.3ms | 0.5ms | 6.6ms | 151.7 | **1.00×** |
| 4 | 285.5 | 7.4ms | 6.3ms | 7.7ms | **519.5** | **0.89×** |
| 8 | 482.4 | ~7.4ms | ~9.4ms | ~9.7ms | 824.7 | 0.74× |
| 16 | 856.3 | ~10ms | ~8.4ms | ~10.3ms | **1,553** | **0.78×** |
| 32 | 1,440.9 | ~10ms | ~12.2ms | ~12.5ms | **2,560** | **0.88×** |

*c=4 GPU/serving from nsys (PMAT-267). c=8-32 estimated from aggregate throughput minus nsys GPU.*

**Key insight:** At c=4, pipelining alone (no graph changes, no kernel fusion) would reach **0.89× vLLM** — nearly closing the gap. The serving overhead (6.3ms) almost fully overlaps the GPU time (7.4ms). This is achievable by replacing `cuStreamSynchronize` with `cuEventRecord`/`cuEventQuery` in the decode loop.

**PMAT-283 timing decomposition (measured Mar 19, PhaseTimer on yoga):**

| Phase | c=4 | c=16 | % of step |
|-------|-----|------|-----------|
| lock | 0µs | 0µs | 0.0% |
| sched | 0µs | 0µs | 0.0% |
| **decode** | **13,000µs** | **15,000µs** | **99.99%** |
| dist | 1µs | 2µs | 0.0% |

**⚠️ PMAT-280 FALSIFIED:** The "6.3ms serving overhead" is NOT serving overhead — it is GPU sync time INSIDE `batched_decode_step()`. Lock contention = 0µs. Token distribution = 1µs. Scheduling = 0µs. **99.99% of step time is GPU compute + sync.** Pipelining serving with GPU will NOT yield 0.89× vLLM because there is no serving to pipeline.

**Five-Whys: Why is per-M graph the binding change? (PMAT-283→285)**

1. **Why** is realizr 0.44× vLLM at c=4? → Step time 13ms vs vLLM ~6.8ms
2. **Why** 2× longer? → realizr dispatches 771 kernels per M=1 × M replays. vLLM dispatches 1 CUTLASS GEMM for M tokens
3. **Why** no per-M dispatch? → `decode_graph` is `Option<CudaGraphExec>` (single M=1 graph), not `HashMap<usize, CudaGraphExec>`
4. **Why** was batched graph disabled? → H-CB11 FALSIFIED: graph replay 3ms slower — attention grid dims frozen at capture, seq_len grows each step
5. **Why** can't grid dims be dynamic? → CUDA graphs freeze `gridDim` at capture. Fix: position-independent kernels with max-grid capture + seq_len-based early-exit (vLLM's approach)

**Existing infrastructure in realizr:**
- `batched_decode_graphs: HashMap<usize, CudaGraphExec>` — exists but graphs are stale (H-CB11)
- `forward_batched_graphed_replay()` — exists in `par-121.rs`
- `try_batched_graph_capture()` — exists with pre-upload support
- `BATCHED_GRAPH=1` env var — enables graphed path (disabled by default, 25% slower)

**PMAT-285 re-verification (Mar 19, v0.8.3):** `BATCHED_GRAPH=1` confirmed still −32% at c=4 (194.5 vs 285.9) and −46% at c=8 (267.9 vs 494.6). H-CB11 is NOT fixed in current binary. Root cause confirmed: graph captured with dummy `seq_lens=1`, replays with `seq_lens=128+`. The batched attention grid `(num_heads, M, 1)` is M-dependent but NOT seq_len-dependent — however the kernel READS seq_lens from a buffer and the capture may corrupt internal state. Full nsys profiling of graph overhead needed to identify exact graph nodes causing regression.

**PMAT-285 fix attempt (realistic seq_lens): NO IMPROVEMENT.** Passed real positions → 194.3 tok/s (same as 194.5 with dummies). **PMAT-286 fused KV scatter: 4 attempts, root cause identified, NET NEGATIVE.**
- Attempts 1-3: CUDA_ERROR_ILLEGAL_ADDRESS — root cause was **em dash (U+2014) in PTX comment** (non-ASCII), NOT selp.b64 or parameter alignment. PTX requires pure ASCII.
- Attempt 4 (selp.b64, ASCII-clean): **Works correctly (6/6 correctness)** but −12% regression at c=4-16 (258/427/758 vs baseline 291/495/869). Near-neutral at c=32 (1458 vs 1469).
- **Root cause of regression:** 7 params (4×u64 + 1×u64 + 2×u32) vs 5 params — extra `ld.param.u64` instructions for loading unused K/V pointers. The 28 saved launches (0.49ms) are offset by ~1.5ms of extra param loading across all blocks/layers.
- **Conclusion:** Fusing kernels that ADD parameters is not profitable. Fusion must REDUCE total instruction count, not just launch count.

**PMAT-287: Fused Q+K DP4A — FALSIFIED (net negative).**

Existing fusions already active:
1. `fused_gate_up_swiglu` (HW DP4A): gate+up+SwiGLU → 1 kernel/layer
2. `batched_qkv_dp4a` (M=2-8, all Q4K only): shared Q8 input for 3 GEMV

Attempted: relax Q6K V condition + extend M<=32. Results:
- Q6K V relaxation at M<=8: −12% regression (257 vs 291 at c=4). DP4A GEMV for Q+K is SLOWER than `batched_gemv_or_gemm` which selects FP8 cuBLASLt at M≥4
- M>8 extension: CUDA_ERROR_ILLEGAL_ADDRESS (code 700). DP4A Q8 launch crashes at M>8
- **Root cause:** `batched_gemv_or_gemm` auto-selects FP8 cuBLASLt at M≥4 which is faster than DP4A GEMV for the Q4K projections. The fused QKV DP4A was designed for M≤8 when DP4A was faster
- **Conclusion:** The existing `batched_gemv_or_gemm` dispatch is already optimal. Fused QKV DP4A is never active for Qwen2.5-Coder (V is Q6K) and extending it is net negative

**Remaining path: TransformerBlockMegakernel (PAR-039)** — exists in trueno as a stub (Phase 1 RMSNorm complete, Phase 3 QKV/attention/FFN is TODO). Also PersistentDecoderKernel (PAR-036). Both need full Phase 3 implementation: Q4K GEMV for all projections + attention + SwiGLU in one launch. 514→28 launches. Multi-week PTX kernel development in trueno-gpu.

**All incremental optimizations have been tested and FALSIFIED:**
- Fused KV scatter (PMAT-286): −12% from extra params
- Fused Q+K DP4A (PMAT-287): −12% (FP8 cuBLASLt faster than DP4A at M≥4)
- Batched graph (PMAT-285): −32% (654 node overhead)
- Event sync pipelining (PMAT-283): 0% (no serving overhead)
- DP4A M>8: CUDA crash (illegal address)
- Megakernel PAR-039: 1-block = 1/24 SM utilization. Wrong approach

**PMAT-288: Fused non-GEMM layer kernel — FALSIFIED by PMAT-092.**
- PMAT-092 (already in codebase) tried residual+rmsnorm fusion: **-5% regression**
- Root cause: RMSNorm forces grid (1,M) for cooperative reduction → 17% SM occupancy
- Residual uses (6×M) grid → full occupancy. Fusion loses 6x parallelism
- Pattern applies to ALL non-GEMM fusions involving RMSNorm
- Megakernel ABANDONED (1 SM). Non-GEMM fusion ABANDONED (occupancy loss)
- **All optimization paths from this benchmarking repo are exhausted**
- Per-launch overhead: ~11.9µs average (from M=1 vs M=4 step time delta / launch count delta)
- cuBLASLt launches are ~3µs (fast), raw PTX launches are ~25µs (slow)
- Non-GEMM (224 × 25µs = 5.6ms) dominates over GEMM (196 × 3µs = 0.6ms)
- But non-GEMM fusion loses SM occupancy (PMAT-092: -5% from grid restriction)
- **Prefill chunking (PMAT-289): LOW ROI for production workloads.** Medium prompts (102 tokens) fit in one chunk (256). Zero benefit on standard benchmarks

**PMAT-290: −12% regression bisected to trueno PTX disk caching (commit 4c8022e).**
- Bisection: 9341847 (284.7, good) → 4c8022e (252.0, bad). PMAT-276 baseline: 291.2
- Root cause: trueno `perf: shared memory tiled NF4 GEMM + PTX disk caching` introduced CUDA linker API (`cuLinkCreate/AddData/Complete`) for cubin extraction
- NF4 tiling was reverted (9fdf2a4) but PTX disk caching code REMAINED
- The linker API produces different cubins than `cuModuleLoadData` — ~12% slower kernels
- Clearing `~/.cache/trueno/ptx/` does NOT fix (cache is rebuilt, still uses linker API)
- **FIXED (trueno 4bb8a1e):** Pass `CU_JIT_TARGET` to `cuLinkCreate` for sm < 120. Without it, the linker skips target-specific optimizations. Blackwell (sm_121+) still uses auto-detect. Results: 285.1/482.8/867.0/1440.3 at c=4/8/16/32 — within ±2.5% of PMAT-276 baseline. Regression eliminated.

**PMAT-291: Cross-project analysis (vLLM/llama.cpp/PyTorch, Mar 21).**
- llama.cpp achieves 8-15 launches/step vs realizr's 430 via fused `mul_mat_vec_q` + ncols_dst templating
- vLLM uses ~80 nodes with CUTLASS GEMM, captured per batch size, replayed via CUDA graphs
- PyTorch/inductor: two-stage (IR fusion → CUDA graph replay)
- `cuGraphExecUpdate` API added to trueno (c047e5c) but **will NOT fix -32% batched graph** — the bottleneck is 654-node REPLAY overhead, not instantiation. llama.cpp graphs work because they have 8-15 nodes. The fix is reducing kernel count, not improving the graph API.
- **Concrete reference implementation found:** `ggml-cuda/mmvq.cu` `mul_mat_vec_q` with `ncols_dst` compile-time switch.
- **Deeper analysis (Mar 21):** llama.cpp's 8-15 nodes comes from ggml's TENSOR-LEVEL graph (1 node per tensor op). realizr dispatches at KERNEL level (multiple kernels per tensor op: Q8 quantize + DP4A GEMV, or FP8 dequant + cuBLASLt). The existing `BatchedHwDp4aQ4KGemvKernel` is ALREADY a fused 1-step kernel (like llama.cpp's mul_mat_vec_q). The extra launches are from the non-GEMM pipeline (rmsnorm, rope, scatter, attention, residual) which can't be fused (PMAT-092).
- **Paradox:** FP8 cuBLASLt is 2 launches per projection but ~3us each (6us total). DP4A is 1 launch at ~25us. FP8 has LESS dispatch overhead despite more launches.
- **Conclusion: realizr already uses fused GEMV (DP4A path) and fast GEMM (cuBLASLt path). The 430 launches include ~224 non-GEMM ops that can't be fused. Reducing to llama.cpp's 8-15 requires a ggml-style tensor graph architecture -- a complete rewrite, not an incremental change.**

### PMAT-054 Implementation Brief: Fused Q4K GEMM (Binding Fix)

**All alternative optimizations have been falsified.** The investigation chain (PMAT-279→280→283→285) proves:

| Approach | Status | Evidence |
|----------|--------|----------|
| Event sync pipelining | **FALSIFIED** | PMAT-283: 99.99% of step = decode, no serving to overlap |
| Per-M CUDA graph | **FALSIFIED** | PMAT-285: −32% regression, 654 node overhead |
| M=1 graph at c≥4 | **FALSIFIED** | PMAT-279: 0% benefit at c≥4 |
| Realistic seq_lens capture | **FALSIFIED** | PMAT-285: no improvement (194.3 vs 194.5) |

**Kernel fusion (PMAT-054) is the ONLY remaining path to close the gap.** Evidence:
- vLLM uses 1 CUTLASS GEMM kernel for 95.7% of GPU time (PMAT-214)
- llama.cpp uses fused Q4K GEMM — prompt-invariant ±4% (PMAT-272)
- realizr dispatches ~16 kernels/layer × 28 layers = 448+ kernels/step
- PMAT-054 targets: fused Q4K dequant→GEMM (reads Q4K weights once, dequants in registers)

**PMAT-286: Decode sub-phase timing (Mar 20, PhaseTimer inside batched_decode_step).**

| Phase | M=4 steady | % |
|-------|-----------|---|
| embed | 0-2µs | 0.0% |
| prep | 0-5µs | 0.0% |
| **fwd+sync+argmax** | **11,400-13,900µs** | **100%** |

**Definitive root cause: CPU kernel dispatch time.**
- ~402 kernel launches × ~17.5µs/launch = **~7.0ms CPU dispatch time** (corrected: fused_gate_up_swiglu already active, 14 kernels/layer not 23)
- GPU execution: 7.4ms (from nsys) — GPU finishes BEFORE CPU is done dispatching
- cuStreamSync: ~0ms (GPU already idle by the time last kernel is dispatched + executed)
- Total: ~11.4ms dispatch + ~1.5ms (argmax + D2H + misc) ≈ 13ms ✓

**The GPU is NOT the bottleneck — the CPU is.** The CPU spends 11.4ms dispatching 654 `cuLaunchKernel` calls. The GPU finishes each kernel before the next arrives. vLLM dispatches ~3 CUTLASS GEMM calls total (95.7% of GPU time in 1 kernel) — ~0.05ms CPU dispatch.

**Fix: kernel fusion reduces CPU dispatch time.**
- Current: ~402 launches → ~7.0ms CPU dispatch (fused_gate_up_swiglu already active)
- Next fusions (QKV + KV scatter + rmsnorm+gemv + residual+rmsnorm): 402→262 → ~4.6ms dispatch
- Step time: 13ms → ~12ms (GPU 7.4ms + dispatch 4.6ms, overlapped) → ~10.6ms → **378 tok/s at c=4**
- With CUDA graph on 262-node fused kernel set: step ≈ 7.5ms → **533 tok/s (0.91× vLLM)**

**Existing fusions already active (GpuProfile `fused_gate_up=true`):**
- `FusedGateUpSwigluHwDp4aQ4KGemvKernel` (PMAT-034): gate + up + SwiGLU → 1 kernel/layer (saves 2)

**Next fusion targets (by ROI):**

| Fusion | Saves | New total | CPU dispatch |
|--------|-------|-----------|-------------|
| Fused QKV (q+k+v → 1) | 56 | 346 | 6.1ms |
| + Fused KV scatter | 28 | 318 | 5.6ms |
| + Fused rmsnorm+gemv | 28 | 290 | 5.1ms |
| + Fused residual+rmsnorm | 28 | 262 | 4.6ms |

**Expected ROI (revised):**
- **Pipelining inside decode**: 13ms → ~7.7ms at c=4 (overlap 5.6ms sync with next H2D). 285→520 tok/s (0.89× vLLM) — PMAT-280 projection REVIVED
- Prompt-length invariance: eliminates −24-26% long penalty (requires PMAT-054 fused prefill)
- Competitive ratio: recovers 0.18× gap at c=128 (PMAT-274: 0.31→~0.49×)
- TTFT: 3× improvement (39.7→~13ms long prompt, requires PMAT-054 fused prefill)

### vLLM Graph Benefit (PMAT-282, Mar 19)

vLLM with `--enforce-eager` (no CUDA graphs) vs default:

| c | vLLM graph | vLLM eager | Graph benefit | realizr | r/v-eager |
|---|-----------|-----------|--------------|---------|-----------|
| 1 | 152.3 | 123.9 | **+22.9%** | 147.2 | **1.19×** |
| 4 | 586.8 | 464.4 | **+26.4%** | 291.2 | 0.63× |
| 8 | 1,114.4 | 890.8 | **+25.1%** | 494.6 | 0.56× |
| 16 | 1,983.2 | 1,676.5 | **+18.3%** | 868.8 | 0.52× |
| 32 | 2,898.5 | 2,747.7 | +5.5% | 1,469.4 | 0.53× |

**Key findings:**
1. **vLLM graphs provide +18-27% at c≤16.** Multi-M graph capture (pre-captured at M=1,2,4,8,16,32) is effective because each step dispatches the graph matching the current batch size.
2. **realizr BEATS vLLM-eager at c=1** (1.19×). Without graphs, realizr's kernel pipeline is faster. The c≥4 gap (0.52-0.63×) is purely scheduling + batching architecture.
3. **Graph accounts for ~25% of the realizr/vLLM gap** at c=4-16. realizr/vLLM improves from 0.44× to 0.56× when vLLM graphs are disabled.
4. **vLLM c=32 graph benefit drops to +5.5%** — at saturation, batch size is stable (always ~32) so graph recapture is rare.

**Contrast with PMAT-279:** realizr's M=1 graph provides 0% at c≥4 because it replays the same single-token graph M times. vLLM's multi-M graphs process M tokens in one launch. **Per-M graph capture is the key differentiator** — not just "having CUDA graphs".

### Stability and Correctness (PMAT-281, Mar 19)

**5-minute sustained load (c=16):** 873.6 tok/s (baseline 868.8 = +0.6%), 2,012 requests, **0 errors**. ITL P50 17.5ms (identical to 60s). GPU memory 7,456 MB (stable). 6/6 correctness tests pass before and after.

**10-minute sustained load (c=32):** 1,531.1 tok/s (baseline 1,469.4 = +4.2%), 6,843 requests, **0 errors**. ITL P50 20.4ms, ITL drift 3.16 ms/min (31.6ms over 10 min — within tolerance). GPU memory 7,478 MB (no leak). 6/6 correctness tests pass after.

**Conclusion:** Iteration scheduler + BATCH=32 is production-stable over sustained load. No memory leaks, no error accumulation, no throughput degradation, no state corruption. Ready for production deployment.

### Event Sync Implementation Target (PMAT-283, Mar 19)

Source analysis of realizr decode path identifies the exact sync bottleneck:

**Critical sync point:** `src/cuda/executor/layers/reduces.rs:92`
```
self.stream.synchronize()  // BLOCKS CPU until GPU graph completes
```

This is called after every M=1 graph replay (line 86: `self.stream.launch_graph(graph_exec)`). The CPU thread is idle for ~6.3ms (at c=4) waiting for GPU decode to finish. During this time, the serving layer could be:
- Distributing previous step's tokens via SSE
- Accepting new HTTP connections
- Running the iteration scheduler logic

**Implementation plan (4 files, ~100 LOC):**
1. `src/cuda/executor/streams.rs` — Add `record_event()` and `query_event()` wrappers
2. `src/cuda/executor/layers/reduces.rs:92` — Replace `synchronize()` with `record_event()`
3. `src/api/iteration_scheduler.rs:406` — Before `batched_decode_step()`, query previous event
4. `src/api/iteration_scheduler.rs:422-423` — Move token distribution BEFORE next decode launch

**Projected impact (PMAT-280):** At c=4, step time drops from 13.7ms (7.4ms GPU + 6.3ms serving, sequential) to ~7.7ms (max(7.4, 6.3) + 0.3ms sync) → **520 tok/s (0.89× vLLM)**

### Tensor Graph Dispatch (PMAT-291, Mar 21)

Pure Rust tensor compute graph replacing per-kernel dispatch. 14-node graph per transformer layer (1 leaf + 13 ops: 2 RMSNorm, 7 MulMat, 1 attention compound, 2 residual add, 1 SwiGLU). Graph executor walks nodes in topological order, dispatching via KernelDispatch trait.

**Key difference:** Graph path routes all projections through `batched_gemv_or_gemm` auto-selection (FP8 cuBLASLt at M>=5, DP4A at M<5). Baseline forces fused DP4A QKV for M=2-8, which is suboptimal when the iteration scheduler produces M>=5.

| c | Baseline (decode tok/s) | Graph (decode tok/s) | Delta | Aggregate |
|---|------------------------|---------------------|-------|-----------|
| 1 | 148.6 | 148.6 | 0.0% | 148.6 |
| 4 | 73.9 | 78.9 | **+6.8%** | 315.6 |
| 8 | 64.7 | 65.8 | +1.7% | 526.4 |
| 16 | 58.3 | 59.4 | +1.9% | 950.4 |
| 32 | 48.9 | 50.2 | +2.7% | 1,606.4 |

**Falsification:** BATCHED_DP4A=0 (disable fused DP4A, same individual projections) gives 68.3 tok/s at c=4 — WORSE than both baseline (73.9) and graph (79.7). The graph benefit is not just from bypassing fused DP4A; the simplified dispatch loop and graph-directed execution also reduce CPU overhead.

Now ON by default (opt-out: `GRAPH_DISPATCH=0`). Implementation: trueno `ComputeGraph` + realizr `KernelDispatch` impl + `graph_builder` + `graph_decode`.

### CUDA Graph Capture on Tensor Graph (PMAT-292, Mar 21)

Wired tensor graph dispatch into CUDA graph capture path (`BATCHED_GRAPH=1`). Reduces captured graph nodes from 654 to 392 (14×28 layers).

| Config | c=4 decode tok/s | vs baseline |
|--------|-----------------|-------------|
| Tensor graph only (default) | 78.9 | +6.8% |
| Tensor graph + CUDA graph | 61.1 | **-22.6%** |
| Old CUDA graph (654 nodes) | ~50 | -32% (PMAT-285) |

**FALSIFIED:** CUDA graph overhead scales with node count. Improved from -32% (654 nodes) to -22.6% (392 nodes) but still net negative. Linear extrapolation: need <~150 nodes for breakeven, requiring kernel fusion to reduce from 14 to ~5 ops/layer.

### Fused FP32 Q4K GEMV Kernel (PMAT-293, Mar 21)

New PTX kernel in trueno: `FusedFp32Q4KGemvKernel`. Reads FP32 activations directly (no Q8_1 pre-quantization), dequants Q4K weights to FP32 in-thread, accumulates via FP32 FMA. Same half-warp structure as `BatchedHwDp4aQ4KGemvKernel`.

**Design rationale:** Each DP4A projection currently requires 2 kernel launches (Q8 quantize + GEMV). The fused kernel eliminates the Q8 launch, reducing from 2 to 1 launch per projection. At 7 projections x 28 layers = 196 saved launches per decode step (~2.4ms at 12us/launch).

**Trade-off:** FP32 MAD (128 ops/cycle) vs DP4A (256 ops/cycle). But GEMV is bandwidth-bound (weight reads from DRAM dominate), so compute throughput is irrelevant. The activation reads hit L2 cache regardless.

**Measurement (FUSED_FP32_GEMV=1):**

| c | Graph dispatch (baseline) | Fused FP32 GEMV | Delta |
|---|--------------------------|-----------------|-------|
| 1 | 148.6 | 148.6 | 0.0% |
| 4 | 78.9 | 26.4 | **-66.5%** |
| 8 | 65.8 | 14.3 | **-78.3%** |

**FALSIFIED.** FP32 dequant+multiply is 2x slower compute than DP4A (128 vs 256 ops/cycle on sm_89). At M=1, kernel is bandwidth-bound so parity. At M>1, compute becomes binding and DP4A's 2x throughput advantage dominates. The Q8 quantize launch overhead (~12us) is trivial vs the compute penalty.

**Lesson:** Kernel fusion that changes the compute datatype (FP32 vs INT8 DP4A) is only viable when bandwidth-bound (M=1). For batched decode (M>1), DP4A is essential.

### Inline Q8 DP4A GEMV (PMAT-295, Mar 21)

Correct fusion approach: keep DP4A compute, inline Q8 quantize using per-thread absmax (no shuffle). Each thread loads 4 FP32 values, finds local absmax, quantizes to INT8 in registers, packs into u32 for DP4A.

| c | Baseline (graph dispatch) | Inline Q8 | Delta |
|---|--------------------------|-----------|-------|
| 1 | 148.6 | 148.6 | 0.0% |
| 4 | 80.2 | 25.0 | **-69%** |
| 8 | 65.8 | 12.4 | **-81%** |
| 16 | 59.4 | 59.4 | 0.0% (FP8 bypasses) |
| 32 | 50.2 | 50.2 | 0.0% (FP8 bypasses) |

**FALSIFIED.** In-register Q8 quantize adds ~60 extra instructions per super-block (absmax + quantize + pack vs 2 loads from pre-computed Q8 buffer). This causes register pressure and scattered FP32 cache misses. The separate Q8 quantize kernel has full SM utilization and writes a contiguous L2-cached Q8 buffer.

**Definitive conclusion: the 2-kernel pattern (Q8 quantize + DP4A GEMV) is optimal at M>1.** The separate Q8 kernel exploits full GPU parallelism for a trivially small operation, while the GEMV kernel reads from a perfectly coalesced Q8 buffer. Neither FP32 fusion (PMAT-293) nor inline Q8 DP4A (PMAT-295) can beat this. The 430-launch bottleneck is an architectural ceiling that cannot be addressed by GEMV kernel fusion.

### CPU Format Parity (PMAT-297, Mar 21)

Fresh same-machine CPU benchmark (Intel Xeon W-3245 @ 3.2GHz, c=1, 60s):

| Runtime | Decode tok/s | ITL P50 (ms) | Prefill tok/s |
|---------|-------------|-------------|--------------|
| llama.cpp | 59.0 | 16.9 | 4,269 |
| realizr (GGUF) | 17.1 | 58.4 | 16.8 |

**Gap: 2.40x** with `RAYON_NUM_THREADS=16` (was 3.45x at default 32 threads). HyperThreading contention caused +44% regression. The remaining 2.4x gap is SIMD kernel quality (trueno AVX-512 VNNI 4-row kernel vs ggml LLAMAFILE-optimized AVX-512).

| Config | Decode tok/s | Gap to llama.cpp |
|--------|-------------|------------------|
| realizr (default 32 threads) | 17.1 | 3.45x |
| realizr (PMAT-297: 16 physical cores) | 25.4 | 2.32x |
| realizr (PMAT-298: + adaptive parallelism) | 26.4 | 2.23x |
| realizr (PMAT-299: + deep prefetch) | 29.9 | 1.97x |
| realizr (PMAT-300: + stack Q8K + tile sweep) | 30.0 | 1.97x |
| realizr (PMAT-301: + ggml-style kernel) | 29.6 | 1.99x (instruction-neutral) |
| realizr (PMAT-302: + HUGEPAGE + mlock) | 30.7 | 1.92x |
| realizr (PMAT-306: + lean pointer dispatch) | **31.9** | **1.85x** |
| realizr (PMAT-306: + raw inner dot) | 30.2 | 1.95x (function call overhead) |
| realizr (PMAT-306: + inline(always)) | 28.9 | 2.04x (I-cache bloat) |
| realizr (PMAT-307: + QKV workspace) | 32.1 | 1.84x |
| realizr (PMAT-308: + raw inner dot) | **32.6** | **1.81x** |
| realizr (PMAT-309: shared Q8K gate+up) | 31.5 | 1.87x (Vec alloc overhead) |
| realizr (PMAT-310: GEMV barrier pool) | deadlock | FALSIFIED (concurrent requests) |
| llama.cpp (16 threads, LLAMAFILE) | 59.0 | -- |

**PMAT-300 tile sweep:** 64 optimal. 128 = -10%, 256 = -37% (load imbalance). Stack-allocated Q8K buffers: negligible improvement (malloc not bottleneck).

**Cumulative CPU improvement: +91%** (17.1 → 32.6). Gap vs llama.cpp: **1.81x**.

**PMAT-298:** AVX-512 VNNI (6/6 correct) FALSIFIED on perf (-16%, Cascade Lake freq penalty).

**PMAT-301: ggml-style scale-shuffle-accumulate kernel.** Transpiled ggml's architectural pattern via decy analysis: scale applied as i16 in integer path (madd_epi16), single hsum at end (was 8 per SB), pre-computed bsums. **6/6 correct, 29.6 tok/s — same as PMAT-299 (30.0).**

**PMAT-304: perf stat root cause analysis.** System-wide profiling during inference:

| Metric | realizr | llama.cpp | Ratio |
|--------|---------|-----------|-------|
| IPC | **1.60** | 1.01 | 0.63x |
| Cache misses/s | 572M | **1,003M** | 1.75x |
| DRAM traffic | 36.6 GB/s | **64.2 GB/s** | 1.75x |
| Decode tok/s | 30.8 | 63.9 | 2.07x |

**Root cause: realizr spends too many cycles on NON-memory work** (IPC 1.60 = doing computation). llama.cpp has IPC 1.01 = nearly ALL cycles are memory stalls = maximum bandwidth utilization. realizr's Q8K quantize + rayon dispatch + scale extraction burn cycles that could be spent issuing memory loads.

Added MAP_POPULATE + MADV_RANDOM (matching llama.cpp's llama-mmap.cpp).

**PMAT-305: Direct FP32 (skip Q8K) — FALSIFIED (-17%).** Q8K maddubs essential.

### GPU Format Parity — 3B Model (PMAT-314, Mar 22)

Qwen2.5-Coder-3B-Distill-Qwen3-Coder-Next, RTX 4060L, BATCH=8, c=1, 60s, Q4_K_M:

| Format | Decode tok/s | Aggregate | TTFT ms | ITL ms | Correct |
|--------|-------------|-----------|---------|--------|---------|
| GGUF Q4_K_M | 80.9 | 79.9 | 31.6 | 12.4 | 5/6 |
| SafeTensors→Q4K | **91.6** | **88.5** | 62.5 | **10.9** | 5/6 |
| APR Q4K | correct | — | — | — | 5/6 |

**SafeTensors +13% decode vs GGUF** at same Q4K quantization. TTFT 2x slower (BF16→Q4K
streaming conversion overhead). All three formats produce correct output (5/6 each).

**PMAT-314 fix:** `resolve_model_path` picked `model-00001-of-00002.safetensors` (344/434
tensors) instead of `model.safetensors.index.json`. Layer 28 split across shards. Fix: check
index.json BEFORE individual shard files.

**PMAT-315 fix:** ALB-095 forward path missing QKV bias addition — produced "HHHH" on all
Qwen2 APR models. Fix: extract q/k/v biases per layer, add after GEMV. APR probador benchmark
blocked by SSE streaming format (curl works, separate issue).

### 3B Model Concurrency (PMAT-316, Mar 22)

3B GGUF, BATCH=8, RTX 4060L 8GB:

| c | Aggregate tok/s | Decode tok/s | TTFT P50 ms |
|---|----------------|-------------|-------------|
| 1 | 79.9 | 80.9 | 31.6 |
| 4 | 80.1 | 80.9 | 5,704 |
| 8 | 79.7 | 80.6 | 13,024 |

**Effectively serial** — aggregate flat at ~80 tok/s. 8GB VRAM too tight for concurrent
KV cache with 3B model. The 1.5B model achieves 10x scaling at c=32 (BATCH=32 fits).
**Conclusion:** 3B on 8GB is single-user only. Use 1.5B for concurrent workloads.

### Official Qwen2.5-Coder-3B-Instruct (PMAT-319, Mar 23)

3-runtime comparison on RTX 4060L 8GB, BATCH=8:

| Runtime | Decode c=1 | Aggregate c=4 | TTFT c=1 | Correct |
|---------|-----------|---------------|---------|---------|
| realizr | 81.9 | 80.3 (serial) | 27ms | **6/6** |
| llama.cpp | **90.7** | **195.7** | **15ms** | — |
| ollama | 92.1 | — | 74ms | — |

**Official 3B: 6/6 correctness** (distill was 5/6). llama.cpp 10.7% faster at c=1 AND
scales at c=4 (not VRAM-constrained like realizr). The serialization is realizr's iteration
scheduler under VRAM pressure, not a model limitation.

**Recommendation:** Deploy official 3B for single-user quality; 1.5B for concurrent serving.

### Intel CPU Benchmarks (PMAT-320, Mar 23)

Intel Xeon W-3245 (16 cores @ 3.2GHz), 192GB RAM, 2x Radeon Pro W5700X (unused — WGPU not wired):

| Runtime | Model | Decode c=1 | TTFT P50 | ITL P50 | Correct |
|---------|-------|-----------|---------|---------|---------|
| realizr | 1.5B | 34.4 | 792ms | 29.1ms | 5/6 |
| realizr | 3B | 19.2 | 1433ms | 52.0ms | **6/6** |
| llama.cpp | 3B | **35.6** | **34ms** | **28.1ms** | — |

- realizr 3B: 19.2 tok/s (55.8% of 1.5B) — expected for 2x params, 2.3x layers
- llama.cpp 3B: 35.6 tok/s — **1.85x faster** (same gap as 1.5B at 1.71x)
- **WGPU**: trueno has WGSL compute shaders (matmul, dot, softmax) but not integrated
  into realizr's inference pipeline. W5700X GPUs idle. Integration would require wiring
  trueno's `GpuBackend` into the Q4K GEMV forward pass.

**Recommendation:** Deploy official 3B for single-user quality; 1.5B for concurrent serving.

### PMAT-317: Fused Q4K Prefill FALSIFIED (Mar 23)

**FALSIFIED:** `FUSED_Q4K_PREFILL=1` is **-52% slower** on prefill, -1.9% decode (noise).

| Config | Decode tok/s | Prefill tok/s | TTFT P99.9 |
|--------|-------------|--------------|------------|
| Baseline (HGEMM+FP8) | 150.9 | 6.4 | 17.1s |
| FUSED_Q4K_PREFILL=1 | 148.1 | 3.1 | 34.2s |

In-kernel Q4K dequant cannot compete with cuBLAS HGEMM+FP8 tensor core GEMM at M>1 prefill.
The fused path is optimal for M=1 GEMV (decode) but counterproductive for M>1 GEMM (prefill).
**PMAT-268 "required at c>=16" FALSIFIED** — the long-prompt penalty is from iteration
scheduler contention, not from HGEMM overhead. Reverted config.

**Nightly automation kept:** Extended `scripts/nightly.sh` with `yoga` mode:
- Isolated serial: deploy one runtime → benchmark c=1,4,8,16,32 → teardown → next
- Scoring gate: `probador llm score --fail-on-grade C`
- Run: `./scripts/nightly.sh yoga`

### Recommended Next Moves (PMAT-317, Mar 23)

| Priority | Approach | Projected ROI | Effort |
|----------|----------|--------------|--------|
| ~~P0~~ | ~~PGO~~ | ~~+5-15%~~ | FALSIFIED (0%). No branch misprediction |
| ~~P0~~ | ~~Inline F16C conversion~~ | ~~+5%~~ | FALSIFIED (-47%). target_feature breaks register alloc |
| ~~P0~~ | ~~Fused Q4K prefill~~ | ~~reduce penalty~~ | FALSIFIED (-52%). In-kernel dequant < HGEMM+FP8 |
| ~~P0~~ | ~~Eliminate rayon per-matmul overhead~~ | ~~+5-10% CPU~~ | FALSIFIED. scoped threads -77%, large-chunk -75%. 4 pool approaches deadlocked/regressed |
| **P1** | CPU KV cache workspace (remaining allocs) | +2-5% CPU | 3 days |
| **P1** | cuBLAS grouped GEMM (batch QKV) | +5% GPU | 2 weeks |
| **P2** | GPU persistent kernel | +10-15% GPU | 4 weeks |

**PMAT-318: Rayon replacement FALSIFIED.** `std::thread::scope` per-matmul: -77% (thread spawn
overhead: 896 spawns/token × 16 threads). Large-chunk rayon (1 chunk/thread): -75% (serial
execution from env detection). Rayon's work-stealing with 64-row chunks is the right approach —
the CPU gap is kernel quality, not dispatch overhead.

**CPU gap root cause (PMAT-304):** perf stat IPC 1.59 vs llama.cpp 1.01. The 0.58 excess IPC = Rust abstraction overhead between DRAM loads.

### Q8 Activation Cache for Batched DP4A (PMAT-294, Mar 21)

**Root cause found:** `batched_hw_dp4a_q4k_gemv_into` always re-quantized input to Q8_1, even when the same buffer was already quantized (K/V share input with Q, up shares with gate). The Q8 activation cache (`q8_activation_valid`) existed for M=1 but was never used in the batched (M>1) path.

**Fix:** Add Q8 cache check to batched path + add cache invalidation to graph dispatch (RMSNorm, residual add, SwiGLU, attention). Saves 84 Q8 launches per decode step (3/layer x 28).

| c | Pre-fix | Post-fix | Delta |
|---|---------|----------|-------|
| 1 | 148.6 | 148.6 | 0.0% |
| 4 | 78.9 | 80.2 | **+1.6%** |
| 8-32 | same | same | 0.0% |

+1.6% at c=4 only (DP4A active at M=2-4). At M>=5, FP8 cuBLASLt fires instead of DP4A, so Q8 cache is irrelevant.

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

**PMAT-267 corrected projections (per-step pipeline analysis):**

| Scenario | c=4 | c=8 | c=16 | c=32 |
|----------|-----|-----|------|------|
| ~~PMAT-265 (decode_rate 0.85×)~~ | ~~0.81×~~ | ~~0.82×~~ | ~~0.82×~~ | ~~0.81×~~ |
| ~~PMAT-266 (flat 17%)~~ | ~~0.58×~~ | ~~0.52×~~ | ~~0.52×~~ | ~~0.62×~~ |
| **PMAT-267 (50% overlap)** | **0.71×** | **0.63×** | **0.61×** | **0.73×** |
| **PMAT-267 (80% overlap)** | **0.79×** | — | — | — |

**PMAT-267 key insight:** PMAT-266's "4.6× GPU kernel gap" was misleading — it compared realizr total GPU (7.4ms, not 10ms) to a single vLLM GEMM call (2.17ms). The actual per-step wall-time gap is **2.0×** (13.8 vs 6.8ms). The step decomposes as: GPU kernels (7.4ms, 54%) + serving overhead (5.5ms, 40%) + CUDA launch (1.4ms, 10%). Graph + event-based sync enables **CPU-GPU pipelining**: GPU executes while CPU handles serving. Projected: max(7.4, serving×overlap%) + serving×(1−overlap%) ≈ **8.6-10.2ms** (50-80% overlap). PMAT-265's ~0.81× was approximately correct at high overlap. The achievable overlap is an implementation question.

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

**Per-step budget (M=4, 6,527 steps in 90s):** GPU kernels **7.4ms** (DP4A 2.8ms + attention 2.0ms + FP8 GEMM 2.2ms + other 0.4ms), launch 0.9ms, H2D 0.4ms, serving 5.5ms = **13.8ms total.** The 10ms stated in initial PMAT-266 was overstated — actual GPU kernel time is 7.4ms (54% of step), serving overhead is 5.5ms (40%). **Wall-time gap vs vLLM: 2.0× (not 4.6×).**

**Critical finding:** cuStreamSynchronize profile is **identical to PMAT-217** (batch-and-step: 82.4%, 10.4ms median). The iteration scheduler does NOT change the CUDA dispatch pattern — scheduling improvement is purely CPU-side slot management. Per-M graph + event-based sync enables **CPU-GPU pipelining** (serving overlaps GPU execution). Projected improvement: **42% at 50% overlap** (13.8 → 10.2ms), not the 17% from PMAT-266's naive launch-saving estimate.

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
