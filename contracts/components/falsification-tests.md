# Falsification Tests: GPU Decoder Performance

**Parent Spec:** [REALIZAR-GPU-PERF-001](../gpu-performance-spec.md)
**Methodology:** Popperian Critical Rationalism
**Date:** 2026-03-04

> "The game of science is, in principle, without end." — Karl Popper, *The Logic of Scientific Discovery*

---

## 1. GEMV Kernel Hypotheses

### H1: Memory Coalescing Hypothesis

**Claim:** Restructuring thread assignment to read consecutive addresses will increase bandwidth utilization from 1.4% to >90%.

**Falsification test:** `nvprof --metrics gld_efficiency`
**Prediction:** gld_efficiency > 0.90
**Falsified if:** efficiency < 50%

### H2: Latency Reduction Hypothesis

**Claim:** Coalesced GEMV will reduce 1×4096×4096 latency from 4.41ms to <0.05ms.

**Falsification test:** 1000 iterations, compute mean latency
**Prediction:** mean_latency < 0.05ms
**Falsified if:** mean > 0.1ms

### H3: Throughput Parity Hypothesis

**Claim:** With coalesced GEMV, decode throughput will reach >200 tok/s on RTX 4090.

**Falsification test:** End-to-end LLaMA-7B inference, measure tok/s
**Prediction:** throughput > 200 tok/s
**Falsified if:** throughput < 150 tok/s

**Note:** H3 depends on attention, KV cache, quantization — refinement: "GEMV throughput >10,000 ops/s for 1×4096×4096."

### H4: Vectorization Benefit Hypothesis

**Claim:** float4 (128-bit) loads will provide 2-4x bandwidth improvement over float (32-bit).

**Falsification test:** Compare `ld.global.f32` vs `ld.global.v4.f32`
**Prediction:** vectorized_bandwidth / scalar_bandwidth > 2.0
**Falsified if:** improvement < 1.5x

### H5: Occupancy Independence Hypothesis

**Claim:** For memory-bound GEMV, increasing occupancy beyond 50% won't significantly improve performance [Volkov10].

**Falsification test:** Vary block size 64-1024, measure throughput
**Prediction:** throughput_ratio(1024/256) < 1.2
**Falsified if:** 1024-thread blocks >20% faster than 256-thread

---

## 2. APR Throughput Hypotheses

### H-APR1: Silent Corruption Hypothesis

**Claim:** APR throughput < 1 tok/s because `convert` incorrectly maps Qwen2 GQA tensors.

**Prediction (P1):** Correcting mapping → immediate jump to > 50 tok/s.
**Falsified if:** Throughput < 10 tok/s after fix.

**Status:** Partially confirmed — mapping was correct but GPU↔CPU transfers were the primary bottleneck. After eliminating transfers: 740.5 tok/s.

### H-APR2: SafeTensors Cold Cache Myth

**Claim:** SafeTensors inference is slow only due to conversion overhead.

**Prediction (P2):** Warm run statistically indistinguishable from native APR.
**Falsified if:** |Latency(warm) - Latency(APR)| > 5%.

### H-APR3: Memory Wall Delusion (GGUF CPU)

**Claim:** Not compute bound; bound by redundant GQA memory operations.

**Prediction (P3):** Reducing KV duplication → linear throughput increase.
**Falsified if:** Throughput improves < 50% despite removing 85% of KV traffic.

**Status:** FALSIFIED — GQA was already correctly implemented. The 3.0 tok/s was outdated; actual was 12.5-17.3 tok/s. Bottleneck was elsewhere (Q4K CPU routing).

---

## 3. GQA Fix Verification

| # | Test | Hypothesis | Pass Criteria | Status |
|---|------|------------|---------------|--------|
| A1 | `test_gqa_no_clone` | GQA uses index remapping | No `.clone()` in profile | ✅ PASS |
| A2 | `test_gqa_bandwidth` | Memory reduced by GQA ratio | BW < baseline / 7 | ✅ PASS |
| A3 | `test_gqa_throughput` | Throughput scales linearly | >= 10 tok/s | ✅ PASS (12.5-17.3) |

---

## 4. SwiGLU Fusion Verification

| # | Test | Hypothesis | Pass Criteria | Status |
|---|------|------------|---------------|--------|
| B1 | `test_swiglu_no_transfer` | No PCIe round-trip | 0 cudaMemcpy in profile | ✅ PASS |
| B2 | `test_swiglu_fused_kernel` | Single kernel for FFN | 1 kernel launch/layer | ✅ PASS |
| B3 | `test_apr_q4_throughput` | GPU-resident FFN | >= 50 tok/s | ✅ PASS (740.5) |

---

## 5. Attention Quantization Verification

| # | Test | Hypothesis | Pass Criteria | Status |
|---|------|------------|---------------|--------|
| C1 | `test_sage_int8_speedup` | 2x faster than FP16 | latency < baseline / 2 | Pending |
| C2 | `test_sage_quality` | No quality degradation | perplexity delta < 0.1% | Pending |
| C3 | `test_sage_memory` | Reduced memory | peak < baseline × 0.6 | Pending |

---

## 6. PMAT Compliance Hypothesis

**Hypothesis (H-PMAT):** All optimization work can be completed while maintaining quality gates.

**Falsification Conditions (STOP and refactor if any occur):**
1. TDG Score drops below 90.0
2. New SATD comments introduced
3. Test coverage drops below 75%
4. Complexity exceeds cognitive 25

---

## 7. Pre-Flight Controls

| # | Control | Threshold | Notes |
|---|---------|-----------|-------|
| PF1 | CPU Frequency | > 3.0 GHz (No Powersave) | Avoid throttling |
| PF2 | GPU Link Speed | PCIe Gen4 x16 | Full bandwidth |
| PF3 | Background Load | < 1.0 Load Avg | Minimize interference |

---

## 8. Additional QA Tests

| # | Check | Test | Expected | Points |
|---|-------|------|----------|--------|
| E9 | cuBLAS comparison | `cublasSgemv` same dims | Within 1.5x | 3 |
| E10 | Bank conflict check | `nvprof shared_efficiency` | >90% | 2 |
| E11 | L2 cache hit rate | `nvprof l2_hit_rate` | >50% | 2 |
| E12 | Warp efficiency | `nvprof warp_execution_efficiency` | >90% | 2 |

---

## 9. Catastrophic Failure Protocol

If all hypotheses are falsified (throughput < 5 tok/s after all fixes):

1. **Stop Engineering.** Do not "tweak" parameters.
2. **Audit the Clock.** Verify `std::time::Instant` precision.
3. **Audit the Bus.** `lspci -vv` for PCIe link.
4. **Audit the Kernels.** Check for 1-thread-per-block launch errors.
5. **Audit the Profiler.** Verify measurements with independent tools.

---

## References

- [Popper59] Popper, K.R. (1959). "The Logic of Scientific Discovery." Hutchinson.
- [Volkov10] Volkov, V. (2010). "Better Performance at Lower Occupancy." GTC 2010.
- [Hoefler15] Hoefler, T. & Belli, R. (2015). "Scientific Benchmarking." SC15.
