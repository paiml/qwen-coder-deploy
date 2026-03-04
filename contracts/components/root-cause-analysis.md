# Root Cause Analysis: GPU Decoder Throughput

**Parent Spec:** [REALIZAR-GPU-PERF-001](../gpu-performance-spec.md)
**Status:** Reference Document
**Date:** 2026-03-04

---

## 1. Decode-Phase GEMV Bottleneck (Primary)

### 5 Whys (Toyota Way)

**Why #1:** Why is decode throughput 190x slower than Ollama?
Each token generation requires ~192 M=1 matrix-vector multiplications, each taking 4.41ms instead of 0.023ms.

**Why #2:** Why does each GEMV take 4.41ms instead of 0.023ms?
Memory bandwidth utilization is 1.4% of theoretical maximum (1008 GB/s on RTX 4090).

**Why #3:** Why is memory bandwidth utilization only 1.4%?
The GEMV kernel uses strided memory access patterns that defeat the GPU's memory coalescing hardware [McKee24].

**Why #4:** Why are memory accesses strided?
The kernel assigns one warp (32 threads) per output column, causing threads to read matrix elements 16KB apart (N × 4 bytes stride for N=4096).

**Why #5:** Why was the kernel designed with column-per-warp assignment?
The initial implementation prioritized algorithmic simplicity over memory access patterns.

### Root Cause Statement

> The GEMV kernel's thread-to-data mapping causes non-coalesced global memory reads, reducing effective memory bandwidth by 68x.

---

## 2. APR Format: Silent Corruption Hypothesis

**Observation:** GPU throughput was 0.9 tok/s (0.05% of theoretical max).

**Naïve Cause:** "The code is slow."

**Popperian Analysis:**
1. **Hypothesis:** Conversion corrupts tensor names/shapes, leading to a fallback "slow path" (CPU emulation or unoptimized kernel).
2. **Attack:** If tensors were merely named wrong, the graph shouldn't run at all (crash).
3. **Refined Root Cause:** The APR converter *force-validates* incorrect tensor mappings (e.g., mapping GQA heads to MHA slots), causing the inference engine to perform excessive broadcasting or memory copies during `forward()`, destroying locality.

**Status:** Resolved — APR Q4 adapter now runs fully GPU-resident after Fixes 2-4 (see main spec §4).

---

## 3. GGUF CPU: Memory Waste Hypothesis

**Observation:** 3.0 tok/s on CPU.

**Hypothesis:** GQA implementation naively broadcasts KV pairs (Qwen2.5 uses 7:1 GQA ratio). Physical KV head expansion to match query heads consumes 7x necessary bandwidth.

**Falsification Test:**
- If memory bandwidth utilization > 80% at 3.0 tok/s → **CONFIRMED** (bandwidth bound by waste)
- If utilization < 10% → **FALSIFIED** (bottleneck is compute/latency)

**Resolution:** Verified that GQA was already correctly implemented via integer division (`kv_head = q_head / q_per_kv`). The 3.0 tok/s figure was outdated — actual throughput was 12.5-17.3 tok/s. The real CPU bottleneck was Q4K dequantization routing to CPU instead of GPU (Fix 1).

---

## 4. GPU↔CPU Transfer Overhead

**Observation:** Multiple operations in the APR Q4 GPU adapter performed unnecessary round-trips:

| Operation | Transfers/Layer | Direction |
|-----------|----------------|-----------|
| SwiGLU activation | 3 | GPU→CPU→GPU |
| GELU activation | 2 | GPU→CPU→GPU |
| RMSNorm | 2 | GPU→CPU→GPU |
| Residual add | 1 | GPU→CPU |

For a 28-layer model (Qwen2-7B), this totaled **224+ PCIe round-trips** per token.

**Root Cause:** Initial GPU adapter implemented compute operations but fell back to CPU for activation functions that lacked dedicated GPU kernels.

**Resolution:** Fixes 2-4 wired GPU-resident kernels for all operations, eliminating all PCIe transfers in the hot path.

---

## 5. Async Runtime Blocking

**Observation:** Even after eliminating PCIe transfers, throughput plateaued due to tokio runtime starvation.

**Root Cause:** Synchronous GPU inference blocked the async runtime's executor threads, preventing concurrent request handling.

**Resolution:**
- Fix 3: `spawn_blocking` for GPU inference to avoid blocking async executor
- Fix 5: Queue-based dispatch with backpressure (bounded channel, 32 slots)
- Fix 6: Continuous batching with configurable batch intervals

---

## 6. The "Impossible" Observation

The fact that GGUF CPU (3.0 tok/s) outperformed APR GPU (0.9 tok/s) falsified the hypothesis that "GPUs are inherently faster." It proved the existence of software overhead so severe it negated 1000x raw hardware superiority.

| Format | Initial (tok/s) | Theoretical Max (tok/s) | Efficiency |
|--------|-----------------|-------------------------|------------|
| GGUF (CPU) | 3.0 | ~35.0 | 8.5% |
| APR (GPU) | 0.9 | ~1800.0 | 0.05% |
| SafeTensors | 0.0 | ~1800.0 | 0.0% |

*Theoretical Max: DDR5-4800 (CPU), RTX 4090 (GPU). 0.5B model ~400MB. RTX 4090 BW ~1000GB/s.*

---

## 7. Kernel Launch Overhead (PMAT-015, Discovered 2026-03-02)

**Observation:** Host-side profiling shows **52.5% of decode time** (128,484µs) is kernel launch overhead, not actual computation.

**Evidence:**
- Nsight Systems shows individual kernels are fast (e.g., `fused_swiglu` 0.9µs, `residual_add` 1.0µs)
- Host-side telemetry shows AttentionScore at 88,390µs (76% of decode)
- The gap is kernel dispatch/synchronization overhead from ~180 individual kernel launches per token

**5 Whys:**
1. **Why is decode 130.7 tok/s (not 1000+)?** Kernel launch overhead consumes 52.5% of time
2. **Why is launch overhead so high?** ~180 separate kernel launches per token (28 layers × ~6 ops/layer + overhead)
3. **Why so many launches?** Each operation (GEMV, RMSNorm, residual, SwiGLU, RoPE, attention) is a separate kernel
4. **Why aren't they fused?** Only SwiGLU is fused so far; the 3-way fusion (QWEN-009) is kernel-complete but not integrated into the serving path
5. **Why not CUDA graphs?** CUDA graph capture (`SKIP_CUDA_GRAPH=1` visible in profile) is disabled during profiling; unclear if enabled in production

**Root Cause Statement:**
> Excessive kernel launch count (~180/token) causes 52.5% overhead. The gap between kernel compute time (fast) and end-to-end latency (slow) is dominated by CUDA driver dispatch and synchronization.

**Mitigation:**
1. **CUDA Graphs** — capture the full decode forward pass as a single graph (eliminates per-launch overhead)
2. **Aggressive kernel fusion** — reduce kernel count from ~180 to ~28 (one per layer) via fused attention + fused FFN
3. **Stream-based pipelining** — overlap kernel launches with compute on multiple CUDA streams

---

## 8. APR Native GPU Regression (PMAT-016, Discovered 2026-03-03)

**Observation:** APR native format shows **100% error rate** on GPU (0 tok/s) across all 3 runs of competition benchmarks. ~3,913 requests failed per 60s run.

**Evidence:** `bench-results-v2/apr-apr-gpu-20260303.json`:
- Run 1: 3,909 failed / 0 successful
- Run 2: 3,914 failed / 0 successful
- Run 3: 3,915 failed / 0 successful

**Context:**
- APR native works on CPU (9.5 tok/s, 0% errors)
- Previous internal microbenchmarks showed 740.5 tok/s (M=8) — this may have been a different code path or build
- SafeTensors GPU path works (96.5 tok/s)
- GGUF GPU path works but slow (25.8 tok/s)

**Hypothesis:** The APR native GPU inference endpoint crashes or returns errors under concurrent load (c=4). Possible causes:
1. GPU memory allocation failure under concurrent requests
2. Queue-based dispatch (Fix 5) rejecting requests under load
3. APR format-specific GPU code path has a concurrency bug
4. Model loading race condition for APR format

**Status:** P0 regression, investigation required.

---

## 9. Catastrophic Failure Protocol

If all hypotheses are falsified (throughput remains < 5 tok/s after fixes):

1. **Stop Engineering.** Do not "tweak" parameters.
2. **Audit the Clock.** Verify `std::time::Instant` precision and overhead.
3. **Audit the Bus.** Verify PCIe link width/speed (`lspci -vv`).
4. **Audit the Kernels.** Check for 1-thread-per-block launch errors.

---

## References

- [McKee24] McKee, D. (2024). "GPU Atomic Performance Modeling." Vulkanised 2024.
- [Abdelkhalik23] Abdelkhalik, H., et al. (2023). "Modeling NVIDIA Ampere GPU Performance." MEMSYS '23.
- [Williams09] Williams, S., et al. (2009). "Roofline Model." CACM 52(4). DOI:10.1145/1498765.1498785.
- [Liker04] Liker, J.K. (2004). "The Toyota Way." McGraw-Hill.
