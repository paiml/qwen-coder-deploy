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

## 7. Catastrophic Failure Protocol

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
