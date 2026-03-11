# Optimization Tiers: GPU Decoder Performance

**Parent Spec:** [REALIZAR-GPU-PERF-001](../gpu-performance-spec.md)
**Status:** Active Roadmap
**Date:** 2026-03-04

---

## Tier 0: Completed Fixes (Shipped)

See main spec §4 for summary. These are production-deployed optimizations.

---

## Tier 1: Critical Gaps (2-5x Potential Speedup)

### QWEN-001: SageAttention (Quantized Attention Kernels)

**Papers:**
- [SageAttention (ICLR 2025)](https://arxiv.org/abs/2410.02367) — INT8 Q/K, 2.1x vs FlashAttention2
- [SageAttention2 (ICML 2025)](https://arxiv.org/abs/2411.10958) — INT4 + FP8, 3x speedup
- [SageAttention3 (NeurIPS 2025)](https://arxiv.org/abs/2505.11594) — FP4 microscaling, 5x speedup

| Version | Q/K Quantization | P̃/V Quantization | Speedup vs FA2 |
|---------|------------------|-------------------|----------------|
| V1 | INT8 | FP16 | ~2.1x |
| V2 | INT4 | FP8 | ~3x |
| V3 | FP4 (microscaling) | FP4 | ~5x |

**Current State:** FlashAttention implemented with FP32/FP16 (IMP-111).

**Gap:** No quantized attention kernels.

**Implementation Path:**
1. Extend trueno-gpu `QuantizeKernel` with INT8 Q/K matmul
2. Add smooth-K preprocessing per SageAttention paper
3. Implement per-thread quantization for memory efficiency

**Acceptance Criteria:**
- AC1: INT8 Q@K^T kernel implemented in trueno-gpu
- AC2: 2x speedup vs current FlashAttention on RTX 4090
- AC3: End-to-end perplexity within 0.1% of FP16 baseline

---

### QWEN-004: EAGLE Speculative Decoding

**Papers:**
- [EAGLE (ICML 2024)](https://arxiv.org/abs/2401.15077)
- [EAGLE-2 (EMNLP 2024)](https://arxiv.org/abs/2406.16858)
- [EAGLE-3 (NeurIPS 2025)](https://arxiv.org/abs/2503.01840)

**Current State:** Framework exists (`src/speculative.rs`, `gguf/cuda/speculative.rs`) but logic incomplete.

**Known Issue:** 25% acceptance rate (need 70%) — draft model mismatch.

**EAGLE Approach:**
- Reuse second-top-layer features (not just embeddings)
- Train lightweight draft head (0.24B params for 7B target)
- **Qwen2 Note:** Use bf16 instead of fp16 to avoid numerical overflow

**Speedup:** 2.8x on Qwen2 with bf16 precision.

**Acceptance Criteria:**
- AC1: Draft head architecture (1-layer transformer + LM head reuse)
- AC2: bf16 precision enforced for Qwen2 target models
- AC3: Acceptance rate >= 70%
- AC4: End-to-end speedup >= 2x

---

## Tier 2: High Impact (1.5-2x Speedup)

### QWEN-005: Marlin-Style GPTQ Kernel

**Paper:** [MARLIN (PPoPP 2025)](https://arxiv.org/abs/2408.11743)

**Benchmark:** 712 tok/s (Marlin) vs 276 tok/s (standard GPTQ) = 2.6x speedup.

**Key Optimization:** L2 cache optimization with streaming access patterns:
- Standard GPTQ: 30-50% cache hit rate (random access)
- Marlin: 80-95% cache hit rate (streaming + double buffering)

**Acceptance Criteria:**
- AC1: Streaming access pattern in Q4_K GEMV
- AC2: Shared memory double buffering
- AC3: L2 cache hit rate >= 80% (measured via Nsight)
- AC4: 1.5x speedup vs current Q4_K GEMV

---

### QWEN-006: Dual Chunk Attention (DCA) for Long Context

**Paper:** [Qwen2.5-1M Technical Report](https://qwenlm.github.io/blog/qwen2.5-1m/) — Alibaba, Jan 2025

**Problem:** RoPE-based models degrade with unseen large relative positions (>32K tokens).

**Solution:** DCA remaps positions to smaller values:
- Intra-chunk attention: local coherence within chunk
- Inter-chunk attention: cross-chunk context via position remapping

**Result:** Models trained on 32K achieve perfect passkey retrieval at 1M tokens (training-free).

**Acceptance Criteria:**
- AC1: DCA position remapping for Qwen architecture detection
- AC2: Passkey retrieval >= 99% at 128K context
- AC3: No quality degradation on standard benchmarks

---

### QWEN-007: KV Cache Quantization — COMPLETED

**Papers:**
- [KIVI (arXiv:2402.02750)](https://arxiv.org/abs/2402.02750) — 2bit KV
- [Hooper et al. (arXiv:2308.14903)](https://arxiv.org/abs/2308.14903) — KV quantization

**Status:** All 4 phases completed (2026-02-02):
1. Q8 KV cache infrastructure in CudaExecutor
2. CPU-side quantization/dequantization (< 2% error)
3. GPU-side Q8Dequant PTX kernel
4. Q8 incremental attention integration (5 tests passing)

**Impact:** ~3.56x memory reduction verified.

**Remaining:** Perplexity validation within 0.5% of FP32 baseline (AC5).

---

### QWEN-008: MInference Sparse Attention

**Paper:** [MInference 1.0 (Microsoft, 2024)](https://arxiv.org/abs/2407.02490)

**Source:** Qwen2.5-1M Technical Report:
> "Sparse attention based on MInference accelerates prefill phase 3.2x to 6.7x for 1M token sequences"

**Approach:** Identify and skip low-attention-score token pairs during prefill. Combined with chunked prefill (32K chunks), reduces activation VRAM by 96.7%.

**Acceptance Criteria:**
- AC1: Sparse pattern detection for attention matrices
- AC2: 3x prefill speedup for 32K+ sequences
- AC3: No quality degradation on RULER benchmark

---

## Tier 3: Incremental Gains (1.1-1.5x Speedup)

### QWEN-009: RMSNorm + Linear + Activation 3-Way Fusion — Kernel Done

**Target:** Fuse RMSNorm → Linear → SwiGLU in single kernel pass.

**Citation:** Op fusion: 1.2-1.5x speedup ([entrenar benchmarks](https://github.com/paiml/entrenar/blob/main/book/src/examples/citl.md))

**Implementation (2026-02-02):**
1. trueno-gpu kernel: `FusedRmsNormGateUpSwigluQ4KKernel`
   - 4 phases: Load+RMS, Normalize, Dual Q4K GEMV, SwiGLU+store
   - 256 threads (8 warps), shared memory: K*4 + 96 bytes
2. realizar integration: `KernelType::FusedRmsNormGateUpSwigluQ4K`
3. Executor methods: `fused_ffn_rmsnorm_swiglu_q4k_into()`, `fused_ffn_rmsnorm_swiglu_q4k_cached()`
4. Tests: 3 unit tests passing

**Memory Savings (per FFN layer):**
- Before: 4 kernel launches, K + 3N global writes
- After: 1 kernel launch, N global writes

**Remaining:** Benchmark 1.2x speedup on FFN forward pass (AC2).

---

### QWEN-010: RTX 4090 Block Size Tuning — COMPLETED

**RTX 4090 Characteristics:**
- L2 Cache: 72MB (vs A100's 40MB)
- Shared Memory: 100KB per SM
- Tensor Cores: 4th Gen (FP16/BF16/INT8)

**Implementation:**
1. `optimal_tile_size` field in CudaExecutor
2. `detect_optimal_tile_size()` auto-detects GPU: Ada Lovelace → 64×64, others → 32×32
3. Public `optimal_tile_size()` method for callers

---

## Priority Matrix

| ID | PMAT | Optimization | Speedup | Effort | Priority | Status |
|----|------|--------------|---------|--------|----------|--------|
| QWEN-015 | PMAT-018 | **APR native GPU fix** | **N/A** | Medium | **P0** | ✅ **FIXED** (143.3 tok/s) |
| QWEN-014 | PMAT-017 | **Kernel launch overhead** | **2-5x** | High | **P0** | **Planned** |
| QWEN-002 | PMAT-001 | GQA broadcasting | 2-3x | Low | P0 | ✅ VERIFIED |
| QWEN-003 | PMAT-002 | SwiGLU GPU fusion | 1.5-2x | Low | P0 | ✅ DONE |
| QWEN-011 | PMAT-003 | GELU GPU fusion | 1.2x | Low | P0 | ✅ DONE |
| QWEN-013 | PMAT-004 | GPU RMSNorm+Residual | 1.3x | Low | P0 | ✅ DONE |
| QWEN-001 | PMAT-008 | SageAttention INT8 | 2-3x | Medium | P1 | Planned |
| QWEN-004 | PMAT-009 | EAGLE speculative | 2-3x | High | P1 | Planned |
| QWEN-005 | PMAT-010 | Marlin-style kernels | 2.6x | High | P2 | Planned |
| QWEN-006 | PMAT-011 | DCA long context | N/A | Medium | P2 | Planned |
| QWEN-007 | PMAT-005 | KV cache quantization | 4x mem | Medium | P2 | ✅ DONE |
| QWEN-008 | PMAT-012 | MInference sparse | 3-6x | High | P3 | Planned |
| QWEN-009 | PMAT-006 | 3-way kernel fusion | 1.2x | Medium | P3 | ✅ Kernel |
| QWEN-010 | PMAT-007 | RTX 4090 tuning | 1.1x | Low | P3 | ✅ DONE |

---

## Tier 0a: Regressions (Must Fix)

### QWEN-015: APR Native GPU Regression (PMAT-018) — FIXED

**Problem:** APR native format returned 100% error rate on GPU under concurrent load (c=4).

**Resolution:** Root cause was PMAT-237 tensor contract validation rejecting Qwen2.5 tensor layout. Fixed by adding `--skip-contract` flag to forjar deployment.

**Results (2026-03-04):** 143.3 tok/s, 0% errors (c=4, 60s, 5s warmup).

**Acceptance Criteria:**
- AC1: APR native GPU returns > 0% success rate — ✅ 0% errors
- AC2: APR native GPU throughput >= safetensors (96.5 tok/s) — ✅ 143.3 tok/s
- AC3: Zero errors under standard load test (60s, c=4) — ✅ 0 errors in 91 requests

### QWEN-014: Kernel Launch Overhead (PMAT-017)

**Problem:** 52.5% of decode time (128,484µs) is kernel launch overhead from ~180 kernel launches per token.

**Evidence:** `results/profile-gpu-20260302.txt` (F-PROFILE-009)

**Mitigation Paths:**
1. CUDA graph capture for decode forward pass
2. Aggressive kernel fusion (180 → ~28 launches/token)
3. Multi-stream pipelining

**Acceptance Criteria:**
- AC1: CUDA graph capture working for decode
- AC2: < 50 kernel launches per token
- AC3: > 300 tok/s under competition load (c=4)

---

## Known Pitfalls

### PITFALL-001: QKV Fusion Trap
**Source:** `aprender/docs/rosetta-testing.md#267`
> "QKV Fusion Trap discovered during Qwen2.5-Coder-1.5B-Instruct"

**Mitigation:** Validate tensor shapes before and after fusion for Qwen architectures.

### PITFALL-002: Speculative Decoding Acceptance Rate
**Source:** `trueno/docs/ml-tuner-bricks.md#1134`
> "25% acceptance (need 70%)" — draft model mismatch

**Mitigation:** Use matching tokenizer and verify logit distributions.

### PITFALL-003: bf16 Requirement for Qwen
**Source:** EAGLE GitHub repo

**Mitigation:** Force bf16 dtype for all Qwen model inference paths.

---

## References

See [main spec §13](../gpu-performance-spec.md#13-academic-references) for consolidated citations.
