# Performance Baselines: GPU Decoder Throughput

**Parent Spec:** [REALIZAR-GPU-PERF-001](../gpu-performance-spec.md)
**Status:** Reference Document
**Date:** 2026-03-04

---

## 1. Hardware Reference Platform

| Component | Specification |
|-----------|--------------|
| GPU | NVIDIA RTX 4090 (Ada Lovelace, sm_89) |
| GPU Memory | 24GB GDDR6X |
| GPU Bandwidth | 1008 GB/s theoretical |
| L2 Cache | 72MB |
| Shared Memory | 100KB per SM |
| Tensor Cores | 4th Gen (FP16/BF16/INT8) |
| CPU | AMD Ryzen / Intel Core (DDR5-4800) |
| PCIe | Gen4 x16 |

---

## 2. Model Reference: Qwen2/Qwen2.5 Architecture

Per [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671):

| Parameter | 0.5B | 1.5B | 7B | 72B | 57B-A14B (MoE) |
|-----------|------|------|-----|-----|----------------|
| Hidden Size | 896 | 1,536 | 3,584 | 8,192 | 3,584 |
| Layers | 24 | 28 | 28 | 80 | 28 |
| Query Heads | 14 | 12 | 28 | 64 | 28 |
| KV Heads | 2 | 2 | 4 | 8 | 4 |
| GQA Ratio | 7:1 | 6:1 | 7:1 | 8:1 | 7:1 |
| Head Size | 64 | 128 | 128 | 128 | 128 |
| Intermediate Size | 4,864 | 8,960 | 18,944 | 29,568 | 2,560 |
| RoPE Theta | 1M | 1M | 1M | 1M | 1M |
| RMSNorm Epsilon | 1e-6 | 1e-6 | 1e-6 | 1e-6 | 1e-6 |
| Vocab Size | 151,646 | 151,646 | 151,646 | 151,646 | 151,646 |

**Qwen-Specific Characteristics:**
- Aggressive GQA ratios (6:1 to 8:1) for KV memory reduction
- Large RoPE theta (1M vs LLaMA's 10K) for extended context
- SwiGLU activation in FFN
- Smaller RMSNorm epsilon (1e-6 vs 1e-5)

---

## 3. Affected Model Families

All autoregressive decoder-only transformers with token-by-token generation:

| Family | Models | Architecture Notes |
|--------|--------|--------------------|
| LLaMA | 2-7B, 2-13B, 3-8B, 3-70B | GQA, SwiGLU, RoPE |
| Mistral | 7B, Nemo, Mixtral-8x7B | Sliding window attention |
| Phi | 2, 3-mini, 3-medium | LayerNorm + GELU, partial attention |
| Qwen | 7B, 14B, Qwen2-7B, 2-72B | Aggressive GQA, large RoPE theta |
| Others | Falcon, Yi, Gemma, GPT-J/NeoX | Various attention patterns |

---

## 4. Pre-Fix Baselines

### Initial State (Dec 2025)

| Metric | Value | Source |
|--------|-------|--------|
| GGUF CPU throughput | 3.0 tok/s | REALIZAR-QWEN-PERF-001 §1.1 |
| APR GPU throughput | 0.9 tok/s | REALIZAR-QWEN-PERF-001 §1.1 |
| SafeTensors throughput | 0.0 tok/s | REALIZAR-QWEN-PERF-001 §1.1 |
| Ollama baseline (GPU) | 240.1 tok/s | IMP-700 verification |
| GEMV latency (1×4096×4096) | 4.41ms | Decoder Throughput Spec §1.1 |
| Memory bandwidth utilization | 1.4% | Decoder Throughput Spec §1.1 |
| Gap to Ollama | 1,090x | IMP-700 |

### After KV Cache (PARITY-001)

| Metric | Value | Improvement |
|--------|-------|-------------|
| CPU throughput | 4.98 tok/s | 22.6x from 0.22 |
| Gap to Ollama | 40x | From 1,090x |

### After SIMD Optimization (PARITY-003)

| Metric | Value | Improvement |
|--------|-------|-------------|
| CPU throughput | 5.31 tok/s | 6.6% from 4.98 |
| Gap to Ollama | 38x | CPU vs GPU gap |

---

## 5. Post-Fix Baselines (Feb 2026)

### Internal Microbenchmarks (APR Q4 Adapter, M=batch)

| Metric | M=8 | M=16 | Notes |
|--------|-----|------|-------|
| Throughput (tok/s) | 740.5 | 583.6 | After Fixes 1-4, internal bench |
| Ollama ratio | 2.54x | 2.01x | vs Ollama 291.2 tok/s (internal) |
| PCIe transfers/token | 0 | 0 | Fully GPU-resident |

*Note: These numbers reflect internal microbenchmarks with batch scheduling. Production benchmarks under load test conditions (§5a) show different results.*

### CPU Performance (GGUF)

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | 12.5-17.3 tok/s | After GQA verification |
| Q4K routing | CPU-optimal | Fix 1 resolved GPU routing |

---

## 5a. Competition Baselines (Mar 2026)

Measured via `probador llm load` (60s, concurrency=4, 3 runs, 95% CI). Model: Qwen2.5-Coder-1.5B-Instruct Q4_K_M.

### GPU (RTX 4090) — 2026-03-04 (v3, with --skip-contract fix)

| Rank | Runtime | Tok/s | Decode tok/s | P50 (ms) | P95 (ms) | P99 (ms) | Error Rate |
|------|---------|-------|-------------|----------|----------|----------|------------|
| 1 | llama.cpp | **948.2** | **238.0** | 537.8 | 565.4 | 588.7 | 0% |
| 2 | ollama | **568.9** | **142.3** | 899.5 | 938.7 | 947.2 | 0% |
| 3 | realizar (safetensors) | **167.1** | **43.3** | 2,643.7 | 4,428.2 | 4,469.5 | 0% |
| 4 | realizar (GGUF) | **150.7** | **39.1** | 3,259.5 | 4,078.2 | 4,162.8 | 0% |
| 5 | realizar (APR native) | **143.3** | **39.9** | 2,728.3 | 3,560.9 | 4,057.3 | 0% |

*Previous (2026-03-03): APR native was 100% errors. Fixed with --skip-contract.*

### CPU (Intel EPYC, 192.168.50.100) — 2026-03-03

| Rank | Runtime | Tok/s | P50 (ms) | P95 (ms) | P99 (ms) | Error Rate |
|------|---------|-------|----------|----------|----------|------------|
| 1 | llama.cpp | **218.5** | 2,340 | 2,381 | 2,389 | 0% |
| 2 | ollama | **149.5** | 3,356 | 3,782 | 3,817 | 0% |
| 3 | realizar (safetensors) | **28.3** | 18,110 | 18,293 | 18,317 | 0% |
| 4 | realizar (GGUF) | **23.0** | 20,007 | 30,699 | 31,408 | 0% |
| 5 | realizar (APR native) | **9.5** | 53,263 | 54,537 | 54,537 | 0% |

### Gap Analysis (2026-03-04)

| Metric | Value | Notes |
|--------|-------|-------|
| Realizar best GPU vs llama.cpp | **5.7x slower** | safetensors: 167.1 vs 948.2 tok/s |
| Realizar best GPU vs ollama | **3.4x slower** | safetensors: 167.1 vs 568.9 tok/s |
| Realizar decode vs llama.cpp | **5.5x slower** | 43.3 vs 238.0 decode tok/s |
| APR native GPU | **FIXED** | 143.3 tok/s (was 0 tok/s, 100% errors) |
| All APR formats cluster | **39-43 decode tok/s** | Common bottleneck: kernel launch overhead |

**Improvement from v2 (Mar 3) → v3 (Mar 4):**
- APR native: 0 → 143.3 tok/s (PMAT-018 regression fixed)
- GGUF: 25.8 → 150.7 tok/s (+5.8x, --skip-contract removed overhead)
- SafeTensors: 96.5 → 167.1 tok/s (+1.7x)
- Gap to llama.cpp narrowed from 10.5x to 5.7x

**Source:** `bench-results-v2/*-gpu-20260304.json`

**GitHub Issues Filed:** #1 (6x gap), #4 (kernel launch), #5 (GEMV bandwidth)

---

## 6. Theoretical Limits

| Format | Theoretical Max (tok/s) | Basis |
|--------|-------------------------|-------|
| GGUF CPU (Q4_K) | ~35 | DDR5-4800, 0.5B model |
| GPU FP16 | ~1,800 | RTX 4090 BW, 0.5B model |
| GPU INT8 (SageAttention) | ~3,600 | 2x over FP16 |
| GPU INT4 (SageAttention2) | ~5,400 | 3x over FP16 |

*Conservative estimates. Actual peak depends on model size, sequence length, and batch size.*

---

## 7. Benchmarking Standards

### Measurement Protocol

Per [Hoefler & Belli SC'15](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf):

1. **CV-based stopping:** Auto-stop when CV < 0.05 (5% threshold)
2. **Warmup discard:** Separate warmup from measurement iterations
3. **Outlier detection:** MAD-based with k=1.4826 scale factor
4. **Percentiles:** Report p50, p95, p99 latencies
5. **Environment metadata:** OS, arch, CPU cores, Rust version, profile

### Benchmark Infrastructure

Implemented in PARITY-007 through PARITY-010:
- `CVStoppingBenchmark` — automatic convergence
- `WarmupBenchmark` — JIT/cache effect elimination
- `EnvironmentMetadata` — reproducibility
- `VersionedBenchmarkResult` — schema versioning

### External Contracts

For authoritative benchmark methodology and baselines, see:
- `qwen-coder-deploy/contracts/benchmarking-v2.md`
- `qwen-coder-deploy/contracts/inference-showdown-v1.yaml`

---

## 8. Threshold Registry

### Internal Microbenchmark Thresholds (Feb 2026)

| ID | Claim | Threshold | Unit | Status |
|----|-------|-----------|------|--------|
| THRESH-001 | Ollama baseline (internal) | >= 180 | tok/s | ✅ 291.2 |
| THRESH-002 | Realizar M=8 (internal) | >= 5 | tok/s | ✅ 740.5 |
| THRESH-005 | CV stability | < 0.05 | ratio | ✅ |
| THRESH-006 | KV cache speedup | >= 10x | ratio | ✅ 128x avg |
| THRESH-007 | GPU GEMM speedup | >= 10x | ratio | ✅ 57x |
| THRESH-008 | ContiguousKV speedup | >= 100x | ratio | ✅ 16,640x |
| THRESH-009 | Multi-acc SIMD speedup | >= 2x | ratio | Partial (6.6%) |
| THRESH-010 | FlashAttention speedup | >= 4x | ratio | Pending |

### Competition Benchmark Thresholds (Mar 2026)

| ID | Claim | Threshold | Unit | Status |
|----|-------|-----------|------|--------|
| THRESH-C01 | Ollama GPU baseline | >= 500 | tok/s | ✅ 607.9 |
| THRESH-C02 | llama.cpp GPU baseline | >= 900 | tok/s | ✅ 1,013.6 |
| THRESH-C03 | Realizar GPU (best) | >= 50 | tok/s | ✅ 167.1 (safetensors) |
| THRESH-C04 | Realizar GPU parity | <= 2x Ollama | ratio | ❌ **3.4x** (167.1 vs 568.9) |
| THRESH-C05 | APR native GPU | >= 1 | tok/s | ✅ **143.3** (was BROKEN, fixed Mar 4) |
| THRESH-C06 | GGUF GPU acceleration | >= 3x CPU | ratio | ⚠️ Pending CPU retest |
| THRESH-C07 | llama.cpp CPU baseline | >= 150 | tok/s | ✅ 218.5 |
| THRESH-C08 | Realizar decode parity | <= 2x llama.cpp decode | ratio | ❌ **5.5x** (43.3 vs 238.0) |

**Key Finding (Mar 4):** --skip-contract flag and bug fixes improved all APR formats significantly. Gap to ollama narrowed from 6.3x to 3.4x. APR native GPU regression (PMAT-018) resolved. Remaining bottleneck is kernel launch overhead (52.5% of decode time, ~180 launches/token).

---

## References

- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) — Architecture specification
- [Hoefler & Belli SC'15](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf) — Scientific benchmarking
- [Williams09] Williams, S., et al. (2009). "Roofline Model." CACM 52(4).
