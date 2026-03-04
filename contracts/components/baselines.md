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

### GPU Performance (APR Q4 Adapter)

| Metric | M=8 | M=16 | Notes |
|--------|-----|------|-------|
| Throughput (tok/s) | 740.5 | 583.6 | After Fixes 1-4 |
| Ollama ratio | 2.54x | 2.01x | Exceeds Ollama |
| PCIe transfers/token | 0 | 0 | Fully GPU-resident |

### CPU Performance (GGUF)

| Metric | Value | Notes |
|--------|-------|-------|
| Throughput | 12.5-17.3 tok/s | After GQA verification |
| Q4K routing | CPU-optimal | Fix 1 resolved GPU routing |

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

| ID | Claim | Threshold | Unit | Status |
|----|-------|-----------|------|--------|
| THRESH-001 | Ollama baseline | >= 180 | tok/s | ✅ 240.1 |
| THRESH-002 | Realizar current | >= 5 | tok/s | ✅ 740.5 |
| THRESH-003 | Gap to Ollama | <= 50x | ratio | ✅ 0.39x (we're faster) |
| THRESH-004 | Parity target | <= 1.25x | ratio | ✅ 0.39x |
| THRESH-005 | CV stability | < 0.05 | ratio | ✅ |
| THRESH-006 | KV cache speedup | >= 10x | ratio | ✅ 128x avg |
| THRESH-007 | GPU GEMM speedup | >= 10x | ratio | ✅ 57x |
| THRESH-008 | ContiguousKV speedup | >= 100x | ratio | ✅ 16,640x |
| THRESH-009 | Multi-acc SIMD speedup | >= 2x | ratio | Partial (6.6%) |
| THRESH-010 | FlashAttention speedup | >= 4x | ratio | Pending |

---

## References

- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) — Architecture specification
- [Hoefler & Belli SC'15](https://htor.inf.ethz.ch/publications/img/hoefler-scientific-benchmarking.pdf) — Scientific benchmarking
- [Williams09] Williams, S., et al. (2009). "Roofline Model." CACM 52(4).
