# PMAT Work Tickets: GPU Decoder Performance

**Parent Spec:** [REALIZAR-GPU-PERF-001](../gpu-performance-spec.md)
**Roadmap:** [roadmap.yaml](../../docs/roadmaps/roadmap.yaml)
**Status:** Active Tracking
**Date:** 2026-03-04

---

## 1. Completed Tickets

### QWEN-PMAT-001: GQA Broadcasting Fix

```yaml
id: QWEN-PMAT-001
pmat_id: PMAT-001
status: completed
priority: critical
title: "Fix GQA naive KV head broadcasting in attention"
acceptance_criteria:
  - "AC1: No .clone() in GQA attention hot path ✅ VERIFIED"
  - "AC2: Memory bandwidth reduced by GQA ratio (7x for Qwen2-7B) ✅ VERIFIED"
  - "AC3: Throughput >= 10 tok/s — ACTUAL: 12.5-17.3 tok/s ✅ EXCEEDED"
files: [src/gguf/inference/attention.rs, src/layers/attention.rs]
notes: "Already correctly implemented. 3.0 tok/s was outdated data."
```

### QWEN-PMAT-002: SwiGLU GPU Fusion

```yaml
id: QWEN-PMAT-002
pmat_id: PMAT-002
status: completed
priority: critical
title: "Wire fused SwiGLU kernel into APR Q4 adapter"
acceptance_criteria:
  - "AC1: Zero GPU↔CPU transfers during FFN forward pass ✅"
  - "AC2: Fused kernel from cuda/executor/activations.rs used in apr_q4.rs ✅"
  - "AC3: APR Q4 throughput >= 50 tok/s (pending benchmark)"
files: [src/gpu/adapters/apr_q4.rs, src/cuda/executor/activations.rs]
notes: "Eliminates 84 PCIe transfers for Qwen2-7B (3/layer × 28 layers)."
```

### QWEN-PMAT-011: GELU GPU Fusion

```yaml
id: QWEN-PMAT-011
pmat_id: PMAT-003
status: completed
priority: high
title: "Wire GELU GPU kernel into standard FFN"
acceptance_criteria:
  - "AC1: Zero GPU↔CPU transfers during standard FFN forward pass ✅"
  - "AC2: In-place gelu_gpu kernel used instead of CPU roundtrip ✅"
files: [src/gpu/adapters/apr_q4.rs, src/cuda/executor/quantized.rs]
```

### QWEN-PMAT-013: GPU RMSNorm + Residual

```yaml
id: QWEN-PMAT-013
pmat_id: PMAT-004
status: completed
priority: critical
title: "Wire GPU RMSNorm and fused residual kernels into APR Q4 adapter"
acceptance_criteria:
  - "AC1: Zero GPU↔CPU transfers for RMSNorm ✅"
  - "AC2: Zero GPU↔CPU transfers for residual connections ✅"
  - "AC3: All existing tests pass (45/45) ✅"
  - "AC4: Throughput >= 500 tok/s — ACHIEVED: 740.5 tok/s ✅"
files: [src/cuda/executor/layer.rs, src/gpu/adapters/apr_q4.rs]
```

---

## 2. Planned Tickets

### QWEN-PMAT-003: SageAttention INT8

```yaml
id: QWEN-PMAT-003
pmat_id: PMAT-008
status: planned
priority: high
title: "Implement SageAttention INT8 Q/K quantized attention"
estimated_effort: 5 days
labels: [sage-attention, quantization, trueno-gpu]
files: ["../trueno/trueno-gpu/src/kernels/attention.rs", src/cuda/executor/attention.rs]
```

### QWEN-PMAT-004: EAGLE Speculative Decoding

```yaml
id: QWEN-PMAT-004
pmat_id: PMAT-009
status: planned
priority: high
title: "Complete EAGLE speculative decoding for Qwen models"
estimated_effort: 7 days
labels: [speculative-decoding, eagle]
dependencies: [QWEN-PMAT-001, QWEN-PMAT-002]
files: [src/speculative.rs, src/gguf/cuda/speculative.rs, src/gguf/batch_scheduler.rs]
```

### QWEN-PMAT-005: Marlin-Style Kernel

```yaml
id: QWEN-PMAT-005
pmat_id: PMAT-010
status: planned
priority: medium
title: "Implement Marlin-style L2-optimized GPTQ kernel"
estimated_effort: 5 days
labels: [marlin, l2-cache]
```

### QWEN-PMAT-006: Dual Chunk Attention

```yaml
id: QWEN-PMAT-006
pmat_id: PMAT-011
status: planned
priority: medium
title: "Implement DCA for Qwen long context"
estimated_effort: 4 days
labels: [long-context, dca]
```

### QWEN-PMAT-007: KV Cache Quantization

```yaml
id: QWEN-PMAT-007
pmat_id: PMAT-005
status: completed
priority: medium
title: "Implement INT8 KV cache quantization"
estimated_effort: 4 days
labels: [kv-cache, quantization]
notes: "Phases 1-4 complete. AC5 (perplexity validation) pending."
```

### QWEN-PMAT-008: MInference Sparse Attention

```yaml
id: QWEN-PMAT-008
pmat_id: PMAT-012
status: planned
priority: medium
title: "Implement MInference sparse attention for long-context prefill"
estimated_effort: 5 days
labels: [sparse-attention, minference]
notes: "Combined with chunked prefill (32K chunks), reduces activation VRAM by 96.7%. Target: 3-6x prefill speedup."
```

### QWEN-009: 3-Way FFN Fusion

```yaml
id: QWEN-009
pmat_id: PMAT-006
status: completed
priority: medium
title: "Implement RMSNorm+Linear+Activation 3-way FFN fusion"
estimated_effort: 3 days
labels: [kernel-fusion]
notes: "FusedRmsNormGateUpSwigluQ4KKernel in trueno-gpu. Kernel done, 3 tests passing. AC2 (1.2x benchmark) remaining."
```

### QWEN-010: RTX 4090 Tile Tuning

```yaml
id: QWEN-010
pmat_id: PMAT-007
status: completed
priority: low
title: "RTX 4090 block size tuning"
estimated_effort: 1 day
labels: [tile-tuning, rtx4090]
notes: "detect_optimal_tile_size() auto-detects GPU: Ada Lovelace→64×64, others→32×32."
```

---

## 3. PMAT ID Cross-Reference

| QWEN Ticket | PMAT ID | Status |
|-------------|---------|--------|
| QWEN-PMAT-001 (GQA) | PMAT-001 | ✅ Completed |
| QWEN-PMAT-002 (SwiGLU) | PMAT-002 | ✅ Completed |
| QWEN-PMAT-011 (GELU) | PMAT-003 | ✅ Completed |
| QWEN-PMAT-013 (RMSNorm) | PMAT-004 | ✅ Completed |
| QWEN-PMAT-007 (KV Cache) | PMAT-005 | ✅ Completed |
| QWEN-009 (3-Way Fusion) | PMAT-006 | ✅ Completed |
| QWEN-010 (Tile Tuning) | PMAT-007 | ✅ Completed |
| QWEN-PMAT-003 (SageAttention) | PMAT-008 | Planned |
| QWEN-PMAT-004 (EAGLE) | PMAT-009 | Planned |
| QWEN-PMAT-005 (Marlin) | PMAT-010 | Planned |
| QWEN-PMAT-006 (DCA) | PMAT-011 | Planned |
| QWEN-PMAT-008 (MInference) | PMAT-012 | Planned |

---

## 4. Quality Gate Thresholds

| Metric | Threshold | Command |
|--------|-----------|---------|
| TDG Score | >= 93.0 (A Grade) | `pmat analyze tdg` |
| Dead Code | <= 15% | `pmat quality-gate --checks dead-code` |
| Cognitive Complexity | <= 25 | `pmat analyze complexity` |
| Cyclomatic Complexity | <= 30 | `pmat analyze complexity` |
| SATD | 0 critical | `pmat analyze satd` |
| Test Coverage | >= 80% lines | `make coverage` |
| Clippy Warnings | 0 | `make lint` |

---

## 5. Pre-Commit Protocol

```bash
# Tier 1: Sub-second (ON-SAVE)
make tier1

# Tier 2: Pre-commit (30s)
make tier2

# Full quality gate
pmat quality-gate --fail-on-violation

# Coverage (95% target)
make coverage-95
```

Any commit related to this spec must pass all tiers. If quality gate fails, the implementation is **REJECTED** regardless of throughput gains.
