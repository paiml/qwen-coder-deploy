# Kernel Specifications: GPU Decoder Performance

**Parent Spec:** [REALIZAR-GPU-PERF-001](../gpu-performance-spec.md)
**Status:** Reference Document
**Date:** 2026-03-04

---

## 1. Coalesced GEMV Kernel

**Purpose:** M=1 matrix-vector multiplication for autoregressive decode.

### Problem

Non-coalesced access pattern:
```
Block b computes y[b]
Thread t reads A[t, b], A[t+32, b], ... (stride = N×4 bytes)
→ 16KB stride between consecutive thread reads
→ 1.4% memory bandwidth utilization
```

### Solution

Coalesced access pattern:
```
Block b computes y[b×256 : (b+1)×256]  (256 outputs per block)
Thread t reads A[row, b×256 + t]        (stride = 4 bytes, COALESCED)
Shared memory caches x[row] for all threads
→ 128 consecutive 4-byte addresses = ONE 512-byte transaction
→ Maximum memory coalescing achieved
```

### Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Threads per block | 256 | 8 warps, good occupancy |
| Outputs per block | 256 | One output per thread |
| Shared memory | 16KB | Cache x vector (4096 floats max) |
| Grid size | ceil(N/256) | Cover all outputs |
| Registers per thread | ≤64 | Maintain occupancy |

### Implementation

**trueno-gpu:** `CoalescedGemvKernel` in `trueno-gpu/src/kernels/gemv.rs`

```rust
pub struct CoalescedGemvKernel { k: u32, n: u32 }

impl Kernel for CoalescedGemvKernel {
    fn name(&self) -> &str { "gemv_coalesced" }
    fn build_ptx(&self) -> PtxKernel {
        PtxKernel::new("gemv_coalesced")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U64, "a_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U32, "k_dim")
            .param(PtxType::U32, "n_dim")
            .shared_memory(self.k * 4)
            .build(|ctx| { /* ... */ })
    }
}
```

**realizar dispatch** (`src/cuda.rs`):
```rust
let (kernel_type, cache_key) = if m == 1 {
    (KernelType::CoalescedGemv { k, n }, format!("gemv_coalesced_{}_{}", k, n))
} else {
    // existing GEMM path
};
```

---

## 2. Fused SwiGLU GPU Kernel (QWEN-003)

**Purpose:** Eliminate GPU↔CPU transfers for SwiGLU activation in FFN layers.

**Source:** `cuda/executor/activations.rs` (PAR-023)

### Before (3 PCIe transfers)
```rust
let gate = gpu_to_host(&gate_gpu)?;
let up = gpu_to_host(&up_gpu)?;
let activated: Vec<f32> = gate.iter().zip(up.iter())
    .map(|(&g, &u)| silu(g) * u).collect();
let activated_gpu = GpuBuffer::from_host(executor.context(), &activated)?;
```

### After (GPU-only)
```rust
let activated_gpu = executor.fused_swiglu_gpu(&gate_gpu, &up_gpu, intermediate_dim as u32)?;
```

**Impact:** Eliminates 3 transfers × 28 layers = 84 PCIe round-trips for Qwen2-7B.

**Citation:** [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — Shazeer, 2020.

---

## 3. In-Place GELU GPU Kernel (QWEN-011)

**Purpose:** Eliminate GPU↔CPU transfers for standard FFN (non-SwiGLU models).

### Before (2 PCIe transfers)
```rust
let up = gpu_to_host(&up_gpu)?;
let activated: Vec<f32> = up.iter().map(|&x| gelu(x)).collect();
let activated_gpu = GpuBuffer::from_host(executor.context(), &activated)?;
```

### After (GPU-only, in-place)
```rust
executor.gelu_gpu(&up_gpu, intermediate_dim as u32)?;
// up_gpu now contains activated values
```

**Impact:** Eliminates 2 transfers per FFN layer for Phi/GPT-style models.

---

## 4. GPU RMSNorm + Residual Kernels (QWEN-013)

**Purpose:** Keep normalization and residual connections GPU-resident.

### Changes
1. `get_rmsnorm_gamma_ptr()` — expose cached gamma buffer pointers
2. `upload_weights()` — cache all norm weights on GPU
3. `forward_layer()` — use GPU-resident `rmsnorm_gpu_ptr()` and `residual_add_gpu()`
4. Output norm via GPU RMSNorm

### Before
```rust
let normed = executor.rmsnorm(&hidden, &gamma, eps)?;  // GPU→CPU→GPU
let residual = hidden.iter().zip(out.iter()).map(|(h, o)| h + o).collect();  // CPU
```

### After
```rust
let normed_gpu = executor.rmsnorm_gpu_ptr(input, gamma_ptr, gamma_len, hidden_dim, eps)?;
let residual = executor.residual_add_gpu(input, &out_gpu, hidden_dim)?;
```

**Benchmark:** M=8: 740.5 tok/s (2.54x Ollama), M=16: 583.6 tok/s (2.01x Ollama).

---

## 5. Fused RMSNorm+GateUp+SwiGLU Q4K Kernel (QWEN-009)

**Purpose:** 3-way fusion to minimize kernel launches and global memory writes.

### Architecture
```
Phase 1: Load input + compute RMS sum
Phase 2: Normalize in shared memory
Phase 3: Dual Q4K GEMV (gate + up projections)
Phase 4: SwiGLU activation + store output
```

**trueno-gpu:** `FusedRmsNormGateUpSwigluQ4KKernel` in `trueno-gpu/src/kernels/quantize/fused.rs`
- 256 threads (8 warps) for cooperative loading
- Shared memory: K×4 + 96 bytes (normalized input + warp partial sums)

**realizar:** `KernelType::FusedRmsNormGateUpSwigluQ4K` in `src/cuda/kernels.rs`
- PTX name: `fused_rmsnorm_gate_up_swiglu_q4k`
- Executor: `fused_ffn_rmsnorm_swiglu_q4k_into()`, `fused_ffn_rmsnorm_swiglu_q4k_cached()`

### Memory Savings (per FFN layer)

| Metric | Before | After |
|--------|--------|-------|
| Kernel launches | 4 | 1 |
| Global writes | K + 3N floats | N floats |

---

## 6. Q8 KV Cache Dequantization Kernel (QWEN-007)

**Purpose:** INT8 KV cache for 3.56x memory reduction.

**PTX:** `Q8Dequant` kernel type in `src/cuda/kernels.rs`
```
output[i] = quants[i] * scales[i / 32]
```

**Pipeline:**
1. `init_kv_cache_q8_gpu()` — allocate Q8 buffers
2. `write_kv_q8()` / `read_kv_q8()` — CPU roundtrip
3. `dequantize_kv_q8_gpu()` — GPU dequantization (strided memory)
4. `incremental_attention_q8_gpu()` — quantize K/V → append → dequant → attention

---

## 7. Existing realizar GPU Kernels (Reference)

| Kernel | Purpose | File |
|--------|---------|------|
| `GemmKernel` | Matrix multiplication (naive, tiled, tensor core) | `src/cuda/kernels.rs` |
| `AttentionKernel` | FlashAttention-style tiled attention | `src/cuda/kernels.rs` |
| `SoftmaxKernel` | Numerically stable with warp shuffle | `src/cuda/kernels.rs` |
| `LayerNormKernel` | Fused layer normalization | `src/cuda/kernels.rs` |
| `QuantizeKernel` | Q4_K dequantization fused with matmul | `src/cuda/kernels.rs` |
| `Q5KKernel` | Q5_K dequantization | `src/cuda/kernels.rs` |
| `Q6KKernel` | Q6_K dequantization | `src/cuda/kernels.rs` |
| `CoalescedGemv` | M=1 GEMV with coalesced access | `src/cuda/kernels.rs` |

---

## References

- [Volkov10] Volkov, V. (2010). "Better Performance at Lower Occupancy." GTC 2010.
- [Shazeer20] Shazeer, N. (2020). "GLU Variants Improve Transformer." arXiv:2002.05202.
- [Dao22] Dao, T., et al. (2022). "FlashAttention." NeurIPS 2022.
- [NVIDIA23] NVIDIA Corporation. (2023). "CUDA C++ Programming Guide v12.3."
