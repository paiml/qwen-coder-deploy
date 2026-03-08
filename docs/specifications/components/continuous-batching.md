# Component: Continuous Batching (PMAT-044)

**Parent:** [perf-parity-spec.md](../perf-parity-spec.md)
**Status:** Active — correctness bug blocking benchmarks
**Test target:** ssh yoga, forjar setup/teardown, one runtime at a time

---

## Goal

Serve M=2..32 concurrent `/v1/chat/completions` requests with true weight
sharing, achieving aggregate throughput proportional to batch size.

| Concurrency | Target (aggregate tok/s) | Mechanism |
|-------------|-------------------------|-----------|
| c=1 | Baseline (single-request optimized path) | CUDA graph decode |
| c=2 | >= 1.8x baseline | Batched GEMV |
| c=4 | >= 3.0x baseline | Batched GEMV |
| c=8 | >= 5.0x baseline | Batched GEMV |

---

## Architecture

### Batch Scheduler

```
HTTP request -> cuda_chat_backend
  -> stream: true? -> batch_tx.send(CudaBatchRequest)
  -> Batch scheduler (10ms accumulation window):
      M=1 -> generate_gpu_resident_streaming (optimized single path)
      M>1 -> generate_batched_streaming (batched GEMV path)
```

### Batched Decode Pipeline

```
1. Prefill (sequential per slot):
   for each slot:
     reset single KV cache
     forward_gpu_resident(token, cache, pos) x seq_len  [graph path]
     scatter_single_kv_to_batched(slot, seq_len)         [D2D copy]

2. Decode (batched, all slots):
   loop:
     embed M tokens (CPU)
     forward_batched_to_token_ids(embeds, positions)
       for each layer:
         transformer_layer_batched(input, layer, weights, M, positions)
           Phase 1: batched RMSNorm -> batched QKV GEMV -> batched RoPE
           Phase 2: batched attention (scatter K/V + read) -> output proj
                    -> residual -> batched FFN -> residual
     distribute_tokens (streaming callbacks, position advance)
```

### KV Cache Layout

All KV caches use head-first layout: `[num_kv_heads, max_len, head_dim]`

Batched cache: `[M, num_kv_heads, max_len, head_dim]`
- Stride per slot: `num_kv_heads x max_len x head_dim`
- Scatter writes per-head (not contiguous) due to max_len padding between heads

---

## Current Bug: Batched Decode Produces Zero Tokens

### Symptom (yoga, 2026-03-08)

c=4 completes prefill (96.4 req/s, 2430 prefill tok/s) but generates
**0 decode tokens** per request. All 5788 requests return empty output.
Previous testing on Jetson showed "The The The..." repetition at c=2.
c=1 through batch scheduler works correctly (M=1 uses single-request path).

### Bugs Found and Fixed

1. **Contiguous scatter** — head-first layout needs per-head copy (fixed: `99a9585`)
2. **cuBLAS prefill incompatibility** — switched to serial forward_gpu_resident (fixed: `b00cf6d`)
3. **batched_kv_stride visibility** — made pub(crate) for stride toggling (fixed: `cf9443d`)

### Diagnostic Plan

1. **M=1 through batched path** — modify batch scheduler to route M=1 through
   `generate_batched_streaming`. If correct output, bug is in multi-slot.
   If wrong, bug is in the batched code path itself.

2. **KV readback after scatter** — after scatter, read back batched KV cache
   values for slot 0, layer 0, head 0, position 0 and compare with single
   KV cache values.

3. **Attention input/output dump** — in first decode step, dump Q/K/V vectors
   entering `batched_incremental_attention_into` for layer 0 and compare with
   what single-token path produces for the same input.

---

## Forjar Templates

```yaml
# forjar-yoga-realizr.yaml — deploy realizr only (port 8081)
# forjar-yoga-teardown.yaml — stop all services
```

Benchmark command:
```bash
probador llm load \
  --url http://yoga:8081 \
  --model qwen \
  --duration 60 \
  --concurrency ${C} \
  --warmup 5 \
  --stream true
```

---

## Pass Criteria

1. **Correctness:** c=2 and c=4 produce coherent, non-repetitive output
2. **Throughput:** c=4 aggregate >= 3x c=1 single-request decode tok/s
3. **No regression:** c=1 through batch scheduler matches c=1 direct path
4. **Stability:** 60-second load test at c=4 completes with zero errors

---

## Test Matrix

| Test | c | Pass | Notes |
|------|---|------|-------|
| Baseline decode | 1 | PASS | 138.7 tok/s on yoga |
| Batched correctness | 2 | FAIL | "The" bug (Jetson), 0 tokens (yoga) |
| Batched throughput | 4 | FAIL | 0 decode tokens, prefill works |
| Batched stress | 8 | BLOCKED | Needs c=4 fix first |
| llama.cpp c=4 ref | 4 | PASS | 74.0 decode tok/s, 2.3 req/s |
