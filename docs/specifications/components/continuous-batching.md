# Component: Continuous Batching (PMAT-044 / PMAT-088)

**Parent:** [gpu-performance-spec.md](../gpu-performance-spec.md) §5b
**Status:** Active — Phase 1 complete, Phase 2 is critical path
**Test target:** ssh yoga, forjar setup/teardown, isolated serial benchmarks

---

## Goal

Serve M=2..32 concurrent `/v1/chat/completions` requests with true weight
sharing, achieving aggregate throughput proportional to batch size.

| Concurrency | Target (aggregate tok/s) | Current (Mar 11) | Status |
|-------------|-------------------------|-------------------|--------|
| c=1 | Baseline (single-request optimized path) | 152.8 tok/s | PASS |
| c=4 | >= 3.0x baseline (≥460 tok/s, 74% eff) | 256.9 tok/s (42.0% eff) | GAP |
| c=8 | >= 5.0x baseline | Not measured | Pending |

---

## Architecture

### Current: Iteration Scheduler (PMAT-088a, Mar 11)

```
HTTP request -> cuda_chat_backend
  -> stream: true? -> cuda_batch_tx.send(CudaBatchRequest)
  -> Iteration Scheduler (ITERATION_SCHEDULER=1):
      waiting_queue: VecDeque<CudaBatchRequest>  (new arrivals)
      rx channel:    mpsc::Receiver               (HTTP handler sends)

      Loop:
        1. Block on rx.recv() if idle
        2. Drain rx into waiting_queue
        3. Form batch from waiting_queue (up to max_slots)
        4. M=1 fast path: generate_gpu_resident_streaming (CUDA graph)
        5. M>1: batched_setup_and_prefill → decode loop:
           - Each iteration: check waiting_queue THEN rx for mid-batch joins
           - Slot recycling from waiting_queue THEN rx
           - batched_decode_step → distribute_tokens
```

### Batched Decode Pipeline (PMAT-072/073/074)

```
1. Prefill (sequential per slot):
   for each slot:
     reset single KV cache
     forward_gpu_resident(token, cache, pos) x seq_len  [graph path]
     scatter_single_kv_to_batched(slot, seq_len)         [D2D copy]

2. Decode (batched, all slots):
   loop:
     embed M tokens
     forward_batched_to_token_ids(embeds, positions)
       for each layer:
         batched RMSNorm -> batched QKV GEMV -> batched RoPE
         batched attention (KV scatter + read) -> output proj
         -> residual -> batched FFN -> residual
     distribute_tokens (streaming callbacks, position advance)

     # Mid-batch scheduling (PMAT-073/074/088):
     - Join new requests from waiting_queue/rx into empty slots
     - Recycle finished slots with pending requests
```

### KV Cache Layout

All KV caches use head-first layout: `[num_kv_heads, max_len, head_dim]`

Batched cache: `[M, num_kv_heads, max_len, head_dim]`
- Stride per slot: `num_kv_heads x max_len x head_dim`
- Scatter writes per-head (not contiguous) due to max_len padding between heads

---

## Phase 1 Results (PMAT-088a — Iteration Scheduler)

| Metric | Baseline (batch sched) | PMAT-088a (iter sched) | Delta |
|--------|----------------------|----------------------|-------|
| c=1 decode tok/s | 154.8 | 152.8 | -1.3% |
| c=4 aggregate tok/s | 210.8 | **256.9** | **+21.9%** |
| c=4 decode/slot tok/s | 52.2 | 66.8 | +28.0% |
| c=4 TTFT P50 | 128.5ms | **80.6ms** | **-37.3%** |
| c=4 ITL P50 | 19.2ms | 15.0ms | -21.9% |
| c=4 scaling efficiency | 34.1% | 42.0% | +8pp |

**Root cause of remaining gap:** Batched GEMV attention scales O(M×seq_len) — at M=4 each
slot reads 4x more KV entries. Weight GEMV is amortized (828 MB read once for M slots), but
attention is not. FlashAttention-2 batched (Phase 2) is the critical path.

---

## Remaining Phases

| Phase | PMAT | Status | Expected Impact |
|-------|------|--------|----------------|
| **P2: Batched FlashAttention** | PMAT-088b | Next | Reduce 2.55x M=4 penalty to ~1.5x |
| P3: Chunked prefill | PMAT-088c | Planned | TTFT at c=4 ≈ c=1 |
| P4: Paged KV cache | PMAT-088d | Planned | <4% memory waste, enable c>4 |

---

## Falsification Results

| ID | Result | Detail |
|----|--------|--------|
| H-CB4 | **FALSIFIED** | M=1 per-slot ITL 2.31x c=1 (15.0/6.5ms). Sequential M=1 worse than batched. |
| H-CB7 | **FALSIFIED** | 42.0% < 60% threshold. Weight BW amortization essential. |
| H-CB8 | **CONFIRMED** | +21.9% aggregate from waiting queue integration. |

---

## Forjar Templates

```yaml
# forjar-yoga-realizr.yaml — deploy realizr only (port 8081)
# forjar-yoga-teardown.yaml — stop all services
# Env: ITERATION_SCHEDULER=1 SKIP_PARITY_GATE=1
```

Benchmark command:
```bash
probador llm load \
  --url http://192.168.50.38:8081 \
  --duration 60 \
  --concurrency ${C} \
  --warmup 5 \
  --stream true
```

---

## Pass Criteria

1. **Correctness:** c=4 produces coherent output (PASS — 141/141 successful)
2. **Throughput:** c=4 aggregate >= 3x c=1 (FAIL — 1.68x, need 3.0x)
3. **No regression:** c=1 through iteration scheduler matches baseline (PASS — 152.8 vs 154.8)
4. **Stability:** 60-second load test at c=4 with zero errors (PASS — 0 failures)
