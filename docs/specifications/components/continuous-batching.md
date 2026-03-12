# Component: Continuous Batching (PMAT-044 / PMAT-088)

**Parent:** [gpu-performance-spec.md](../gpu-performance-spec.md) §5b
**Status:** Active — Phase 1 complete (corrected), Phase 2 is critical path
**Test target:** ssh yoga, forjar setup/teardown, isolated serial benchmarks

---

## Goal

Serve M=2..32 concurrent `/v1/chat/completions` requests with true weight
sharing, achieving aggregate throughput proportional to batch size.

| Concurrency | Target (aggregate tok/s) | Current (Mar 11, PMAT-088b) | Theoretical Ceiling | Status |
|-------------|-------------------------|-------------------------------|--------------------|---------|
| c=1 | Baseline (single-request) | 151.6 tok/s | 151.6 | PASS |
| c=4 | >= 3.0x baseline (>=455) | 234.0 tok/s (1.54x, 76% of ceiling) | 306 tok/s | GAP |
| c=8 | >= 5.0x baseline (>=758) | **306.5 tok/s** (2.02x, 87% of ceiling) | 352 tok/s | GAP |
| c=∞ | — | — | **412 tok/s** (DP4A limit) | Ceiling |

**DP4A GEMV aggregate ceiling = 412 tok/s** (1/compute_per_token). Batched Q4K GEMV
reads weights once for M tokens but runs M independent DP4A chains → compute scales
linearly with M. The 3.0x target at c=4 (455 tok/s) **exceeds the theoretical DP4A ceiling
(306 tok/s)** and cannot be achieved without changing the matmul kernel architecture.
Reaching 3.0x requires W4A16 tensor core GEMM (like vLLM AWQ: Q4 storage + FP16 compute).

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
        5. M>1: batched_setup_and_prefill -> decode loop:
           - Each iteration: check waiting_queue THEN rx for mid-batch joins
           - Slot recycling from waiting_queue THEN rx
           - batched_decode_step -> distribute_tokens
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

### Initial measurement (pre-bugfix)

| Metric | Baseline (batch sched) | PMAT-088a initial | Delta |
|--------|----------------------|-------------------|-------|
| c=1 decode tok/s | 154.8 | 152.8 | -1.3% |
| c=4 aggregate tok/s | 210.8 | 256.9 | +21.9% |
| c=4 TTFT P50 | 128.5ms | 80.6ms | -37.3% |
| c=4 ITL P50 | 19.2ms | 15.0ms | -21.9% |
| c=4 scaling efficiency | 34.1% | 42.0% | +8pp |

### Post-bugfix (final, variable-M buffer fixes)

The initial 256.9 included inflated counts from error-retry cycles caused by three
classes of buffer length mismatch when M changes between iterations.

| Metric | Baseline (batch sched) | PMAT-088 final | Delta |
|--------|----------------------|----------------|-------|
| c=1 decode tok/s | 154.8 | **151.8** | -1.9% (noise) |
| c=4 aggregate tok/s | 210.8 | **232.7** | **+10.4%** |
| c=4 decode/slot tok/s | 52.2 | 66.1 | +26.6% |
| c=4 TTFT P50 | 128.5ms | **81.6ms** | **-36.5%** |
| c=4 ITL P50 | 19.2ms | 15.1ms | -21.4% |
| c=4 scaling efficiency | 34.1% | **38.3%** | +4.2pp |

### Variable-M Buffer Bugs Fixed

1. **Logits buffer** (M*vocab vs vocab): `prepare_capture_buffers` checked `is_none()` not
   size. Fix: check `b.len() != vocab_size`, `clear_decode_graph()` on realloc.
2. **Input/hidden buffer** (M*hidden vs prev M*hidden): Grow-only allocation (`<`) kept
   M=4 capacity. Fix: exact-size reallocation (`!=`), logical size for hidden_buf2.
3. **KV ptr/seq_lens** (M vs max_M): High-water-mark buffers. Fix: `copy_from_host_at(0)`
   for sync copies, `from_raw_parts` exact-M views for async copies.

**Root cause of remaining gap (CORRECTED, PMAT-088b):** ~~Attention KV scaling~~ is NOT
the bottleneck. Attention reads 14 MB = 2.8% of weight BW (491 MB). The actual bottleneck
is **GEMV compute scaling**: batched Q4K GEMV reads weights once for M=4 but runs 4x
independent DP4A accumulation chains, transitioning from memory-bound (M=1) to
compute-bound (M=4).

Profiled M=4 decode step: 13.3ms = 2.56ms BW + 9.7ms compute + 1ms launches.
Phase 2 redesigned: HGEMM crossover at M>1 + CUDA graph for M>1.

---

## Remaining Phases

| Phase | PMAT | Status | Expected Impact |
|-------|------|--------|----------------|
| **P2: M>1 CUDA graph** | PMAT-088b | **CLOSED** — HGEMM falsified (H-CB9), graph falsified (H-CB11) | Graph 3ms SLOWER than eager |
| P3: Chunked prefill | PMAT-088c | Planned | TTFT at c=4 = c=1 |
| P4: Paged KV cache | PMAT-088d | Planned | <4% memory waste, enable c>4 |

---

## Falsification Results

| ID | Result | Detail |
|----|--------|--------|
| H-CB4 | **FALSIFIED** | M=1 per-slot ITL 2.29x c=1 (15.1/6.6ms). Sequential M=1 worse than batched. |
| H-CB7 | **FALSIFIED** | 38.3% < 60% threshold. Weight BW amortization essential. |
| H-CB8 | **CONFIRMED** | +10.4% aggregate from waiting queue integration (initial +21.9% inflated by retries). |
| H-CB9 | **FALSIFIED** | Three variants tested at 1900 MHz: full HGEMM 256.0, hybrid 260.5, DP4A 261.5 aggregate. FP16 3.5x BW penalty not compensated by tensor cores at M=4. |
| H-CB10 | **CONFIRMED** | Attention is 2.8% of BW — GEMV compute (DP4A) is the actual bottleneck. |
| H-CB11 | **FALSIFIED** | Batched CUDA graph 3ms SLOWER than eager (18.1ms vs 15.1ms ITL). 654 launches/step = 2.6ms overhead, but attention grid dims frozen at capture (dummy seq_lens=1). |

---

## Forjar Templates

```yaml
# forjar-yoga-realizr.yaml -- deploy realizr only (port 8081)
# forjar-yoga-teardown.yaml -- stop all services
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

1. **Correctness:** c=4 and c=8 produce coherent output (PASS -- zero errors at c=4, 0 errors at c=8)
2. **Throughput:** c=4 aggregate >= 3x c=1 (**INFEASIBLE** -- 1.54x, DP4A ceiling = 2.02x at c=4)
3. **No regression:** c=1 through iteration scheduler matches baseline (PASS -- 151.6 vs 154.8)
4. **Stability:** 60-second load test at c=8 with zero errors (PASS -- 0 failures)
5. **Efficiency:** Aggregate ≥ 85% of theoretical DP4A ceiling (c=8: 87% PASS, c=4: 76% GAP)

**Note (PMAT-088b):** The 3.0x target at c=4 requires 455 tok/s, but the theoretical DP4A
ceiling at M=4 is 306 tok/s. Reaching 3.0x requires a different matmul kernel architecture
(W4A16 tensor core GEMM). Revised target: maximize aggregate % of DP4A ceiling.
