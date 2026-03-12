# Component: Continuous Batching (PMAT-044 / PMAT-088)

**Parent:** [gpu-performance-spec.md](../gpu-performance-spec.md) §5b
**Status:** Active — Phase 1 done, Phase 2 CLOSED (falsified), Phase 3 is critical path
**Test target:** ssh yoga, forjar setup/teardown, isolated serial benchmarks

---

## Goal

Serve M=2..32 concurrent `/v1/chat/completions` requests with true weight
sharing, achieving aggregate throughput proportional to batch size.

| Concurrency | Target (aggregate tok/s) | Current (Mar 12, PMAT-088c recycle) | Theoretical Ceiling | Status |
|-------------|-------------------------|-------------------------------|--------------------|---------|
| c=1 | Baseline (single-request) | 153.5 tok/s | 153.5 | PASS |
| c=4 | >= 3.0x baseline (>=455) | **256.3 tok/s** (1.67x, 84% of ceiling) | 306 tok/s | GAP |
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
4. **Argmax results buffer** (M vs prev-M): `batched_argmax_results` grow-only allocation
   kept M=4 capacity; `copy_to_host` in `batched_gpu_argmax` (par-062.rs) compared host=M
   vs device=prev_M. Fix: `from_raw_parts(ptr, m)` exact-M view before D2H copy.
   **Root cause of persistent c=4 failure** — reduces.rs was an orphan file; actual code
   was in par-062.rs (included from forward_workspace_captured.rs).

**Root cause of remaining gap (CORRECTED, PMAT-088b):** ~~Attention KV scaling~~ is NOT
the bottleneck. Attention reads 14 MB = 2.8% of weight BW (491 MB). The actual bottleneck
is **GEMV compute scaling**: batched Q4K GEMV reads weights once for M=4 but runs 4x
independent DP4A accumulation chains, transitioning from memory-bound (M=1) to
compute-bound (M=4).

Profiled M=4 decode step: 13.3ms = 2.56ms BW + 8.1ms compute + 2.6ms launches (654 kernels).
Phase 2 CLOSED: HGEMM (H-CB9) and CUDA graph (H-CB11) both FALSIFIED.

### Continuous Batch Recycling (PMAT-088c — Mar 12)

Three fixes to enable true continuous batching without batch restarts:

1. **Continuous loop**: Keep decode loop alive when all slots finish but pending
   requests exist (prevents 20ms+ `batched_setup_and_prefill` restart overhead per cycle)
2. **Recycle-first priority**: Try `recycle_slot` before `add_slot_to_batch` when done
   slots exist (prevents M growing to `max_kv_slots` with wasted done-slot GPU cycles)
3. **Channel-close on done**: Drop callback + error sender for finished slots so SSE
   handler receives channel closure → sends `[DONE]` → probador reconnects → recycling
   gets new requests (was blocking all recycling because SSE never ended)

| Metric | Before (batch restart) | After (recycle) | Delta |
|--------|----------------------|-----------------|-------|
| c=4 aggregate tok/s | 207.4 | **256.3** | **+24%** |
| c=4 TTFT P50 | 116.5ms | **63.6ms** | **-45%** |
| c=4 TTFT P99.9 | 2866ms | **112ms** | **-96%** |
| c=4 ITL P50 | 15.1ms | 15.2ms | Same |
| c=4 ITL CV | 0.22 | **0.01** | Stable |
| Batch restarts / 60s | 45 | **3** (warmup only) | -93% |
| Slot recycles / 60s | 0 | **124** | ∞ |
| c=4 scaling efficiency | 34.1% | **41.8%** | +7.7pp |

---

## Remaining Phases

| Phase | PMAT | Status | Expected Impact |
|-------|------|--------|----------------|
| **P2: M>1 CUDA graph** | PMAT-088b | **CLOSED** — HGEMM falsified (H-CB9), graph falsified (H-CB11) | Graph 3ms SLOWER than eager |
| **P3: Chunked prefill** | PMAT-088c | **In Progress** | TTFT at c=4 ≈ c=1, no decode stalls |
| P4: Paged KV cache | PMAT-088d | Planned | <4% memory waste, enable c>4 |

---

## Phase 3: Chunked Prefill (PMAT-088c)

### Problem

Current prefill is monolithic: `batched_setup_and_prefill` processes all S prompt tokens
per slot sequentially, blocking the decode loop for 14-82ms per new slot join. At c=4,
this produces 14% scheduling overhead (82ms TTFT × 4 = 328ms stall per round) vs
llama.cpp's 5% (18.6ms TTFT × 4 = 74ms). The decode ITL spike during prefill degrades
user experience for in-flight requests.

### Architecture: Interleaved Chunk-Decode

```
Current (monolithic prefill — stalls decode):
  [========= PREFILL slot 0 (82ms) =========][=== PREFILL slot 1 (82ms) ===]
                                                                             [DECODE step 1]

Chunked prefill (interleaved — no decode stalls):
  [CHUNK 0a (4ms)][DECODE 1 (13ms)][CHUNK 0b (4ms)][DECODE 2 (13ms)]...
  └─ slot 0 tokens 0-255            └─ slot 0 tokens 256-511
  All M existing slots decode every ~17ms instead of waiting 164ms+
```

### Implementation Plan

**Step 1: `prefill_chunk()` method** (realizr: `prefill.rs`)
- New method: `prefill_chunk(embeddings_slice, positions_slice, chunk_start, chunk_end)`
- Processes tokens [chunk_start..chunk_end) through all 28 layers
- Writes K/V to single-request cache at positions [chunk_start..chunk_end)
- Returns hidden state of last position (for first-token extraction on final chunk)
- Chunk size: `PREFILL_CHUNK_SIZE` env var, default 256 tokens (~4ms on RTX 4060L)

**Step 2: `ChunkedPrefillState` tracker** (realizr: `generate_batched_streaming.rs`)
```
struct ChunkedPrefillState {
    slot_idx: usize,
    prompt_tokens: Vec<u32>,
    tokens_prefilled: usize,      // How many tokens in KV cache so far
    chunk_size: usize,            // Default 256
    single_kv_cache: allocated,   // Gradually filled
    on_token: callback,           // SSE callback for first decode token
}
```

**Step 3: Iteration scheduler integration** (realizr: `iteration_scheduler.rs`)
```
Loop:
  1. Drain rx → waiting_queue
  2. If pending_prefill.is_some():
     a. Execute ONE chunk of pending prefill (~4ms)
     b. If chunk is final: scatter to batched cache, add slot to decode batch
  3. batched_decode_step for all M active decode slots (~13ms)
  4. distribute_tokens
  5. If waiting_queue.not_empty() && active_slots < max_slots:
     a. Start new ChunkedPrefillState from next waiting request
```

**Step 4: Buffer management**
- Workspace sized for `max(chunk_size, M)` at init — no mid-chunk realloc
- Single KV cache reused between slots (already exists for M=1 path)
- Batched KV cache scatter only on final chunk (unchanged from current)

### Constraints

- **No CUDA graph during chunked prefill**: Eager path only (PREFILL_GRAPH already disabled)
- **One prefill at a time**: Only one slot can be in chunked-prefill state
  (single KV cache shared). Multiple concurrent prefills need paged KV (Phase 4).
- **Decode graph invalidation**: When M changes (new slot joins), clear batched decode graph
  (already handled by `clear_decode_graph` on M change)

### Expected Impact

| Metric | Current (PMAT-088b) | Expected (PMAT-088c) | Delta |
|--------|---------------------|----------------------|-------|
| c=4 TTFT P50 | 82.1ms | ~18ms (chunk + 1 decode) | -78% |
| c=4 ITL P99 (during join) | ~100ms (blocked) | ~17ms (chunk + decode) | -83% |
| c=4 aggregate tok/s | 234.0 | ~250 (+7% from less stall) | +7% |
| Scheduling overhead | 14% | ~5% (matching llama.cpp) | -9pp |

### Falsification Condition

| ID | Hypothesis | Prediction |
|----|-----------|------------|
| H-CB12 | Chunked prefill eliminates decode stalls | c=4 ITL P99 ≤ 1.5x c=1 ITL P99 during prefill |
| H-CB13 | Chunked prefill matches llama.cpp TTFT | c=4 TTFT ≤ 2x llama.cpp TTFT (37ms) |

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
