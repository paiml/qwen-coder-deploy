# Component: GGUF Decode (Single Request)

**Parent:** [perf-parity-spec.md](../perf-parity-spec.md)
**Status:** Active — decode parity achieved on all platforms (0.92-0.96x llama.cpp), TTFT 1.33x on yoga
**Test target:** ssh yoga, forjar setup/teardown, one runtime at a time

---

## Goal

Single-request GGUF Q4_K_M decode throughput at parity with llama.cpp.

Decode is memory-bandwidth bound: read all weights once per token,
compute one output token. Theoretical minimum = model_size / bandwidth.

---

## Theory

Qwen2.5-Coder-1.5B Q4_K_M:
- Model size: ~850MB (Q4K weights)
- RTX 4060 Laptop bandwidth: ~256 GB/s (GDDR6)
- Theoretical max: 256000 / 850 ~ 301 tok/s
- Practical target (60% utilization): ~180 tok/s

For comparison, Jetson Orin Nano Super (MAXN_SUPER 1020MHz):
- Bandwidth: 67 GB/s -> theoretical ~79 tok/s -> achieved 40.8 tok/s (52% util)

---

## Decode Pipeline

```
Token embedding (CPU, single lookup)
  -> Upload to GPU (1536 floats = 6KB)
  -> CUDA Graph replay:
      28 layers x {
        RMSNorm -> Q/K/V GEMV (Q4K DP4A) -> RoPE (NEOX)
        -> KV cache write -> Incremental attention -> Output proj GEMV
        -> Residual -> RMSNorm -> Gate/Up GEMV -> SwiGLU -> Down GEMV -> Residual
      }
      -> Output RMSNorm -> LM head GEMV -> GPU argmax
  -> Download token ID (4 bytes)
```

Key optimizations:
- **CUDA graph capture**: 280 kernel launches -> 1 graph replay (~10us)
- **DP4A Q4K GEMV**: integer dot product, half-warp coalesced
- **GPU-resident KV cache**: no PCIe round-trips for K/V
- **Flash Decoding**: split-K attention for seq_len > 128
- **GPU argmax**: 4 bytes download vs 600KB logits transfer

---

## Jetson Baselines (2026-03-12, MAXN_SUPER 1020MHz, c=1, streaming)

| Runtime | Decode tok/s | TTFT P50 | ITL P50 | Prefill tok/s |
|---------|-------------|----------|---------|---------------|
| **realizr** | **40.8** | 47.8ms | **24.5ms** | 481 |
| llama.cpp | 36.1 | **34.0ms** | 27.7ms | **676** |
| **Gap** | **0.88x (realizr 13% FASTER)** | 1.4x | 0.88x | 0.71x |

---

## Yoga Baselines (2026-03-12, c=1, 60s, streaming, locked 1900MHz)

| Runtime | Decode tok/s | TTFT P50 (ms) | ITL P50 (ms) | Prefill tok/s |
|---------|-------------|---------------|--------------|---------------|
| vLLM | **168.3** | 11.4 | **5.9** | 2,016 |
| llama.cpp | 160.7 | **10.1** | 6.2 | **2,280** |
| ollama | 163.5 | 70.5 | 6.1 | 326 |
| realizr | 154.8 | 13.4 | 6.5 | 1,718 |
| **Gap (r/l)** | **0.96x (PASS)** | **1.33x (PASS)** | **1.05x (PASS)** | **0.75x** |

---

## Forjar Test Sequence

```bash
# 1. Deploy realizr only
forjar apply -f forjar-yoga-realizr.yaml --yes
probador llm load --url http://yoga:8081 --model qwen \
  --duration 60 --concurrency 1 --warmup 5 --stream true
forjar apply -f forjar-yoga-teardown.yaml --yes

# 2. Deploy llama.cpp only
forjar apply -f forjar-yoga-llamacpp.yaml --yes
probador llm load --url http://yoga:8083 --model qwen2.5-coder:1.5b-instruct \
  --duration 60 --concurrency 1 --warmup 5 --stream true
forjar apply -f forjar-yoga-teardown.yaml --yes
```

---

## Pass Criteria

- realizr decode tok/s >= llama.cpp decode tok/s
- realizr ITL P50 <= 1.5x llama.cpp ITL P50
- realizr TTFT P50 <= 2x llama.cpp TTFT P50 (prefill gap acceptable)

---

## Known Bottlenecks

### Prefill (TTFT)

1.33x gap on yoga (13.4ms vs 10.1ms): realizr reads FP8 (1 B/elem),
llama.cpp reads Q4K directly (0.5625 B/elem). 1.4x gap on Jetson
(47.8ms vs 34.0ms): HGEMM on-demand FP16 caching (warmup OOMs on 8GB).
See [gguf-prefill.md](gguf-prefill.md).

### Decode instruction count

ncu profiling (4090): Q4K GEMV is 72% compute-bound due to dequant
instruction count. llama.cpp uses ~5 bitmask ops for scale extraction
vs realizr ~20 shr+and+selp. Reducing instruction count shifts to
memory-bound (the goal for GEMV).
