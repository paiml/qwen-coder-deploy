# qwen-coder-deploy

<p align="center">
  <img src="docs/assets/architecture.svg" alt="Architecture" width="720"/>
</p>

Deploy and benchmark Qwen2.5-Coder-1.5B-Instruct across five inference runtimes. Infrastructure via [forjar](https://github.com/paiml/forjar). Scoring via [probador](https://github.com/paiml/probador). 284 provable contracts enforced by build.rs.

## Quick Start

```bash
make bench-yoga-prod         # All 4 runtimes, production methodology
make score-prod              # Production scorecards
make teardown-yoga           # Stop all services
```

## Runtimes

| Runtime | Port | Format | Quantization | GPU |
|---------|------|--------|-------------|-----|
| [realizar](https://github.com/paiml/realizar) | 8081 | GGUF | Q4_K_M (DP4A + FP8) | CUDA |
| ollama | 8082 | GGUF | Q4_K_M | CUDA |
| llama.cpp | 8083 | GGUF | Q4_K_M | CUDA |
| vLLM | 8084 | AWQ | INT4 (CUTLASS) | CUDA |
| **realizar-wgpu** | 8081 | GGUF | Q4_K_M→F32 (or Q4K fused) | **WGPU/Vulkan** |

## Performance

RTX 4060 Laptop, 1900MHz locked, production methodology (medium prompt, uniform output, streaming, 60s).

### Throughput (tok/s aggregate)

**CUDA — Yoga RTX 4060L (Mar 26, PMAT-370):**

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 137 | 160 | 154 | 163 |
| 4 | 320 | 351 | 598 | 635 |
| 8 | **534** | 419 | 1,142 | -- |
| 16 | **950** | 912 | 2,037 | -- |
| 32 | **1,621** | 1,949 | 2,998 | -- |

**WGPU — Intel Radeon Pro W5700X (Mar 26, PMAT-375/377):**

| Model | F32 | Q4K | VRAM (Q4K) |
|-------|-----|-----|------------|
| 1.5B | 0.74 tok/s | 0.46 tok/s | 626 MB |
| 3B | -- | 0.31 tok/s | 2,907 MB |
| 7B | -- | 0.07 tok/s | 3,906 MB |

**Blackwell GB10 — Grace ARM + sm_121, 120 GB unified (Mar 27, PMAT-390/394):**

| Model | c=1 | c=8 | c=32 | HumanEval |
|-------|-----|-----|------|-----------|
| 1.5B | 92 tok/s | 413 | 851 | -- |
| 7B | 29 tok/s | 154 | 197 | 84.76% |
| **32B** | **7.5 tok/s** | -- | -- | -- (zero-copy `cuMemHostRegister`) |

### Quality Scores

| c | realizr | llama.cpp | vLLM | ollama |
|---|---------|-----------|------|--------|
| 1 | 94 A | 93 A | 98 A+ | 77 B |
| 8 | **76 B** | 66 C+ | 97 A+ | 57 C |
| 32 | **75 B** | 63 C+ | 87 A- | 57 C |
| 128 | **66 C+** | -- | 64 C+ | -- |

realizr overtakes llama.cpp at c=8 (+25% aggregate) and beats vLLM on quality at c=128.

### Asymptotes

| Runtime | Peak | Architecture |
|---------|------|-------------|
| vLLM | 3,163 tok/s | PagedAttention + continuous batching + CUTLASS |
| realizr | 1,511 tok/s | Iteration scheduler + BATCH=32 |
| llama.cpp | 923 tok/s | Fixed 16 slots |
| ollama | 157 tok/s | Serial |

### Cross-Platform (c=1 decode tok/s)

| Platform | realizr | llama.cpp | vLLM |
|----------|---------|-----------|------|
| RTX 4060L (24 SMs) | 149 | 159 | 154 |
| RTX 4090 (128 SMs) | 412 | 437 | -- |
| Jetson Orin (8 SMs) | 25 | -- | -- |

### CPU (Xeon W-3245, c=1 decode tok/s)

| Runtime | tok/s | Gap |
|---------|-------|-----|
| llama.cpp | 59.0 | -- |
| realizr | **32.6** | **1.81x** |

+91% from baseline (17.1), 18 approaches tested (PMAT-297-310).

## Key Results

- **GPU tensor graph dispatch (PMAT-291)**: +8.5% at c=4. Pure Rust graph executor + Q8 cache
- **CPU +91% (PMAT-297-310)**: thread pool, prefetch, hugepage, lean pointer dispatch
- **GPU kernels within 8% of vLLM** (7.4ms vs 6.8ms). 16 kernel fusion approaches falsified
- **CPU DRAM-bandwidth bound**: perf stat IPC 1.59 vs llama.cpp 1.01 — remaining gap is Rust abstraction overhead
- **Quality crossover**: realizr beats vLLM at c=128 on combined score
- **3B model format parity (PMAT-314)**: SafeTensors→Q4K 91.6 tok/s (+13% vs GGUF 80.9). Fixed sharded SafeTensors loading
- **Qwen2.5-Coder-3B-Instruct (PMAT-319)**: 6/6 correctness, 81.9 tok/s. Single-user quality mode

Full analysis: [gpu-performance-spec.md](docs/specifications/gpu-performance-spec.md) (v5.45.0, 319 PMAT items) | [performance.md](performance.md)

## Infrastructure

| File | Purpose |
|------|---------|
| `forjar-yoga-*.yaml` | Yoga deployment configs (RTX 4060L) |
| `forjar-gpu.yaml` | 4090 profiling deployment |
| `forjar.yaml` | CPU deployment (intel host) |
| `prompts/correctness.yaml` | 6-prompt correctness suite |
| `scripts/nightly.sh` | Automated benchmark pipeline |
| `docs/specifications/gpu-performance-spec.md` | Performance spec v5.40.0 |
| `docs/specifications/scoring.yaml` | Scoring contract v2.0.0 |

## Testing

```bash
make test          # Correctness (6/6 all runtimes)
make load          # Load tests
make score-gate    # CI quality gate (fail if any runtime below C)
```

Results in `results/`, aggregated in `performance.md`.

<!-- PERFORMANCE_START -->
## Performance Results

| Date | Runtime | Concurrency | RPS | P50 (ms) | P95 (ms) | P99 (ms) | TTFT P50 (ms) | Tok/s | Avg tok/req | ITL P50 (ms) | Decode tok/s | Prefill tok/s | TPOT P50 (ms) | Err% | Requests |
|------|---------|-------------|-----|----------|----------|----------|---------------|-------|-------------|--------------|--------------|---------------|---------------|------|----------|
| 2026-03-01 | realizar-apr | 4 | 0.4 | 12807.2 | 12950.4 | 12963.4 | 12807.2 | 6.9 | 0.0 | - | - | - | - | 0% | 13 |
| 2026-03-01 | realizar-gguf-1 | 4 | 1.5 | 2586.0 | 4155.6 | 4179.1 | 2586.0 | 1.5 | 0.0 | - | - | - | - | 0% | 45 |
| 2026-03-01 | realizar-gguf-2 | 4 | 1.5 | 2510.7 | 3839.4 | 3876.5 | 2510.6 | 1.5 | 0.0 | - | - | - | - | 0% | 45 |
| 2026-03-02 | realizar-gpu | 4 | 10.2 | 392.6 | 599.6 | 705.2 | 392.6 | 10.2 | 0.0 | - | - | - | - | 0% | 609 |
| 2026-03-02 | ollama-gpu | 4 | 120.3 | 30.8 | 48.8 | 72.0 | 30.8 | 240.5 | 0.0 | - | - | - | - | 0% | 7216 |
| 2026-03-02 | llamacpp-gpu | 4 | 328.2 | 11.4 | 15.6 | 18.5 | 11.4 | 656.4 | 0.0 | - | - | - | - | 0% | 19692 |
| 2026-03-02 | realizar-gpu | 4 | 4.5 | 743.6 | 1647.6 | 2154.1 | 743.6 | 4.5 | 0.0 | - | - | - | - | 0% | 267 |
| 2026-03-02 | ollama-gpu | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 0% | 400038 |
| 2026-03-02 | llamacpp-gpu | 4 | 97.5 | 33.1 | 91.3 | 95.1 | 33.1 | 195.1 | 0.0 | - | - | - | - | 0% | 5853 |
| 2026-03-02 | realizar-gpu | 4 | 5.1 | 608.6 | 1611.8 | 1664.8 | 608.5 | 5.1 | 0.0 | - | - | - | - | 0% | 306 |
| 2026-03-02 | ollama-gpu | 4 | 91.9 | 35.0 | 75.4 | 85.6 | 35.0 | 183.8 | 0.0 | - | - | - | - | 0% | 5514 |
| 2026-03-02 | llamacpp-gpu | 4 | 161.0 | 11.6 | 67.8 | 70.7 | 11.6 | 322.0 | 0.0 | - | - | - | - | 0% | 9660 |
| 2026-03-04 | realizar-gpu | 4 | 1.2 | 2529.5 | 5013.3 | 5794.3 | 2529.4 | 151.4 | 121.9 | 24.5 | 40.8 | - | 0.0 | 0% | 78 |
| 2026-03-04 | ollama-gpu | 4 | 4.4 | 914.7 | 951.5 | 957.8 | 914.6 | 561.6 | 128.0 | 7.1 | 139.9 | - | 0.0 | 0% | 264 |
| 2026-03-04 | llamacpp-gpu | 4 | 7.3 | 548.2 | 576.4 | 586.9 | 548.2 | 931.5 | 128.0 | 4.3 | 233.5 | - | 0.0 | 0% | 440 |
| 2026-03-04 | ollama-jetson | 1 | 0.1 | 11313.3 | 11600.5 | 11600.5 | 11313.3 | 11.3 | 128.0 | 88.4 | 11.3 | - | 0.0 | 0% | 6 |
| 2026-03-04 | llamacpp-jetson | 1 | 0.3 | 3986.4 | 3990.2 | 3991.6 | 3986.4 | 32.1 | 128.0 | 31.1 | 32.1 | - | 0.0 | 0% | 16 |
| 2026-03-04 | llamacpp-jetson-c4 | 4 | 0.5 | 7484.2 | 7505.4 | 7507.4 | 7484.2 | 68.5 | 128.0 | 58.5 | 17.1 | - | 0.0 | 0% | 36 |
| 2026-03-04 | ollama-jetson-c4 | 4 | 0.1 | 41578.9 | 42946.3 | 42946.3 | 41578.9 | 12.3 | 128.0 | 324.8 | 3.1 | - | 0.0 | 0% | 9 |
| 2026-03-04 | realizr-jetson-cpu | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 100.0% | 377086 |
| 2026-03-05 | realizar-jetson | 1 | 0.1 | 11581.4 | 11584.8 | 11584.9 | 3960.3 | 11.1 | 128.0 | 60.0 | 16.7 | 25.8 | 60.0 | 0% | 6 |
| 2026-03-05 | llamacpp-jetson | 1 | 0.3 | 3981.9 | 3986.1 | 3986.7 | 48.5 | 32.1 | 128.0 | 31.0 | 32.3 | 2101.8 | 31.0 | 0% | 16 |
| 2026-03-05 | ollama-jetson | 1 | 0.2 | 4258.4 | 4282.1 | 4283.4 | 426.4 | 30.1 | 128.0 | 30.2 | 33.2 | 239.2 | 30.2 | 0% | 15 |
| 2026-03-05 | realizar-jetson | 1 | 0.1 | 11583.0 | 11587.2 | 11587.9 | 3962.0 | 11.1 | 128.0 | 60.0 | 16.7 | 25.7 | 60.0 | 0% | 6 |
| 2026-03-05 | llamacpp-jetson | 1 | 0.3 | 3981.8 | 3982.9 | 3983.2 | 48.8 | 32.1 | 128.0 | 31.0 | 32.3 | 2090.6 | 31.0 | 5.9% | 17 |
| 2026-03-05 | realizar-jetson-nodp4a | 1 | 0.1 | 14016.7 | 14020.7 | 14021.0 | 3976.0 | 9.1 | 128.0 | 79.1 | 12.6 | 25.7 | 79.1 | 0% | 5 |
| 2026-03-05 | realizar-jetson-4warp | 1 | 0.1 | 12293.0 | 12299.4 | 12299.7 | 3963.6 | 10.4 | 128.0 | 65.6 | 15.2 | 25.7 | 65.6 | 0% | 5 |
| 2026-03-05 | realizar-jetson-2warp | 1 | 0.1 | 12721.0 | 12727.7 | 12728.8 | 3970.3 | 10.1 | 128.0 | 68.9 | 14.5 | 25.7 | 68.9 | 0% | 5 |
| 2026-03-05 | realizar-jetson-nodp4aq6k | 1 | 0.1 | 12601.1 | 12603.2 | 12603.4 | 3817.5 | 10.2 | 128.0 | 69.1 | 14.5 | 26.7 | 69.1 | 0% | 5 |
| 2026-03-05 | realizar-jetson | 1 | 0.1 | 11574.8 | 11585.2 | 11587.0 | 3956.3 | 11.1 | 128.0 | 60.0 | 16.7 | 25.8 | 60.0 | 0% | 6 |
| 2026-03-05 | llamacpp-jetson | 1 | 0.3 | 3980.9 | 3985.6 | 3991.3 | 48.8 | 32.1 | 128.0 | 31.0 | 32.3 | 2092.0 | 31.0 | 0% | 16 |
| 2026-03-05 | ollama-jetson | 1 | 0.1 | 10781.8 | 18705.0 | 20272.3 | 448.9 | 10.1 | 128.0 | 80.0 | 12.5 | 227.2 | 80.0 | 0% | 5 |
| 2026-03-08 | realizar-cpu | 4 | 0.1 | 37029.1 | 40395.4 | 40435.8 | 18846.8 | 13.8 | 128.0 | 143.7 | 7.0 | 5.5 | 143.7 | 0% | 8 |
| 2026-03-08 | ollama-cpu | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 100.0% | 517540 |
| 2026-03-08 | llamacpp-cpu | 4 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | - | - | - | - | 100.0% | 4 |

<!-- PERFORMANCE_END -->
