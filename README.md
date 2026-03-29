# qwen-coder-deploy

## What This Is

A head-to-head benchmark of **five different ways to run the same AI model** on your GPU. We take Qwen2.5-Coder (a code-generation model) and deploy it across five inference runtimes to answer a simple question: **which engine gives you the best throughput, latency, and quality — and at what concurrency?**

If you're running local AI and wondering whether Ollama is fast enough, or whether you should switch to llama.cpp or vLLM, this repo has the data.

## The Five Runtimes

| Runtime | What It Is | Best For |
|---------|-----------|----------|
| [realizar](https://github.com/paiml/realizar) | Rust inference engine with CUDA kernels | Single-user quality (lowest ITL) |
| [Ollama](https://ollama.com) | Popular local AI runtime | Simplicity — `ollama pull` and go |
| [llama.cpp](https://github.com/ggerganov/llama.cpp) | C++ GGUF inference, the gold standard | CPU inference, stable throughput |
| [vLLM](https://github.com/vllm-project/vllm) | Python/CUDA production serving | High-concurrency throughput (c=8+) |
| realizar-wgpu | Rust + Vulkan (AMD/Intel GPUs) | Non-NVIDIA hardware |

All five serve the **same model** (Qwen2.5-Coder-1.5B Q4_K_M) via OpenAI-compatible endpoints, so you can swap runtimes without changing your application code.

## Key Findings

### Should I ditch Ollama?

**For single-user use: Ollama is fine.** At concurrency=1, all four GPU runtimes produce nearly identical throughput (136-163 tok/s). Ollama is the easiest to set up.

**For multi-user serving: yes, switch.** Ollama is serial — it processes one request at a time. At 4 concurrent users, vLLM delivers 598 tok/s vs Ollama's 635 (Ollama batches differently). But at 8+ users, Ollama can't scale at all while vLLM reaches 3,000 tok/s. llama.cpp and realizr both scale to 1,900 tok/s.

### The Throughput Picture

RTX 4060 Laptop GPU, 1900 MHz locked, production workload (streaming, mixed output lengths, 60s runs):

| Concurrent Users | realizar | llama.cpp | vLLM | Ollama |
|-----------------|----------|-----------|------|--------|
| 1 | 136 tok/s | 160 | 154 | 163 |
| 4 | 351 | 351 | 598 | 635 |
| 8 | **610** | 419 | 1,142 | -- |
| 16 | **1,072** | 912 | 2,037 | -- |
| 32 | **1,895** | 1,949 | 2,998 | -- |

**What this means:**
- At **1 user**: pick any runtime — they're all within 20% of each other
- At **4 users**: vLLM pulls ahead (1.7x faster than llama.cpp)
- At **8+ users**: vLLM dominates on raw throughput, but realizr and llama.cpp offer better per-request quality (lower latency jitter)
- At **128 users**: realizr actually beats vLLM on quality score (66 vs 64) because vLLM's per-request latency degrades

### Larger Models on Blackwell GB10

On NVIDIA's Grace Blackwell (120 GB unified memory), we also tested 7B and 32B models:

| Model | 1 User | 32 Users | HumanEval pass@1 |
|-------|--------|----------|-----------------|
| 1.5B | 101 tok/s | 1,677 tok/s | -- |
| 7B | 31 tok/s | 472 tok/s | 84.76% |
| 32B | 8.4 tok/s | -- | **90.85%** (149/164) |

The 32B model achieves 90.85% on HumanEval — near state-of-the-art code generation quality.

## How to Replicate

### Prerequisites

- Linux with NVIDIA GPU (CUDA 12.0+)
- [forjar](https://github.com/paiml/forjar) — declarative deployment tool
- [probador](https://github.com/paiml/probador) — load testing and scoring tool
- The model: `qwen2.5-coder-1.5b-instruct-q4_k_m.gguf` from HuggingFace

### Quick Benchmark (single runtime)

```bash
# 1. Deploy realizr on your GPU
forjar apply -f forjar-yoga-realizr.yaml

# 2. Run a load test
probador llm load \
    --url http://your-gpu-host:8081 \
    --concurrency 4 \
    --duration 60 \
    --stream true

# 3. Tear down
forjar apply -f forjar-yoga-teardown.yaml
```

### Full Comparative Benchmark

```bash
# Run all 4 GPU runtimes in isolated serial mode (deploy one, bench, teardown, repeat)
make bench-yoga-serial

# Generate quality scorecards
make score-prod

# Stop everything
make teardown-yoga
```

### What Gets Measured

Each benchmark captures:
- **Aggregate throughput** (tok/s across all concurrent users)
- **Decode speed** (tok/s per user — how fast text appears)
- **TTFT** (time to first token — how long before you see any output)
- **ITL** (inter-token latency — consistency of token delivery)
- **Tail latency** (P99, P99.9 — worst-case experience)
- **Error rate** (should be 0%)

Results are saved as JSON in `results/` and scored via [probador's scoring system](docs/specifications/scoring.yaml).

## Repository Structure

| Path | Purpose |
|------|---------|
| `forjar-yoga-*.yaml` | Deployment configs for each runtime on Yoga (RTX 4060L) |
| `forjar-gx10.yaml` | Grace Blackwell GB10 deployment |
| `forjar.yaml` | CPU-only deployment (Intel Xeon) |
| `prompts/correctness.yaml` | 6 test prompts (math, code gen, explanation, JSON, SQL) |
| `scripts/nightly.sh` | Automated benchmark pipeline (`yoga`, `gx10`, `wgpu`, `cpu`) |
| `results/` | JSON benchmark results (git-tracked) |
| `docs/specifications/` | Performance spec (v6.34.0, 414 work items) and scoring contracts |

## Available Make Targets

```bash
# Yoga (RTX 4060L) — primary benchmark platform
make bench-yoga-prod          # All 4 runtimes, production methodology
make bench-yoga-prod-realizr  # realizr only
make bench-yoga-prod-vllm     # vLLM only
make score-prod               # Production scorecards

# GB10 (Blackwell) — larger model testing
make bench-gx10               # realizr on GB10 (requires SSH tunnel)
make test-gx10                # Correctness tests

# CPU (Intel Xeon)
make deploy && make test && make load

# WGPU (AMD Radeon)
make build-wgpu && make deploy-wgpu && make test-wgpu

# Scoring
make score                    # All scorecards
make score-gate               # CI gate: fail if any runtime below C grade
```

## Methodology

All benchmarks use **production methodology** (introduced at PMAT-177):
- **Medium prompts** (~102 tokens) — realistic, not cherry-picked short prompts
- **Uniform output distribution** (16-256 tokens) — simulates real traffic
- **Streaming** enabled — measures TTFT and ITL, not just batch throughput
- **60-second runs** with 5-second warmup — steady-state, not burst
- **Locked GPU clocks** (1900 MHz) — eliminates thermal throttle variance
- **Isolated serial** — one runtime at a time, clean GPU state between runs

Quality scoring uses absolute thresholds (not relative rankings), with jitter penalties and best-in-class bonuses. Full scoring contract: [scoring.yaml](docs/specifications/scoring.yaml).

## Deep Dive

The full 414-item performance specification documents every optimization attempt, including 16 kernel fusion approaches that were tested and falsified:

[gpu-performance-spec.md](docs/specifications/gpu-performance-spec.md) (v6.34.0)

This follows Popperian falsification methodology — every claim has a prediction that can be disproved by measurement. The spec includes profiling data, root cause analyses, and academic references for the architectural decisions.
