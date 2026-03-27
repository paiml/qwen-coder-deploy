# CLAUDE.md

## Project Overview

Deployment infrastructure for Qwen2.5-Coder-1.5B-Instruct across five inference runtimes:
- **realizar** (Sovereign AI Stack) — port 8081
- **ollama** — port 8082
- **llama.cpp** (llama-server) — port 8083
- **vLLM** (AWQ INT4) — port 8084
- **realizar-wgpu** (WGPU/Vulkan on AMD) — port 8081 on intel

Uses **forjar** for declarative deployment and **probador** (probar CLI) for correctness testing and load testing.

## Architecture

```
Yoga (PRIMARY benchmark target)          4090 Host (QLoRA training + deep profiling)
├── realizr    :8081  (GGUF, CUDA)       ├── QLoRA fine-tuning (full-time)
├── ollama     :8082  (GGUF, CUDA)       ├── Deep profiling (occasional):
├── llama.cpp  :8083  (GGUF, CUDA)       │   nsys-gpu, ncu-gpu, profile-gpu
├── vLLM       :8084  (AWQ INT4)         └── Builds: apr, llama.cpp, trueno
└── RTX 4060 Laptop, sm_89, 8GB

Jetson Orin (secondary load testing)     Intel (192.168.50.100, WGPU + CPU)
├── realizr    :8081  (GGUF, CUDA)       ├── realizr-wgpu :8081  (GGUF, Vulkan)
├── ollama     :8082  (GGUF, CUDA)       │   Radeon Pro W5700X (Navi 10, 8GB)
├── llama.cpp  :8083  (GGUF, CUDA)       ├── realizr-cpu  :8082  (GGUF, AVX2)
└── 8 SMs, sm_87, 8GB unified           └── llama.cpp    :8083  (GGUF, CPU)

gx10 (Grace Blackwell GB10)
├── realizr    :8081  (GGUF, CUDA sm_121)
├── 120 GB unified memory
├── CUDA 13.0, compute 12.1
└── HumanEval: 84.76% pass@1 (7B Q4K)
```

## Commands

```bash
# Yoga deployment (PRIMARY — serial isolated benchmarks)
make bench-yoga-serial   # All 4 runtimes, c=1 and c=4 (short prompt, isolated)
make bench-yoga-prod     # All 4 runtimes (production-realistic)
make bench-yoga-prod-realizr   # realizr only (c=1-128, medium + hetero output)
make bench-yoga-prod-llamacpp  # llama.cpp only (c=1-32)
make bench-yoga-prod-ollama    # ollama only (c=1-32)
make bench-yoga-prod-vllm     # vLLM only (c=1-128)
make bench-yoga-realizr  # realizr only (short prompt, c=1,4)
make bench-yoga-llamacpp # llama.cpp only (short prompt, c=1,4)
make bench-yoga-ollama   # ollama only (short prompt, c=1,4)
make bench-yoga-vllm     # vLLM only (short prompt, c=1,4)
make teardown-yoga       # Stop all services

# Jetson deployment (secondary load testing)
make deploy-jetson     # forjar apply -f forjar-jetson.yaml
make bench-jetson-serial # Isolated benchmarks
make teardown-jetson   # Stop all services

# 4090 deployment (deep profiling only — 4090 runs QLoRA full-time)
make deploy-gpu        # forjar apply -f forjar-gpu.yaml
make nsys-gpu          # nsys kernel timeline
make ncu-gpu           # ncu per-kernel roofline
make profile-gpu       # apr profile (roofline + hotspots)

# CPU deployment (intel host)
make deploy            # forjar apply
make test              # Correctness tests
make load              # Load tests
make teardown          # Stop services

# WGPU deployment (intel host, Radeon Pro W5700X)
make build-wgpu        # Build apr with WGPU feature
make deploy-wgpu       # Deploy to intel
make start-wgpu        # Start server (F32 mode, 0.74 tok/s, 6175 MB VRAM)
make test-wgpu         # 3 correctness tests
make stop-wgpu         # Stop server
# Q4K mode: WGPU_Q4K=1 — 10× VRAM savings (626 MB), 0.46 tok/s (vec4 optimized)
# Parity gate: make parity-wgpu (WGPU vs CPU factual comparison)
# Streaming: curl -N http://192.168.50.100:8081/v1/chat/completions -d '{"stream":true,...}'

# Reports
make report            # Generate performance.md + update README

# Scoring (probador llm score)
make score             # Yoga c=1,4,8,16,32,64,128 scorecard (table)
make score-prod        # Production methodology results only (PMAT-177+)
make score-jetson      # Jetson scorecards
make score-json        # JSON scorecards to results/
make score-all         # All platforms, all concurrency levels
make score-gate        # CI gate: fail if any runtime below C
```

## Model Formats

| Runtime | Format | Source |
|---------|--------|--------|
| realizar | GGUF (Q4_K_M) | From HuggingFace, served via OpenAI-compat endpoint |
| ollama | GGUF (Q4_K_M) | ollama pull qwen2.5-coder:1.5b-instruct |
| llama.cpp | GGUF (Q4_K_M) | Same GGUF file as realizar |
| vLLM | AWQ INT4 | Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ from HuggingFace |
| realizar-wgpu | GGUF (Q4_K_M→F32) | Same GGUF, dequantized to F32 for WGPU/Vulkan on AMD |

## Forjar Configs

- `forjar-yoga-realizr.yaml` / `forjar-yoga-llamacpp.yaml` / `forjar-yoga-ollama.yaml` / `forjar-yoga-vllm.yaml` — Yoga isolated deploy
- `forjar-yoga-teardown.yaml` — Stop yoga services (including vLLM)
- `forjar-jetson.yaml` / `forjar-jetson-teardown.yaml` — Jetson Orin deployment
- `forjar-gpu.yaml` / `forjar-gpu-teardown.yaml` — 4090 deep profiling
- `forjar.yaml` / `forjar-teardown.yaml` — CPU deployment (intel host, SSH)
- `forjar-intel-wgpu.yaml` — WGPU deployment (intel host, Radeon Pro W5700X)
- `forjar-gx10.yaml` — Grace Blackwell GB10 deployment (sm_121, 120 GB unified)

## Testing

Correctness tests defined in `prompts/correctness.yaml` (6 prompts: math, code gen, explanation, JSON, SQL).
Load tests via `probador llm load` with configurable concurrency and duration.

**Important:** probador `--url` takes the base URL (e.g., `http://jetson:8081`), NOT the full endpoint path. It appends `/v1/chat/completions` internally.

**Important:** Ollama requires `--model qwen2.5-coder:1.5b-instruct` (exact tag from `ollama list`).

**Important:** vLLM requires `--model Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ` (HuggingFace model ID). Uses AWQ INT4, NOT GGUF.

## Key Files

- `forjar-jetson.yaml` — Jetson Orin deployment (primary load testing)
- `forjar-gpu.yaml` — 4090 deployment (deep profiling only)
- `forjar.yaml` — CPU deployment configuration (intel host)
- `prompts/correctness.yaml` — Correctness test suite
- `docs/specifications/gpu-performance-spec.md` — Performance specification (v5.77.0, 349 PMAT items)
- `docs/specifications/scoring.yaml` — Quantitative scoring contract v2.0.0 (weights, thresholds, grades)
- `docs/specifications/probador-llm-score-v1.yaml` — Scoring CLI spec + falsification tests
- `performance.md` — Historical performance data (auto-updated)
- `results/` — JSON result files + scorecards (git-tracked)
- `scripts/nightly.sh` — Automated benchmark + scoring pipeline (cpu|gpu|both)
