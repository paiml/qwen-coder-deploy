# CLAUDE.md

## Project Overview

Deployment infrastructure for Qwen2.5-Coder-1.5B-Instruct across four inference runtimes:
- **realizar** (Sovereign AI Stack) — port 8081
- **ollama** — port 8082
- **llama.cpp** (llama-server) — port 8083
- **vLLM** (AWQ INT4) — port 8084

Uses **forjar** for declarative deployment and **probador** (probar CLI) for correctness testing and load testing.

## Architecture

```
Yoga (PRIMARY benchmark target)          4090 Host (QLoRA training + deep profiling)
├── realizr    :8081  (GGUF, CUDA)       ├── QLoRA fine-tuning (full-time)
├── ollama     :8082  (GGUF, CUDA)       ├── Deep profiling (occasional):
├── llama.cpp  :8083  (GGUF, CUDA)       │   nsys-gpu, ncu-gpu, profile-gpu
├── vLLM       :8084  (AWQ INT4)         └── Builds: apr, llama.cpp, trueno
└── RTX 4060 Laptop, sm_89, 8GB

Jetson Orin (secondary load testing)     CPU (intel, 192.168.50.100)
├── realizr    :8081  (GGUF, CUDA)       ├── realizr     :8081  (nohup)
├── ollama     :8082  (GGUF, CUDA)       ├── ollama      :8082  (nohup)
├── llama.cpp  :8083  (GGUF, CUDA)       └── llama.cpp   :8083  (nohup)
└── 8 SMs, sm_87, 8GB unified
```

## Commands

```bash
# Yoga deployment (PRIMARY — serial isolated benchmarks)
make bench-yoga-serial   # All 4 runtimes, c=1 and c=4 (isolated)
make bench-yoga-realizr  # realizr only (isolated)
make bench-yoga-llamacpp # llama.cpp only (isolated)
make bench-yoga-ollama   # ollama only (isolated)
make bench-yoga-vllm     # vLLM only (isolated)
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

# Reports
make report            # Generate performance.md + update README
```

## Model Formats

| Runtime | Format | Source |
|---------|--------|--------|
| realizar | GGUF (Q4_K_M) | From HuggingFace, served via OpenAI-compat endpoint |
| ollama | GGUF (Q4_K_M) | ollama pull qwen2.5-coder:1.5b-instruct |
| llama.cpp | GGUF (Q4_K_M) | Same GGUF file as realizar |
| vLLM | AWQ INT4 | Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ from HuggingFace |

## Forjar Configs

- `forjar-yoga-realizr.yaml` / `forjar-yoga-llamacpp.yaml` / `forjar-yoga-ollama.yaml` / `forjar-yoga-vllm.yaml` — Yoga isolated deploy
- `forjar-yoga-teardown.yaml` — Stop yoga services (including vLLM)
- `forjar-jetson.yaml` / `forjar-jetson-teardown.yaml` — Jetson Orin deployment
- `forjar-gpu.yaml` / `forjar-gpu-teardown.yaml` — 4090 deep profiling
- `forjar.yaml` / `forjar-teardown.yaml` — CPU deployment (intel host, SSH)

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
- `contracts/gpu-performance-spec.md` — Performance specification (v2.15.0)
- `performance.md` — Historical performance data (auto-updated)
- `results/` — JSON result files (git-tracked)
- `scripts/nightly.sh` — Automated benchmark pipeline (cpu|gpu|both)
