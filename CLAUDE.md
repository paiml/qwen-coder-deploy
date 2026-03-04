# CLAUDE.md

## Project Overview

Deployment infrastructure for Qwen2.5-Coder-1.5B-Instruct across three inference runtimes:
- **realizar** (Sovereign AI Stack) — port 8081
- **ollama** — port 8082
- **llama.cpp** (llama-server) — port 8083

Uses **forjar** for declarative deployment and **probador** (probar CLI) for correctness testing and load testing.

## Architecture

```
Jetson Orin (dedicated load testing)     4090 Host (QLoRA training + deep profiling)
├── realizar   :8081  (GGUF, CUDA)       ├── QLoRA fine-tuning (full-time)
├── ollama     :8082  (GGUF, CUDA)       ├── Deep profiling (occasional):
├── llama.cpp  :8083  (GGUF, CUDA)       │   nsys-gpu, ncu-gpu, profile-gpu
├── apr-native :8084  (APR, CUDA)        └── Builds: apr, llama.cpp, trueno
└── probador load tests (continuous)

CPU (intel, 192.168.50.100)
├── qwen-realizar    :8081  (systemd)
├── qwen-ollama      :8082  (systemd)
└── qwen-llamacpp    :8083  (systemd)
```

## Commands

```bash
# Jetson deployment (dedicated load testing — primary)
make deploy-jetson     # forjar apply -f forjar-jetson.yaml
make health-jetson     # Health check all 4 services
make test-jetson       # Correctness tests
make load-jetson       # Load tests (60s, concurrency=4, 5s warmup)
make teardown-jetson   # Stop all services
make nightly-jetson    # Full pipeline: deploy → health → test → load → report

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

## Forjar Configs

- `forjar-jetson.yaml` — Jetson Orin deployment: apr, ollama, llama.cpp (dedicated load testing)
- `forjar-jetson-teardown.yaml` — Stop Jetson services
- `forjar-gpu.yaml` — 4090 deployment: deep profiling only (occasional)
- `forjar.yaml` — CPU deployment: SSH to intel host, systemd services
- `forjar-gpu-teardown.yaml` / `forjar-teardown.yaml` — Stop services

## Testing

Correctness tests defined in `prompts/correctness.yaml` (6 prompts: math, code gen, explanation, JSON, SQL).
Load tests via `probador llm load` with configurable concurrency and duration.

**Important:** probador `--url` takes the base URL (e.g., `http://jetson:8081`), NOT the full endpoint path. It appends `/v1/chat/completions` internally.

**Important:** Ollama requires `--model qwen2.5-coder:1.5b-instruct` (exact tag from `ollama list`).

## Key Files

- `forjar-jetson.yaml` — Jetson Orin deployment (primary load testing)
- `forjar-gpu.yaml` — 4090 deployment (deep profiling only)
- `forjar.yaml` — CPU deployment configuration (intel host)
- `prompts/correctness.yaml` — Correctness test suite
- `contracts/gpu-performance-spec.md` — Performance specification (v2.3.0)
- `performance.md` — Historical performance data (auto-updated)
- `results/` — JSON result files (git-tracked)
- `scripts/nightly.sh` — Automated benchmark pipeline (cpu|gpu|both)
