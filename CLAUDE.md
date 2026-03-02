# CLAUDE.md

## Project Overview

Deployment infrastructure for Qwen2.5-Coder-1.5B-Instruct across three inference runtimes:
- **realizar** (Sovereign AI Stack) — port 8081
- **ollama** — port 8082
- **llama.cpp** (llama-server) — port 8083

Uses **forjar** for declarative deployment and **probador** (probar CLI) for correctness testing and load testing.

## Architecture

```
GPU (localhost, RTX 4090)                CPU (intel, 192.168.50.100)
├── realizar   :8081  (GGUF, CUDA)       ├── qwen-realizar    :8081  (systemd)
├── ollama     :8082  (GGUF, CUDA)       ├── qwen-ollama      :8082  (systemd)
└── llama.cpp  :8083  (GGUF, CUDA)       └── qwen-llamacpp    :8083  (systemd)
```

## Commands

```bash
# GPU deployment (localhost)
make deploy-gpu        # forjar apply -f forjar-gpu.yaml
make health-gpu        # Health check all 3
make test-gpu          # Correctness tests
make load-gpu          # Load tests (60s, concurrency=4)
make teardown-gpu      # Stop all processes

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
| ollama | GGUF (Q4_K_M) | ollama pull qwen2.5-coder:1.5b |
| llama.cpp | GGUF (Q4_K_M) | Same GGUF file as realizar |

## Forjar Configs

- `forjar-gpu.yaml` — GPU deployment: builds llama-server, verifies models, starts 3 processes
- `forjar.yaml` — CPU deployment: SSH to intel host, systemd services
- `forjar-gpu-teardown.yaml` / `forjar-teardown.yaml` — Stop services

## Testing

Correctness tests defined in `prompts/correctness.yaml` (6 prompts: math, code gen, explanation, JSON, SQL).
Load tests via `probador llm load` with configurable concurrency and duration.

## Key Files

- `forjar-gpu.yaml` — GPU deployment configuration (localhost)
- `forjar.yaml` — CPU deployment configuration (intel host)
- `prompts/correctness.yaml` — Correctness test suite
- `performance.md` — Historical performance data (auto-updated)
- `results/` — JSON result files (git-tracked)
- `scripts/nightly.sh` — Automated benchmark pipeline (cpu|gpu|both)
