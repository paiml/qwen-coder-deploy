# CLAUDE.md

## Project Overview

Deployment infrastructure for Qwen2.5-Coder-1.5B-Instruct across three inference runtimes:
- **realizar** (Sovereign AI Stack) — port 8081
- **ollama** — port 8082
- **llama.cpp** (llama-server) — port 8083

Uses **forjar** for declarative deployment and **probador** (probar CLI) for correctness testing and load testing.

## Architecture

```
intel (192.168.50.100)
├── qwen-realizar    :8081  (systemd, SafeTensors/APR)
├── qwen-ollama      :8082  (systemd, GGUF)
└── qwen-llamacpp    :8083  (systemd, GGUF)
```

## Commands

```bash
# Deploy all services
make deploy

# Run correctness tests against all runtimes
make test

# Run load tests
make load

# Generate performance reports
make report

# Full nightly cycle
make nightly

# Tear down all services
make teardown
```

## Model Formats

| Runtime | Format | Source |
|---------|--------|--------|
| realizar | SafeTensors + APR | HuggingFace Hub |
| ollama | GGUF (Q4_K_M) | Converted from SafeTensors |
| llama.cpp | GGUF (Q4_K_M) | Converted from SafeTensors |

## Testing

Correctness tests are defined in `prompts/correctness.yaml`. Each test sends a prompt and asserts the response contains expected text or matches a regex pattern.

Load tests use `probador llm load` with configurable concurrency and duration. Results are stored as JSON in `results/` and aggregated into `performance.md`.

## Key Files

- `forjar.yaml` — Deployment configuration
- `prompts/correctness.yaml` — Correctness test suite
- `performance.md` — Historical performance data (auto-updated)
- `results/` — JSON result files (git-tracked)
- `systemd/` — Systemd unit files for each runtime
