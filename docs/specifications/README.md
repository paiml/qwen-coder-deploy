# Specifications — Table of Contents

All project specifications live under `docs/specifications/`.

## Top-Level Specs

| Spec | Description |
|------|-------------|
| [gpu-performance-spec.md](gpu-performance-spec.md) | GPU decoder throughput performance specification (v2.29.0) |
| [perf-parity-spec.md](perf-parity-spec.md) | Performance parity specification — scope, methodology, baselines |
| [benchmarking-v2.md](benchmarking-v2.md) | Benchmarking methodology v2 — test protocol |
| [scoring.yaml](scoring.yaml) | Quantitative scoring contract v3.0.0 — weights, thresholds, grades |
| [probador-llm-score-v1.yaml](probador-llm-score-v1.yaml) | Scoring CLI spec + falsification tests |
| [inference-showdown-v1.yaml](inference-showdown-v1.yaml) | Competition baselines (realizr vs llama.cpp vs ollama vs vLLM) |
| [batched-decode-launch-reduction-v1.yaml](batched-decode-launch-reduction-v1.yaml) | Batched decode kernel launch reduction contract |

## Component Specs (gpu-performance-spec subsections)

| Spec | Description |
|------|-------------|
| [components/baselines.md](components/baselines.md) | Baseline tables, threshold registry, measurement protocol |
| [components/kernel-specifications.md](components/kernel-specifications.md) | Kernel implementation details and PTX specs |
| [components/optimization-tiers.md](components/optimization-tiers.md) | Optimization tiers with acceptance criteria and citations |
| [components/root-cause-analysis.md](components/root-cause-analysis.md) | Root cause analysis details |
| [components/falsification-tests.md](components/falsification-tests.md) | Hypothesis definitions, F-tests, QA checklist |
| [components/profiling-data.md](components/profiling-data.md) | Profiling tables, PCIe analysis, warp sweeps |
| [components/pmat-work-tickets.md](components/pmat-work-tickets.md) | PMAT ticket YAML definitions |
| [components/continuous-batching.md](components/continuous-batching.md) | Continuous batching design spec |

## Component Specs (perf-parity-spec subsections)

| Spec | Description |
|------|-------------|
| [components/gguf-decode.md](components/gguf-decode.md) | GGUF decode pipeline specification |
| [components/gguf-prefill.md](components/gguf-prefill.md) | GGUF prefill pipeline specification |
| [components/forjar-yoga.md](components/forjar-yoga.md) | Yoga deployment forjar specification |
| [components/improved-llm-load-testing.md](components/improved-llm-load-testing.md) | Improved LLM load testing spec |
