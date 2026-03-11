# Improved LLM Load Testing

**Version:** 1.3.0
**Date:** 2026-03-09
**Status:** IMPLEMENTED — all bugs fixed, re-verified via cargo install
**Parent:** [perf-parity-spec.md](../perf-parity-spec.md) (Component #7)

---

## Motivation

Probador's LLM load testing (GH-22 through GH-26) ships solid fundamentals:
closed-loop and Poisson open-loop generation, SSE streaming with per-token
timestamps, TTFT/ITL/TPOT percentiles, SLO goodput, multi-run CI, and
regression detection. This puts it ahead of most ad-hoc scripts.

However, comparison with vLLM `benchmark_serving.py`, NVIDIA GenAI-Perf/AIPerf,
Bench360, GuideLLM, MLPerf Inference v5, and LLMPerf reveals five capability
gaps that block production-grade benchmarking:

| # | Gap | Why It Matters | Reference |
|---|-----|----------------|-----------|
| 1 | Manual concurrency sweep | Cannot find saturation knee in one invocation | GuideLLM, AIPerf |
| 2 | No GPU telemetry | Cannot correlate perf with thermal/BW/power | GenAI-Perf (DCGM), Bench360 (NVML) |
| 3 | No tail latency analysis | P99 hides P99.9 outliers; no jitter detection | arXiv:2507.09019, MLPerf |
| 4 | Synthetic-only prompts | Fixed-length profiles miss real I/O distributions | vLLM (ShareGPT), GuideLLM |
| 5 | Quality unchecked under load | Speed measured, correctness assumed | LLMPerf, MLPerf (ROUGE) |

**Scope:** All five features are additive — zero changes to existing CLI flags or
result schema fields. New flags, new output fields, new subcommands only.

---

## Feature 1: Concurrency Sweep with Saturation Detection

### Problem

Finding the optimal operating point requires running `probador llm load` at
c=1, c=2, c=4, c=8, c=16 as separate invocations, then manually comparing
results. GuideLLM research shows >50% of naive sweep runs are wasted on
saturated configurations.

### Design

```
probador llm sweep \
  --url http://192.168.50.38:8081 \
  --concurrency-levels 1,2,4,8,16 \
  --duration 30s \
  --warmup 5s \
  --stream true \
  --saturation-threshold 2.0 \
  --output sweep-results.json
```

**Algorithm:**
1. Run each concurrency level sequentially (with teardown/warmup between)
2. After each level, compute throughput gain ratio:
   `gain = throughput[c] / throughput[c_prev]`
3. If `gain < 1.0 / saturation_threshold` (throughput decreased) → stop early
4. If `latency_p99[c] > saturation_threshold * latency_p99[c=1]` → mark saturated
5. Report Pareto frontier: concurrency levels where throughput increased without
   latency exceeding threshold

**Output schema (new fields in sweep result):**

```json
{
  "sweep": {
    "levels": [
      {
        "concurrency": 1,
        "throughput_rps": 2.3,
        "latency_p99_ms": 450,
        "decode_tok_s": 140.0,
        "saturated": false
      },
      {
        "concurrency": 4,
        "throughput_rps": 5.8,
        "latency_p99_ms": 890,
        "decode_tok_s": 51.8,
        "saturated": false
      },
      {
        "concurrency": 8,
        "throughput_rps": 5.9,
        "latency_p99_ms": 2100,
        "decode_tok_s": 28.1,
        "saturated": true,
        "saturation_reason": "latency_p99 2100ms > 2.0x baseline 450ms"
      }
    ],
    "optimal_concurrency": 4,
    "optimal_throughput_rps": 5.8,
    "pareto_frontier": [1, 4]
  }
}
```

**CLI args (new subcommand `sweep`):**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--concurrency-levels` | `Vec<usize>` | `1,2,4,8,16` | Concurrency levels to sweep |
| `--saturation-threshold` | `f64` | `2.0` | P99 latency multiplier vs c=1 to declare saturated |
| `--early-stop` | `bool` | `true` | Stop sweep when saturated |

**References:**
- GuideLLM oversaturation detection: "reduce LLM benchmarking costs" (Red Hat, 2025)
- AIPerf ramping strategies: concurrent users sweep with stability detection
- NVIDIA NIM: "sweep request rates from 1 to just above max batch size"

### Acceptance Criteria

- [x] `probador llm sweep --concurrency-levels 1,2,4` produces sweep JSON
- [x] Early stop triggers when throughput plateaus
- [x] Pareto frontier identifies optimal concurrency — **GH-33 fixed: excludes zero-decode levels**
- [x] Each level uses isolated warmup — **GH-35: already wired via LoadTest warmup_duration**

---

## Feature 2: GPU Telemetry Collection

### Problem

Cannot correlate performance with GPU state. Jetson Orin benchmarks show
variance from thermal throttling (918 MHz → 600 MHz under sustained load).
4090 benchmarks cannot measure memory bandwidth utilization. No energy
efficiency metrics for edge deployment cost analysis.

### Design

Poll `nvidia-smi` (or NVML via `/dev/nvidiactl`) at 1-second intervals
during benchmark runs. Collect and aggregate in the result JSON.

**Collection method (portable, no NVML crate dependency):**

```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.gr,clocks.mem \
  --format=csv,noheader,nounits -l 1
```

On Jetson (tegrastats):
```bash
tegrastats --interval 1000
```

**CLI activation:**

```
probador llm load \
  --url http://192.168.50.38:8081 \
  --gpu-telemetry \
  --gpu-poll-interval 1s \
  --duration 60s
```

**Output schema (new `gpu_telemetry` section):**

```json
{
  "gpu_telemetry": {
    "samples": 60,
    "gpu_utilization_pct": { "mean": 82.3, "max": 97.0, "min": 45.0 },
    "memory_used_mb": { "mean": 1842, "max": 1856, "min": 1830 },
    "memory_total_mb": 8192,
    "power_draw_w": { "mean": 45.2, "max": 55.0, "min": 38.0 },
    "temperature_c": { "mean": 72, "max": 81, "min": 65 },
    "clock_gpu_mhz": { "mean": 1485, "max": 1500, "min": 1200 },
    "throttle_events": 3,
    "energy_total_wh": 0.753,
    "energy_per_token_mj": 2.84,
    "energy_per_request_mj": 42.6
  }
}
```

**Derived metrics:**
- `energy_total_wh = sum(power_draw_w * interval_s) / 3600`
- `energy_per_token_mj = (energy_total_wh * 3600 * 1000) / completion_tokens_total`
- `throttle_events = count(clock_gpu_mhz < expected_clock * 0.9)`

**Implementation approach:**
1. Spawn `nvidia-smi -l 1` as background process before benchmark starts
2. Parse CSV output into `Vec<GpuSample>` during collection
3. Kill process after benchmark completes
4. Aggregate into summary statistics
5. On Jetson, fall back to `tegrastats` parsing

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--gpu-telemetry` | `bool` | `false` | Enable GPU metric collection |
| `--gpu-poll-interval` | `Duration` | `1s` | Polling interval |
| `--expected-clock-mhz` | `u32` | auto-detect | For throttle detection |

**References:**
- GenAI-Perf: DCGM Exporter metrics (utilization, power, memory)
- Bench360 (arXiv:2511.16682): NVML + psutil, peak/avg GPU utilization
- TokenPowerBench (arXiv:2512.03024): Joules per token, watts average

### Acceptance Criteria

- [x] `--gpu-telemetry` collects nvidia-smi data during load test
- [x] `gpu_telemetry` section present in output JSON (17 samples in 15s test)
- [x] `energy_per_token_mj` computed correctly (269.32 mJ/token on 4090)
- [x] Throttle events detected when clock drops >10% (13 events detected)
- [x] Works on Yoga (nvidia-smi) — **GH-34 fixed: SSH-based remote collection**
- [ ] Jetson (tegrastats fallback) — not implemented

### Verified Results (Dogfooding, 2026-03-09)

GPU telemetry via SSH to yoga (192.168.50.38):
- GPU util: 100% (c=4), Memory: 5812 / 8188 MB, Clock: 1500 MHz (locked)
- Power: 44.8W avg, Energy: 337.63 mJ/token
- No throttle events (clocks stable at 1500 MHz)

### Remaining Gaps

1. **No tegrastats fallback.** Jetson Orin requires `tegrastats` parsing, not implemented.

---

## Feature 3: Tail Latency Analysis and Jitter Detection

### Problem

P99 hides worst-case outliers. A system with P50=7ms, P99=20ms might have
P99.9=500ms (GC pause, graph recapture, KV cache eviction). Current output
stops at P99. No detection of latency spikes or drift over time.

### Design

Extend per-request data analysis with tail percentiles, tail ratio, jitter
metrics, and time-series drift detection.

**New output fields (added to `LoadTestResult`):**

```json
{
  "tail_analysis": {
    "itl_p999_ms": 45.2,
    "itl_p9999_ms": 128.7,
    "ttft_p999_ms": 210.5,
    "ttft_p9999_ms": 890.3,
    "latency_p999_ms": 2100.0,
    "latency_p9999_ms": 4500.0,

    "tail_ratio_itl": 6.37,
    "tail_ratio_ttft": 4.95,
    "tail_ratio_latency": 13.95,

    "jitter": {
      "itl_cv": 0.42,
      "itl_iqr_ms": 3.2,
      "spike_count": 7,
      "spike_threshold_ms": 50.0,
      "spikes": [
        { "request_idx": 42, "itl_ms": 127.3, "timestamp_s": 23.4 },
        { "request_idx": 89, "itl_ms": 85.1, "timestamp_s": 41.2 }
      ]
    },

    "drift": {
      "itl_slope_ms_per_min": 0.3,
      "ttft_slope_ms_per_min": 1.2,
      "throughput_slope_rps_per_min": -0.05,
      "degradation_detected": false
    }
  }
}
```

**Definitions:**
- `tail_ratio = p99 / p50` — measures outlier severity (>5 = concern, >10 = critical)
- `jitter.itl_cv = stddev(itl) / mean(itl)` — coefficient of variation
- `spike_count` — requests where ITL > `spike_threshold` (default: 5x median ITL)
- `drift.itl_slope` — linear regression of ITL over time (positive = degradation)
- `degradation_detected` — true if slope is statistically significant (p < 0.05)

**Algorithm for drift detection:**
1. Divide benchmark into 10-second windows
2. Compute median ITL per window
3. Fit linear regression: `itl_median = a + b * time`
4. If `b > 0` and `r^2 > 0.5` → degradation detected

**CLI flags:**

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--tail-analysis` | `bool` | `true` | Enable tail analysis (on by default) |
| `--spike-threshold` | `f64` | `5.0` | Multiplier of median ITL to classify as spike |
| `--drift-window-s` | `u64` | `10` | Time window for drift regression |

**References:**
- arXiv:2507.09019 "On Evaluating Performance of LLM Inference Serving": recommends
  P99.9 reporting, coefficient of variation, time-series stability
- MLPerf Inference v5: tail latency as primary metric for server scenario
- NVIDIA NIM benchmarking: "monitor for latency degradation over time"

### Acceptance Criteria

- [x] P99.9 and P99.99 computed for ITL, TTFT, and end-to-end latency
- [x] Tail ratio reported (1.0x-2.3x observed at c=1; printed in CLI output)
- [x] Spikes identified with timestamp and request index (unit tested)
- [x] Drift detection catches monotonic ITL increase over 60s run
- [x] Always-on (no `--tail-analysis` flag needed; negligible overhead)

### Known Issues (Dogfooding, 2026-03-09)

1. **Drift false positive fixed.** Initial implementation flagged 11-sample runs as
   "degrading" because r²>0.5 is trivially achievable with small N. Fixed by requiring:
   `n >= 30 AND itl_slope > 0 AND r² > 0.5 AND slope_per_min > 1% of median ITL`.
2. **No `--tail-analysis false` flag.** Spec proposed opt-out; implementation is always-on
   with negligible overhead (percentile computation on in-memory vec). No flag needed.
3. **Tail ratio >5x flag not printed.** Tail ratios are reported but not flagged with a
   warning when exceeding 5x. Low priority since typical values are 1.0-2.3x.

---

## Feature 4: Dataset-Driven Workload Profiles

### Problem

All probador benchmarks use synthetic fixed-length prompts (micro/short/medium/long).
Real LLM traffic has diverse input/output length distributions. A system tuned for
32-token inputs may collapse at 2048-token inputs due to KV cache pressure.

vLLM, GuideLLM, and AIPerf all support dataset-driven workloads that replay
realistic token distributions from ShareGPT, OpenOrca, or custom datasets.

### Design

Add `--dataset` flag that accepts a JSONL file (or HuggingFace dataset name)
where each line specifies a prompt with target output length.

**Dataset format (ShareGPT-compatible JSONL):**

```jsonl
{"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 16}
{"messages": [{"role": "user", "content": "Write a Python function..."}], "max_tokens": 256}
{"messages": [{"role": "system", "content": "You are..."}, {"role": "user", "content": "Explain..."}], "max_tokens": 128}
```

**CLI usage:**

```
probador llm load \
  --url http://192.168.50.38:8081 \
  --dataset prompts/sharegpt-sample.jsonl \
  --concurrency 4 \
  --duration 60s \
  --stream true
```

**Distribution analysis in output:**

```json
{
  "dataset_stats": {
    "source": "prompts/sharegpt-sample.jsonl",
    "total_prompts": 500,
    "input_tokens": { "min": 8, "p50": 87, "p90": 312, "max": 2048 },
    "max_tokens_requested": { "min": 16, "p50": 128, "p90": 256, "max": 512 },
    "actual_output_tokens": { "min": 4, "p50": 95, "p90": 210, "max": 498 }
  }
}
```

**Sampling strategy:**
- During load test, cycle through dataset entries round-robin (deterministic)
- If dataset has fewer entries than requests, wrap around
- Each worker gets a different starting offset to avoid lock-step

**Built-in dataset generator:**

```
probador llm gen-dataset \
  --distribution lognormal \
  --input-mean 128 --input-stddev 64 \
  --output-mean 128 --output-stddev 96 \
  --count 1000 \
  --output prompts/synthetic-lognormal.jsonl
```

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--dataset` | `PathBuf` | None | JSONL dataset file |
| `--dataset-sample` | `usize` | all | Random sample N entries from dataset |

**References:**
- vLLM `benchmark_serving.py`: `--dataset-name sharegpt` with tokenizer-counted lengths
- GuideLLM: HuggingFace dataset loading with automatic token distribution matching
- AIPerf: "fixed schedule (trace) inference load mode"
- arXiv:2507.09019: "workload characterization should match production distributions"

### Acceptance Criteria

- [x] `--dataset prompts/sample.jsonl` loads and cycles prompts from file (20 prompts tested)
- [x] `dataset_stats` section in output JSON with I/O token distributions
- [x] Workers sample different starting offsets (round-robin with worker offset)
- [x] `probador llm gen-dataset` creates synthetic JSONL with configurable distribution
- [x] Backward compatible: existing `--prompt-profile` still works, `--dataset` overrides it

---

## Feature 5: Inline Correctness Validation During Load

### Problem

`probador llm test` (correctness) and `probador llm load` (throughput) are
completely separate. Quality degradation under load is invisible. The c=4
batching bug that produced zero decode tokens went undetected because load
tests only checked `error_rate` (HTTP 200 with empty body = "success").

LLMPerf and MLPerf both validate output quality during load runs.

### Design

Add optional `--validate` flag to `probador llm load` that checks each
response against basic quality criteria and reports degradation.

**Validation levels:**

| Level | Checks | Overhead |
|-------|--------|----------|
| `none` (default) | No validation | Zero |
| `basic` | Non-empty content, finish_reason present, token count > 0 | ~0% |
| `contains:PATTERN` | Basic + response contains substring | ~0% |
| `pattern:REGEX` | Basic + response matches regex | ~1% |

**CLI usage:**

```
probador llm load \
  --url http://192.168.50.38:8081 \
  --concurrency 4 \
  --duration 60s \
  --validate basic \
  --stream true
```

Or with content checks:

```
probador llm load \
  --url http://192.168.50.38:8081 \
  --prompt-profile short \
  --validate "contains:hash table" \
  --duration 60s
```

**Output schema (new `quality` section):**

```json
{
  "quality": {
    "validation_level": "basic",
    "total_validated": 336,
    "passed": 330,
    "failed": 6,
    "pass_rate": 0.982,
    "failures": [
      { "request_idx": 42, "reason": "empty_content", "concurrency_at_time": 4 },
      { "request_idx": 89, "reason": "no_finish_reason", "concurrency_at_time": 4 },
      { "request_idx": 156, "reason": "zero_tokens", "concurrency_at_time": 4 }
    ],
    "quality_vs_concurrency": {
      "1": { "pass_rate": 1.0, "count": 84 },
      "4": { "pass_rate": 0.964, "count": 252 }
    }
  }
}
```

**Integration with existing assertion framework:**

Reuses `jugar_probar::llm::assertion::LlmAssertion` from the `llm-types` feature
(GH-22). The `basic` level is equivalent to `LlmAssertion::new().assert_response_valid()`.
Higher levels compose additional checks.

**Key insight:** `quality_vs_concurrency` cross-tabulates pass rate against
concurrent request count at the time each request completed. This directly
reveals batching-induced quality degradation (the c=4 bug pattern).

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--validate` | `String` | `none` | Validation level: none, basic, contains:X, pattern:X |
| `--fail-on-quality` | `f64` | None | Exit 1 if pass_rate < threshold (e.g., 0.95) |

**References:**
- LLMPerf: correctness test alongside throughput measurement
- MLPerf Inference v5: ROUGE accuracy check on every response
- arXiv:2410.14257 "Smooth Goodput": quality-weighted throughput metric
- Probador c=4 zero-decode-tokens incident: empty responses went undetected

### Acceptance Criteria

- [x] `--validate basic` checks non-empty content on every response
- [x] `quality` section in output JSON with pass_rate and failure details
- [ ] `quality_vs_concurrency` reveals degradation at higher concurrency — not implemented
- [x] `--fail-on-quality 0.95` exits non-zero when quality drops (exit 1 at 0% pass rate)
- [ ] Validation reuses existing `LlmAssertion` from `llm-types` — uses custom logic
- [x] Zero overhead when `--validate none` (default)

### Known Issues (Dogfooding, 2026-03-09)

1. **`quality_vs_concurrency` not implemented.** The spec proposed cross-tabulating pass
   rate against concurrency at the time each request completed. This requires tracking
   active concurrency per-request, which is not currently captured. The quality section
   reports aggregate pass rate and per-request failures.
2. **Does not reuse `LlmAssertion`.** Uses custom validation logic (check tokens>0,
   finish_reason present, content non-empty) rather than composing `LlmAssertion`. The
   checks are equivalent; custom logic avoids pulling in the assertion framework dependency.
3. **`contains:` validation captures streaming content.** For SSE streaming, the full
   streamed content is concatenated and checked. Verified: `contains:the` passes 100%,
   `contains:def ` correctly fails when model doesn't produce Python functions.

---

## Implementation Priority

| # | Feature | Effort | Impact | Depends On |
|---|---------|--------|--------|------------|
| 5 | Inline correctness | 2h | HIGH — catches c=4 class bugs | llm-types (GH-22) | **DONE** |
| 3 | Tail latency + jitter | 3h | HIGH — P99.9, drift detection | None | **DONE** |
| 2 | GPU telemetry | 4h | HIGH — thermal/power/BW correlation | nvidia-smi on target | **DONE** |
| 1 | Concurrency sweep | 4h | MEDIUM — automates manual workflow | None | **DONE** |
| 4 | Dataset workloads | 3h | MEDIUM — realistic I/O distributions | None | **DONE** |

**Recommended order:** 5 → 3 → 2 → 1 → 4

Feature 5 is highest ROI: 2 hours of work, directly prevents the c=4 bug class
that wasted multiple debugging sessions. Feature 3 requires no new dependencies.
Feature 2 needs `nvidia-smi` access on the target machine (already available on
all three targets).

---

## Research References

### Papers

- **"Revisiting SLO and Goodput Metrics in LLM Serving"** (arXiv:2410.14257)
  — Defines smooth goodput: `benefit(r) = n_tokens - alpha * f(idle_latency)`.
  Shows DistServe serves 7.4x more requests within SLO by optimizing for goodput.

- **"On Evaluating Performance of LLM Inference Serving"** (arXiv:2507.09019)
  — Recommends P99.9 reporting, coefficient of variation, time-series stability,
  open-loop Poisson generation, and workload characterization matching production.

- **"Bench360: Benchmarking Local LLM Inference from 360 Degrees"** (arXiv:2511.16682)
  — NVML-based GPU telemetry: peak/avg utilization, memory, power. Energy per
  token derived from average power x generation time.

- **"TokenPowerBench"** (arXiv:2512.03024) — Joules per token, watts average
  during inference, energy efficiency as primary metric for edge deployment.

- **"LLM-Inference-Bench"** (arXiv:2411.00136) — Standardized benchmark suite
  across diverse hardware. Finds 10-100x performance variation across configs.

### Tools

- **vLLM benchmark_serving.py** — `--goodput ttft:3000,tpot:100` for SLO-defined
  goodput; `--burstiness` parameter for gamma-distributed arrivals; ShareGPT dataset.

- **NVIDIA GenAI-Perf / AIPerf** — DCGM telemetry, TUI dashboard, concurrency
  ramping with stability detection, mooncake-format trace replay.

- **GuideLLM** (Red Hat) — Oversaturation detection saves >50% of sweep runs;
  HuggingFace dataset loading; automatic token distribution matching.

- **MLPerf Inference v5** — Server scenario (Poisson arrivals), ROUGE accuracy
  checks, tail latency as primary metric, standardized reporting.

- **LLMPerf** (Anyscale/Ray) — Concurrent correctness + throughput measurement;
  reproducible metrics methodology.

---

## Dogfooding Results (2026-03-09)

All 5 features tested against realizr on Yoga (RTX 4060 Laptop, `192.168.50.38:8081`).
Bugs found during initial dogfooding were filed as GH issues and fixed before re-testing.

### Round 1: Initial Dogfooding (bugs found)

| Bug | Issue | Status |
|-----|-------|--------|
| Sweep quality-blind (picks zero-decode c=2 as optimal) | GH-33 | **FIXED** |
| GPU telemetry collects local GPU, not remote target | GH-34 | **FIXED** |
| Sweep no per-level warmup | GH-35 | Already working (closed) |
| Drift false positive (n=11 triggered degradation_detected) | inline | **FIXED** |

### Round 2: Re-test After Fixes (cargo install, all pass)

**Feature 5: Inline Correctness Validation**

| Test | Result | Notes |
|------|--------|-------|
| `--validate basic` c=1 (warmup) | **PASS** | 16/16 pass, 100% |
| `--validate basic` c=4 (warmup) | **PASS** | 24/24 pass, 100% |
| `--validate contains:the` c=1 | **PASS** | 11/11 pass |
| `--fail-on-quality 0.95` c=1 (no warmup) | **FAIL** | 2/18 zero_tokens (cold start) |
| `--fail-on-quality 0.95` c=1 (5s warmup) | **PASS** | 16/16 pass, 100% |

**Feature 3: Tail Latency Analysis**

| Metric | c=1 | c=4 | Notes |
|--------|-----|-----|-------|
| ITL P99.9 | 7.3ms | 21.0ms | 1.0x / 1.0x tail ratio |
| TTFT P99.9 | 78.6ms | 265.3ms | 1.0x / 1.0x tail ratio |
| ITL CV | 0.00 | 0.01 | Very stable with warmup |
| Drift | Not detected | Not detected | Correct — n<30 |

**Feature 2: GPU Telemetry (GH-34 fixed)**

| Metric | c=1 | c=4 | Notes |
|--------|-----|-----|-------|
| Source | 192.168.50.38 (SSH) | 192.168.50.38 (SSH) | **Remote, not local** |
| GPU util | 100% avg | 100% avg | Correct target GPU |
| Memory | 5584 / 8188 MB | 5812 / 8188 MB | **8GB RTX 4060** (was 24GB local) |
| Clock | 1500 MHz | 1500 MHz | Locked, no throttle |
| Power | 45.2W avg | 44.8W avg | |
| Energy/token | 355.57 mJ/tok | 337.63 mJ/tok | |

**Feature 1: Concurrency Sweep (GH-33 fixed)**

| Level | Throughput | P99 | Decode tok/s | Status |
|-------|-----------|-----|-------------|--------|
| c=1 | 1.1 req/s | 992ms | 137.9 | Baseline |
| c=2 | 0.6 req/s | 3267ms | 40.9 | SATURATED (early stop) |

Optimal: c=1 (1.1 req/s). Pareto front: [1]. Zero-decode levels would now show
`[ZERO QUALITY]` and be excluded from optimal/Pareto.

**Feature 4: Dataset-Driven Workloads**

- `gen-dataset` produces valid JSONL (verified)
- `--dataset` loads prompts and reports `dataset_stats` (verified with 20-prompt file)
- Round-robin cycling with worker offset confirmed

### Summary

| Feature | Status | Issues Filed | Resolution |
|---------|--------|-------------|------------|
| 5. Quality validation | **PASS** | — | Caught realizr c=2 bug |
| 3. Tail analysis | **PASS** | — | Drift false positive fixed inline |
| 2. GPU telemetry | **PASS** | GH-34 | SSH remote collection |
| 1. Concurrency sweep | **PASS** | GH-33 | Quality-aware optimal |
| 4. Dataset workloads | **PASS** | — | Working as designed |

### Remaining Gaps (non-blocking)

- `quality_vs_concurrency` cross-tab not implemented (requires per-request active concurrency tracking)
- Jetson tegrastats fallback not implemented
- `LlmAssertion` reuse deferred (custom validation logic is equivalent)

### Realizr bugs discovered via dogfooding

- c=2 returns HTTP 200 with zero decode tokens (batch scheduler bug, intermittent)
- Server state corruption after c=2 load — requires forjar teardown/redeploy
