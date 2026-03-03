# Inference Benchmarking Methodology v2.0

**Status:** DRAFT — Supersedes ad-hoc methodology in inference-showdown-v1.yaml
**Model:** Qwen2.5-Coder-1.5B-Instruct (Q4_K_M, 1.5B params, 28 layers)
**Machine:** Lambda Vector — Xeon w9-3545X (96 cores), RTX 4090 (24GB), DDR5
**Date:** 2026-03-03

## 1. Problem Statement

Our current benchmarking workflow is ad-hoc and non-deterministic. Specific failures:

| Issue | Impact | Industry Standard |
|-------|--------|-------------------|
| Single 30s run, no repeats | Results are unreproducible noise | MLPerf: 600s minimum, 99% CI. GenAI-Perf: 3 stable windows |
| One hardcoded prompt ("What is 2+2?") | GGUF=1 token, APR=16 tokens — 16x measurement artifact | vLLM: ShareGPT dataset. MLPerf: OpenORCA 5k samples |
| No warmup phase | Cold-start (model load, CUDA init) pollutes P95/P99 | vLLM: `--num-warmup`. TGI: `--warmup 30s`. GenAI-Perf: sliding window |
| `killall; sleep 15; curl health` lifecycle | Race conditions, orphaned processes, port leaks | Orchestrated setup/health-poll/teardown lifecycle |
| tok/s = total_tokens / wall_time | Conflates prefill + decode, rewards verbose output | TTFT (p50/p99) + TPOT (p50/p99) + system throughput as separate metrics |
| TTFB = full response time (non-streaming) | Cannot distinguish prefill from decode | Streaming SSE or server-reported timing |
| No regression detection | Manual eyeballing of numbers | Baseline comparison with threshold-based exit codes |
| No output validation during load test | Garbage responses counted as success | MLPerf: 99% accuracy gate |
| No resource monitoring | Cannot correlate throughput with GPU util/thermal | GenAI-Perf: DCGM telemetry. Optimum-Benchmark: PyNVML |
| Teardown has wrong process names | `pkill 'realizar serve'` doesn't match `apr serve` | Orchestrated cleanup |

## 2. Industry Standard Metrics

References:
- MLPerf Inference Rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- vLLM Benchmarks: https://docs.vllm.ai/en/stable/cli/bench/serve/
- NVIDIA GenAI-Perf: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html
- NVIDIA LLM Benchmarking Concepts: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/
- HuggingFace Inference-Benchmarker: https://github.com/huggingface/inference-benchmarker
- Metron Framework: https://arxiv.org/html/2407.07000v1

### 2.1 Required Metrics

| Metric | Formula | What It Measures |
|--------|---------|-----------------|
| **TTFT** | Time from request to first token | Prefill speed + queue wait. User-perceived responsiveness |
| **TPOT** | `(E2E - TTFT) / (output_tokens - 1)` | Per-token decode speed (request-weighted) |
| **ITL** | `(E2E - TTFT) / (output_tokens - 1)` (token-weighted) | Per-token decode speed (token-weighted, longer responses dominate) |
| **E2E Latency** | `TTFT + TPOT * (tokens - 1)` | Total request time |
| **System Throughput** | `total_output_tokens / wall_time` | Deployment capacity |
| **Goodput** | `requests_meeting_SLO / wall_time` | Useful throughput under constraints |

All metrics reported at: **p50, p90, p95, p99, min, max, stddev**.

### 2.2 MLPerf SLO Reference (Qwen-class 1.5B model)

Extrapolated from MLPerf Llama 3.1 8B targets, scaled for 1.5B:

| Scenario | p99 TTFT | p99 TPOT |
|----------|----------|----------|
| Interactive | 200ms | 20ms |
| Conversational | 500ms | 50ms |

### 2.3 Metrics We Currently Report vs. What We Need

```
CURRENT                          TARGET
────────                         ──────
Throughput (req/s)       ✓       Throughput (req/s)
Latency P50/P95/P99      ✓       E2E Latency P50/P90/P95/P99
TTFT P50                 ~       TTFT P50/P90/P95/P99 (streaming required)
Tokens/sec (total)       ✗       System TPS (output tokens)
Avg tok/req (GH-23)      ✓       Avg tok/req
ITL P50 (GH-23)          ~       TPOT P50/P90/P95/P99
Decode tok/s (GH-23)     ~       Decode TPS
                         ✗       Goodput (SLO-gated throughput)
                         ✗       Stddev / CI for all metrics
                         ✗       Min/Max latencies
                         ✗       Error rate
                         ✗       GPU utilization (avg/max)
                         ✗       GPU temperature (max)
                         ✗       VRAM usage
```

## 3. Prompt Normalization Strategy

### 3.1 Problem

Our single prompt "What is 2+2? Reply with just the number." produces:

| Backend | Completion tokens | Why |
|---------|------------------:|-----|
| apr GGUF | 1 | Correctly answers "4" and stops at EOS |
| llama.cpp | 2 | Answers "4\n" or "4." |
| ollama | 2 | Answers "4\n" or "4." |
| apr APR | 16 | Adds preamble before answering |

This makes tok/s incomparable (probar#23).

### 3.2 Solution: Prompt Profiles

Following HuggingFace Inference-Benchmarker's approach, define standardized prompt profiles:

| Profile | Input tokens | Output tokens | `max_tokens` | `ignore_eos` | Use Case |
|---------|-------------|---------------|-------------|-------------|----------|
| **micro** | ~10 | 1 | 1 | false | TTFT-only measurement (prefill speed) |
| **short** | ~32 | 32 | 32 | true | Quick latency check |
| **medium** | ~128 | 128 | 128 | true | Standard comparison (default) |
| **long** | ~512 | 256 | 256 | true | Sustained decode measurement |
| **sharegpt** | variable | variable | 256 | false | Realistic workload distribution |

**Critical: `ignore_eos=True`** for fixed-length profiles ensures all backends produce the same number of tokens. This is what MLPerf and vLLM do.

### 3.3 Prompt Dataset Files

```yaml
# prompts/medium.yaml
profile: medium
prompts:
  - role: user
    content: |
      Write a detailed explanation of how binary search works,
      including its time complexity, when to use it, and common
      pitfalls. Include a step-by-step example with the array
      [2, 5, 8, 12, 16, 23, 38, 56, 72, 91] searching for 23.
    max_tokens: 128
    temperature: 0.0
    ignore_eos: true
```

## 4. Benchmark Lifecycle

### 4.1 Current (Ad-Hoc)

```
Shell: killall apr
Shell: sleep 2
Shell: nohup apr serve run ... &
Shell: sleep 15
Shell: curl -sf /health  (single shot, proceed on failure)
Shell: probador llm load --duration 30s  (single run, single prompt)
Shell: killall apr
(repeat for next backend)
```

### 4.2 Target (Orchestrated)

```
Phase 1: SETUP
  ├── Port verification (lsof/ss check, fail if occupied)
  ├── Process start (nohup or systemd)
  ├── Health poll with backoff (2s interval, 120s timeout)
  └── Fail-fast if server doesn't become ready

Phase 2: WARMUP (excluded from measurement)
  ├── Send N warmup requests (default: 10)
  ├── Or warmup for T seconds (default: 10s)
  └── Discard all warmup metrics

Phase 3: MEASURE (repeated R times)
  ├── Run load test for D seconds (default: 60s, minimum: 30s)
  ├── Record per-request: latency, ttft, tokens, response text
  ├── Optionally collect GPU telemetry (nvidia-smi polling)
  ├── Cooldown C seconds between runs (default: 5s)
  └── Aggregate across R runs with 95% CI

Phase 4: VALIDATE
  ├── Compare against baseline (if provided)
  ├── Check SLO thresholds (if configured)
  ├── Report regression (exit 1 if threshold violated)
  └── Spot-check output coherence (optional)

Phase 5: TEARDOWN
  ├── Graceful shutdown (SIGTERM → wait 5s → SIGKILL)
  ├── Port release verification
  └── GPU memory cleanup verification
```

### 4.3 Single-Command Invocation

```bash
# Current: ~40 lines of shell per backend
killall apr; sleep 2; nohup apr serve run ... &; sleep 15; ...

# Target: one command
probador llm bench \
  --start "apr serve run model.gguf --port 8081 --gpu" \
  --url http://127.0.0.1:8081 \
  --model qwen \
  --prompt-profile medium \
  --warmup 10s \
  --duration 60s \
  --runs 3 \
  --cooldown 5s \
  --baseline results/baseline.json \
  --fail-on-regression 10 \
  --output results/apr-gguf-gpu.json
```

Or with forjar integration:

```bash
# Target: forjar drives the full lifecycle
forjar bench -f forjar-bench.yaml
```

## 5. Code Changes: probador (probar)

### 5.1 New CLI Subcommand: `probador llm bench`

Distinct from `probador llm load` (which is kept for backwards compatibility). `bench` is the orchestrated lifecycle.

**File:** `crates/probar-cli/src/commands.rs`

```rust
pub struct LlmBenchArgs {
    #[arg(long)]
    pub url: String,

    #[arg(long)]
    pub model: String,

    // Server lifecycle
    #[arg(long)]
    pub start: Option<String>,          // command to start server

    #[arg(long, default_value = "120s")]
    pub health_timeout: String,         // readiness poll timeout

    // Prompt control
    #[arg(long, default_value = "medium")]
    pub prompt_profile: String,         // micro|short|medium|long|sharegpt

    #[arg(long)]
    pub prompt_file: Option<PathBuf>,   // custom prompt dataset

    #[arg(long)]
    pub ignore_eos: bool,               // force fixed-length output

    // Measurement control
    #[arg(long, default_value = "10s")]
    pub warmup: String,                 // warmup duration

    #[arg(long, default_value = "60s")]
    pub duration: String,               // per-run duration

    #[arg(long, default_value = "1")]
    pub concurrency: usize,

    #[arg(long, default_value = "3")]
    pub runs: usize,                    // repeat N times

    #[arg(long, default_value = "5s")]
    pub cooldown: String,               // between runs

    // Analysis
    #[arg(long)]
    pub baseline: Option<PathBuf>,      // compare against

    #[arg(long)]
    pub fail_on_regression: Option<f64>, // % throughput drop threshold

    #[arg(long)]
    pub output: Option<PathBuf>,

    // Telemetry
    #[arg(long)]
    pub monitor_gpu: bool,              // nvidia-smi polling
}
```

### 5.2 New Module: `llm/benchmark.rs`

Orchestrates the full lifecycle:

```rust
pub struct Benchmark { client, config, server_process }

impl Benchmark {
    pub async fn run(&mut self) -> Result<BenchmarkReport> {
        self.start_server()?;               // Phase 1
        self.wait_ready().await?;
        self.warmup().await;                // Phase 2
        let runs = self.measure().await?;   // Phase 3
        let report = self.analyze(runs)?;   // Phase 4
        self.teardown()?;                   // Phase 5
        Ok(report)
    }
}

pub struct BenchmarkReport {
    pub runs: Vec<LoadTestResult>,
    pub aggregate: AggregateResult,       // mean, stddev, CI across runs
    pub regression: Option<Regression>,   // vs baseline
    pub gpu_telemetry: Option<GpuTimeSeries>,
}

pub struct AggregateResult {
    pub ttft_p50: StatSummary,    // { mean, stddev, ci_95_lower, ci_95_upper }
    pub ttft_p99: StatSummary,
    pub tpot_p50: StatSummary,
    pub tpot_p99: StatSummary,
    pub throughput_rps: StatSummary,
    pub tokens_per_sec: StatSummary,
    pub decode_tok_per_sec: StatSummary,
}
```

### 5.3 New Module: `llm/prompts.rs`

Prompt dataset loading and built-in profiles:

```rust
pub enum PromptProfile { Micro, Short, Medium, Long, ShareGpt }

pub fn load_prompts(profile: PromptProfile) -> Vec<ChatRequest>;
pub fn load_from_file(path: &Path) -> Result<Vec<ChatRequest>>;

// Built-in prompts are deterministic, temperature=0, with
// calibrated input length to hit target token counts
```

### 5.4 Enhanced `LoadTestResult` Fields

**File:** `crates/probar/src/llm/loadtest.rs`

Add to `LoadTestResult`:

| Field | Type | Purpose |
|-------|------|---------|
| `ttft_p90_ms` | f64 | TTFT P90 |
| `ttft_p95_ms` | f64 | TTFT P95 |
| `ttft_p99_ms` | f64 | TTFT P99 |
| `tpot_p50_ms` | f64 | Time per output token P50 |
| `tpot_p90_ms` | f64 | TPOT P90 |
| `tpot_p95_ms` | f64 | TPOT P95 |
| `tpot_p99_ms` | f64 | TPOT P99 |
| `itl_p90_ms` | f64 | Inter-token latency P90 |
| `itl_p95_ms` | f64 | ITL P95 |
| `itl_p99_ms` | f64 | ITL P99 |
| `latency_min_ms` | f64 | Minimum latency |
| `latency_max_ms` | f64 | Maximum latency |
| `latency_stddev_ms` | f64 | Standard deviation |
| `error_rate` | f64 | Failed / total |
| `prompt_tokens_total` | u64 | Total input tokens |
| `completion_tokens_total` | u64 | Total output tokens |

### 5.5 Enhanced `LoadTestConfig` Fields

**File:** `crates/probar/src/llm/loadtest.rs`

Add to `LoadTestConfig`:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `warmup_duration` | Duration | 0s | Warmup phase (excluded from metrics) |
| `warmup_requests` | usize | 0 | Alternative: N warmup requests |
| `ignore_eos` | bool | false | Force max_tokens output length |
| `max_tokens_override` | Option<u32> | None | Override prompt max_tokens |

### 5.6 Health Check Polling

**File:** `crates/probar/src/llm/client.rs`

New method:

```rust
pub async fn wait_ready(
    &self,
    timeout: Duration,
    poll_interval: Duration,
) -> Result<Duration, LlmClientError> {
    let start = Instant::now();
    loop {
        if start.elapsed() > timeout {
            return Err(LlmClientError::HealthCheckTimeout);
        }
        if let Ok(true) = self.health_check().await {
            return Ok(start.elapsed());
        }
        tokio::time::sleep(poll_interval).await;
    }
}
```

### 5.7 Fix: Use Actual Elapsed Time

**File:** `crates/probar/src/llm/loadtest.rs`, line 150

```rust
// BEFORE (uses configured duration — may not match actual)
let elapsed = self.config.duration.as_secs_f64();

// AFTER (uses actual wall time)
let actual_start = Instant::now();
// ... run workers ...
let elapsed = actual_start.elapsed().as_secs_f64();
```

### 5.8 Report: Baseline Comparison

**File:** `crates/probar/src/llm/report.rs`

```rust
pub struct Regression {
    pub metric: String,           // e.g. "throughput_rps"
    pub baseline_value: f64,
    pub current_value: f64,
    pub change_pct: f64,          // negative = regression
    pub exceeds_threshold: bool,
}

pub fn compare(
    current: &LoadTestResult,
    baseline: &LoadTestResult,
    threshold_pct: f64,
) -> Vec<Regression>;
```

## 6. Code Changes: forjar

### 6.1 Fix Teardown

**File:** `forjar-gpu-teardown.yaml`

- Change `pkill -f 'realizar serve.*PORT'` to `pkill -f 'apr serve.*PORT'`
- Add ports 8084, 8085, 8091-8095
- Add `fuser -k PORT/tcp` as fallback

### 6.2 New: `forjar-bench.yaml`

Benchmark-specific forjar manifest that orchestrates the full lifecycle:

```yaml
version: "1.0"
name: qwen-coder-bench
description: "Benchmark all contestants with standardized methodology"

params:
  duration: "60s"
  warmup: "10s"
  runs: "3"
  cooldown: "5s"
  prompt_profile: "medium"
  output_dir: "results/{{date}}"

resources:
  bench-apr-gguf-gpu:
    type: task
    machine: gpu
    command: |
      probador llm bench \
        --start "apr serve run {{params.gguf_model}} --port 8081 --gpu" \
        --url http://127.0.0.1:8081 \
        --model qwen \
        --prompt-profile {{params.prompt_profile}} \
        --warmup {{params.warmup}} \
        --duration {{params.duration}} \
        --runs {{params.runs}} \
        --cooldown {{params.cooldown}} \
        --output {{params.output_dir}}/apr-gguf-gpu.json
    depends_on: [model-gguf]
    # forjar runs these serially (no depends_on between bench resources)
    # to avoid GPU contention

  bench-ollama-gpu:
    type: task
    machine: gpu
    command: |
      probador llm bench \
        --start "OLLAMA_HOST=0.0.0.0:8082 ollama serve" \
        --url http://127.0.0.1:8082 \
        --model qwen2.5-coder:1.5b-instruct \
        --prompt-profile {{params.prompt_profile}} \
        --warmup {{params.warmup}} \
        --duration {{params.duration}} \
        --runs {{params.runs}} \
        --output {{params.output_dir}}/ollama-gpu.json
    depends_on: [bench-apr-gguf-gpu]

  # ... one resource per contestant, serialized via depends_on chain

  generate-report:
    type: task
    machine: gpu
    command: |
      probador llm report \
        --results {{params.output_dir}} \
        --output {{params.output_dir}}/performance.md \
        --baseline results/baseline/
    depends_on: [bench-apr-gguf-cpu]  # last benchmark
```

## 7. Implementation Order

| Phase | What | Files | Est. Effort |
|-------|------|-------|-------------|
| **1** | Prompt profiles + `--prompt-file` | `llm/prompts.rs`, `commands.rs`, `handlers/llm.rs` | S |
| **2** | Warmup phase in `LoadTest::run()` | `llm/loadtest.rs` | S |
| **3** | Fix elapsed time (actual vs config) | `llm/loadtest.rs:150` | XS |
| **4** | Health poll with timeout | `llm/client.rs` | S |
| **5** | Extended percentiles (p90/p95/p99 for TTFT, TPOT) | `llm/loadtest.rs` | S |
| **6** | Multi-run + aggregate + CI | `llm/benchmark.rs` | M |
| **7** | Baseline comparison + regression detection | `llm/report.rs` | S |
| **8** | `probador llm bench` CLI subcommand | `commands.rs`, `handlers/llm.rs` | M |
| **9** | Fix forjar teardown (process names, ports) | `forjar-gpu-teardown.yaml` | XS |
| **10** | `forjar-bench.yaml` manifest | `forjar-bench.yaml` | S |
| **11** | GPU telemetry (nvidia-smi polling) | `llm/telemetry.rs` | M |
| **12** | Output validation in load test | `llm/loadtest.rs` | S |

Phases 1-5 are the critical path. Phases 6-8 build on them. Phases 9-10 are forjar fixes. Phases 11-12 are nice-to-haves.

## 8. Falsification Tests

| ID | Assertion | Test |
|----|-----------|------|
| FALSIFY-BENCH-001 | Warmup requests excluded from metrics | Run with `--warmup 5s`, verify first request timestamp > warmup end |
| FALSIFY-BENCH-002 | `ignore_eos=true` normalizes output length | Run same prompt on all 5 backends with `ignore_eos=true --max-tokens 32`, assert all produce 32 tokens |
| FALSIFY-BENCH-003 | Multi-run CI narrows with more runs | Run N=1 vs N=5, assert CI width decreases |
| FALSIFY-BENCH-004 | Regression detection fires on 10% drop | Provide baseline with 100 tok/s, run at 85 tok/s, assert exit code 1 |
| FALSIFY-BENCH-005 | Health poll waits for slow servers | Start server with 20s startup, set `--health-timeout 30s`, assert benchmark completes |
| FALSIFY-BENCH-006 | Teardown kills correct process | Start `apr serve run`, invoke teardown, assert port is free |
| FALSIFY-BENCH-007 | Prompt profile "medium" produces ~128 output tokens | Run with `--prompt-profile medium --ignore-eos`, assert avg_tok_per_req in [120, 136] |

## 9. Success Criteria

The benchmarking methodology is "world-class" when:

1. **Reproducible**: Same benchmark run twice produces results within 5% (coefficient of variation < 0.05)
2. **Comparable**: All backends produce the same number of tokens for the same prompt profile
3. **Automated**: Single command runs full lifecycle (setup, warmup, measure, validate, teardown)
4. **Statistically valid**: 95% confidence intervals reported for all key metrics
5. **Regression-aware**: CI pipeline exits non-zero when throughput drops beyond threshold
6. **Observable**: GPU utilization, temperature, and VRAM tracked alongside latency metrics
