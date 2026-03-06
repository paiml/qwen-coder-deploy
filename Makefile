# ============================================================================
# qwen-coder-deploy — Benchmark realizar vs ollama vs llama.cpp
# ============================================================================
# Targets:
#   Jetson (serial):     make bench-jetson-serial (isolated, one runtime at a time)
#   Jetson (parallel):   make deploy-jetson / make load-jetson (smoke tests only)
#   GPU (4090, profiling only): make deploy-gpu / make nsys-gpu / make profile-gpu
#   CPU (intel host):    make deploy / make test / make load
#
# Load testing runs on Jetson Orin (dedicated). 4090 freed for QLoRA training.
# Deep profiling (nsys/ncu/apr profile) remains 4090-only (occasional).
#
# Deep profiling (apr/realizar tools, 4090 only):
#   make profile-gpu    — Roofline analysis + hotspot breakdown
#   make bench-gpu      — Per-brick timing with budget targets
#   make cbtop-gpu      — ComputeBrick pipeline profiler (headless)
#   make qa-gpu         — 10-gate falsifiable QA checklist
#   make trace-gpu      — Per-request brick/layer tracing
#   make realize-bench  — realizar internal benchmark suites
#   make gpu-util       — nvidia-smi GPU utilization snapshot
#
# NVIDIA Nsight profiling (kernel-level, 4090 only):
#   make install        — Install nsight-systems + nsight-compute via forjar
#   make nsys-gpu       — nsys timeline of GPU decode (per-kernel breakdown)
#   make ncu-gpu        — ncu roofline per GEMV kernel (bandwidth/compute)
#   make nsys-ollama    — nsys timeline of Ollama for A/B comparison
# ============================================================================

# Benchmarks MUST run sequentially — parallel execution causes GPU contention
# and corrupts throughput measurements.
.NOTPARALLEL:

DATE := $(shell date +%Y%m%d)

# --- Intel (CPU-only remote host) ---
INTEL_HOST := 192.168.50.100
INTEL_REALIZAR := http://$(INTEL_HOST):8081
INTEL_OLLAMA   := http://$(INTEL_HOST):8082
INTEL_LLAMACPP := http://$(INTEL_HOST):8083

# --- GPU (localhost, RTX 4090 — deep profiling only, 4090 runs QLoRA full-time) ---
GPU_HOST := 127.0.0.1
GPU_REALIZAR := http://$(GPU_HOST):8081
GPU_OLLAMA   := http://$(GPU_HOST):8082
GPU_LLAMACPP := http://$(GPU_HOST):8083

# --- Jetson Orin (dedicated load testing) ---
JETSON_HOST := 192.168.50.53
JETSON_REALIZAR   := http://$(JETSON_HOST):8081
JETSON_OLLAMA     := http://$(JETSON_HOST):8082
JETSON_LLAMACPP   := http://$(JETSON_HOST):8083
JETSON_APR_NATIVE := http://$(JETSON_HOST):8084

# Ollama requires exact model tag (not "default")
OLLAMA_MODEL := qwen2.5-coder:1.5b-instruct

GGUF_MODEL := /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# Qwen 1.5B transformer layers (for per-layer decode time comparison)
QWEN_LAYERS := 28

.PHONY: deploy teardown test load report nightly health \
        deploy-gpu teardown-gpu test-gpu load-gpu health-gpu nightly-gpu \
        deploy-jetson teardown-jetson test-jetson load-jetson health-jetson nightly-jetson \
        bench-jetson-serial bench-jetson-realizr bench-jetson-ollama bench-jetson-llamacpp \
        bench-gpu-serial bench-gpu-realizr bench-gpu-llamacpp \
        profile-gpu bench-gpu cbtop-gpu qa-gpu trace-gpu realize-bench \
        gpu-util full-gpu install \
        nsys-gpu ncu-gpu nsys-ollama nsys-llamacpp

# ============================================================================
# Intel (CPU) targets
# ============================================================================

deploy:
	forjar apply

teardown:
	forjar apply -f forjar-teardown.yaml

health:
	@echo "Checking realizar..."
	@curl -sf $(INTEL_REALIZAR)/health && echo " OK" || echo " FAIL"
	@echo "Checking ollama..."
	@curl -sf $(INTEL_OLLAMA)/health && echo " OK" || echo " FAIL"
	@echo "Checking llama.cpp..."
	@curl -sf $(INTEL_LLAMACPP)/health && echo " OK" || echo " FAIL"

test:
	probador llm test --config prompts/correctness.yaml --url $(INTEL_REALIZAR) --runtime-name realizar-cpu --output results/realizar-cpu-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(INTEL_OLLAMA) --model $(OLLAMA_MODEL) --runtime-name ollama-cpu --output results/ollama-cpu-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(INTEL_LLAMACPP) --runtime-name llamacpp-cpu --output results/llamacpp-cpu-correctness-$(DATE).json

load:
	probador llm load --url $(INTEL_REALIZAR) --concurrency 4 --duration 60s --runtime-name realizar-cpu --output results/realizar-cpu-load-$(DATE).json
	probador llm load --url $(INTEL_OLLAMA) --model $(OLLAMA_MODEL) --concurrency 4 --duration 60s --runtime-name ollama-cpu --output results/ollama-cpu-load-$(DATE).json
	probador llm load --url $(INTEL_LLAMACPP) --concurrency 4 --duration 60s --runtime-name llamacpp-cpu --output results/llamacpp-cpu-load-$(DATE).json

nightly: deploy health test load report

# ============================================================================
# GPU targets (localhost, RTX 4090)
# ============================================================================

deploy-gpu: teardown-gpu  ## Teardown first, then deploy to 4090
	forjar apply -f forjar-gpu.yaml

teardown-gpu:
	forjar apply -f forjar-gpu-teardown.yaml

health-gpu:
	@echo "Checking realizar (GPU)..."
	@curl -sf $(GPU_REALIZAR)/health && echo " OK" || echo " FAIL"
	@echo "Checking ollama (GPU)..."
	@curl -sf $(GPU_OLLAMA)/api/tags >/dev/null 2>&1 && echo " OK" || echo " FAIL"
	@echo "Checking llama.cpp (GPU)..."
	@curl -sf $(GPU_LLAMACPP)/health && echo " OK" || echo " FAIL"

test-gpu:
	probador llm test --config prompts/correctness.yaml --url $(GPU_REALIZAR) --runtime-name realizar-gpu --output results/realizar-gpu-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(GPU_OLLAMA) --model $(OLLAMA_MODEL) --runtime-name ollama-gpu --output results/ollama-gpu-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(GPU_LLAMACPP) --runtime-name llamacpp-gpu --output results/llamacpp-gpu-correctness-$(DATE).json

load-gpu:
	probador llm load --url $(GPU_REALIZAR) --concurrency 4 --duration 60s --runtime-name realizar-gpu --output results/realizar-gpu-load-$(DATE).json
	probador llm load --url $(GPU_OLLAMA) --model $(OLLAMA_MODEL) --concurrency 4 --duration 60s --runtime-name ollama-gpu --output results/ollama-gpu-load-$(DATE).json
	probador llm load --url $(GPU_LLAMACPP) --concurrency 4 --duration 60s --runtime-name llamacpp-gpu --output results/llamacpp-gpu-load-$(DATE).json

nightly-gpu: deploy-gpu health-gpu test-gpu load-gpu report

# ============================================================================
# GPU serial benchmarks (isolated — one runtime at a time, full GPU memory)
# ============================================================================
# Same methodology as Jetson serial benchmarks.
# --num-layers reports per-layer decode time (µs/layer) for cross-runtime comparison.
# This metric is overhead-free (derived from wall-clock ITL, not per-brick sync).

bench-gpu-realizr:
	@echo "=== teardown before realizr bench ==="
	-forjar apply -f forjar-gpu-teardown.yaml --yes
	@echo "=== realizr (isolated, CUDA) ==="
	forjar apply -f forjar-gpu-realizr.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(GPU_REALIZAR) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name realizr-4090-c1 \
		--output results/4090-serial-realizr-c1-$(DATE).json
	-forjar apply -f forjar-gpu-teardown.yaml --yes

bench-gpu-llamacpp:
	@echo "=== teardown before llama.cpp bench ==="
	-forjar apply -f forjar-gpu-teardown.yaml --yes
	@echo "=== llama.cpp (isolated) ==="
	forjar apply -f forjar-gpu-llamacpp.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(GPU_LLAMACPP) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name llamacpp-4090-c1 \
		--output results/4090-serial-llamacpp-c1-$(DATE).json
	-forjar apply -f forjar-gpu-teardown.yaml --yes

bench-gpu-serial: bench-gpu-realizr bench-gpu-llamacpp
	@echo ""
	@echo "=== 4090 Serial Benchmark Complete ==="
	@echo "Results in results/4090-serial-*-$(DATE).json"
	@echo "Compare per-layer decode time:"
	@jq '{runtime: .runtime_name, decode_tok_s: .decode_tok_per_sec, us_per_layer: .decode_us_per_layer, layers: .num_layers}' results/4090-serial-*-c1-$(DATE).json 2>/dev/null || true

# ============================================================================
# Jetson Orin targets (dedicated load testing — frees 4090 for QLoRA)
# ============================================================================
# Jetson Orin: aarch64, CUDA 12.6, 7.4 GB unified memory, JetPack R36.5
# All load testing runs here. 4090 only used for deep profiling (nsys/ncu).
#
# Full pipeline is forjar-managed: sync repos on Intel → cross-compile → deploy → start services
# See forjar-jetson.yaml for the declarative resource graph.

deploy-jetson: teardown-jetson  ## Teardown first, then build on Intel, deploy to Jetson, start all services
	forjar apply -f forjar-jetson.yaml --yes

teardown-jetson:
	forjar apply -f forjar-jetson-teardown.yaml

health-jetson:
	@echo "Checking realizar (Jetson)..."
	@curl -sf $(JETSON_REALIZAR)/health && echo " OK" || echo " FAIL"
	@echo "Checking ollama (Jetson)..."
	@curl -sf $(JETSON_OLLAMA)/api/tags >/dev/null 2>&1 && echo " OK" || echo " FAIL"
	@echo "Checking llama.cpp (Jetson)..."
	@curl -sf $(JETSON_LLAMACPP)/health && echo " OK" || echo " FAIL"
	@echo "Checking apr native (Jetson)..."
	@curl -sf $(JETSON_APR_NATIVE)/health && echo " OK" || echo " FAIL"

test-jetson:
	probador llm test --config prompts/correctness.yaml --url $(JETSON_REALIZAR) --runtime-name realizar-jetson --output results/realizar-jetson-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(JETSON_OLLAMA) --model $(OLLAMA_MODEL) --runtime-name ollama-jetson --output results/ollama-jetson-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(JETSON_LLAMACPP) --runtime-name llamacpp-jetson --output results/llamacpp-jetson-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(JETSON_APR_NATIVE) --runtime-name apr-native-jetson --output results/apr-native-jetson-correctness-$(DATE).json

load-jetson:
	probador llm load --url $(JETSON_REALIZAR) --concurrency 4 --duration 60s --warmup 5s --runtime-name realizar-jetson --output results/realizar-jetson-load-$(DATE).json
	probador llm load --url $(JETSON_OLLAMA) --model $(OLLAMA_MODEL) --concurrency 4 --duration 60s --warmup 5s --runtime-name ollama-jetson --output results/ollama-jetson-load-$(DATE).json
	probador llm load --url $(JETSON_LLAMACPP) --concurrency 4 --duration 60s --warmup 5s --runtime-name llamacpp-jetson --output results/llamacpp-jetson-load-$(DATE).json
	probador llm load --url $(JETSON_APR_NATIVE) --concurrency 4 --duration 60s --warmup 5s --runtime-name apr-native-jetson --output results/apr-native-jetson-load-$(DATE).json

nightly-jetson: deploy-jetson health-jetson test-jetson load-jetson report

# Quick local cross-compile + deploy to Jetson (skips forjar sync, uses local sources)
APR_CROSS_BIN := /tmp/cross-jetson/aarch64-unknown-linux-gnu/release/apr
APR_CROSS_FEATURES := hf-hub,safetensors-compare,inference,cuda,zram

quick-deploy-jetson:
	@echo "=== Cross-compiling apr-cli for aarch64 ==="
	cd ~/src/aprender && \
	CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc \
	CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc \
	RUSTFLAGS="-A unsafe-op-in-unsafe-fn" \
	cargo +nightly build --release \
		--target aarch64-unknown-linux-gnu \
		--target-dir /tmp/cross-jetson \
		-p apr-cli \
		--no-default-features \
		--features "$(APR_CROSS_FEATURES)"
	@echo "=== Stopping apr on Jetson ==="
	-ssh jetson 'pkill -f "apr serve" 2>/dev/null; sleep 2; true'
	@echo "=== Deploying binary ==="
	scp $(APR_CROSS_BIN) jetson:~/.cargo/bin/apr
	@echo "=== Starting apr with cuBLAS prefill ==="
	ssh jetson 'SKIP_PARITY_GATE=1 CUBLAS_PREFILL=1 REALIZR_FREE_CPU_WEIGHTS=1 REALIZR_MAX_SEQ_LEN=2048 nohup apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --gpu --host 0.0.0.0 --port 8081 --skip-contract > /tmp/apr-gguf-gpu.log 2>&1 & sleep 15; curl -sf http://127.0.0.1:8081/health >/dev/null && echo "HEALTHY" || echo "FAILED"'

# ============================================================================
# Jetson serial benchmarks (isolated — one runtime at a time, full GPU/memory)
# ============================================================================
# Jetson Orin has 7.4 GB UNIFIED memory shared between CPU and GPU.
# Running multiple servers simultaneously causes memory contention and
# invalidates benchmark results. Serial mode: stop all → start one → bench → stop.
#
# Usage:
#   make bench-jetson-serial                  # All 3 runtimes, c=1 and c=4
#   make bench-jetson-realizr                 # realizr only (isolated)
#   make bench-jetson-ollama                  # ollama only (isolated)
#   make bench-jetson-llamacpp                # llama.cpp only (isolated)

BENCH_DURATION := 60s
BENCH_WARMUP   := 5s
BENCH_PROFILE  := short

bench-jetson-realizr:
	@echo "=== teardown before realizr bench ==="
	forjar apply -f forjar-jetson-teardown.yaml --yes
	@echo "=== realizr (isolated) ==="
	forjar apply -f forjar-jetson-realizr.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(JETSON_REALIZAR) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--runtime-name realizr-jetson-isolated-c1 \
		--output results/jetson-serial-realizr-c1-$(DATE).json
	@echo "--- c=4 ---"
	probador llm load --url $(JETSON_REALIZAR) --concurrency 4 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--runtime-name realizr-jetson-isolated-c4 \
		--output results/jetson-serial-realizr-c4-$(DATE).json
	forjar apply -f forjar-jetson-teardown.yaml --yes

bench-jetson-ollama:
	@echo "=== teardown before ollama bench ==="
	forjar apply -f forjar-jetson-teardown.yaml --yes
	@echo "=== ollama (isolated) ==="
	forjar apply -f forjar-jetson-ollama.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(JETSON_OLLAMA) --model $(OLLAMA_MODEL) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--runtime-name ollama-jetson-isolated-c1 \
		--output results/jetson-serial-ollama-c1-$(DATE).json
	@echo "--- c=4 ---"
	probador llm load --url $(JETSON_OLLAMA) --model $(OLLAMA_MODEL) --concurrency 4 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--runtime-name ollama-jetson-isolated-c4 \
		--output results/jetson-serial-ollama-c4-$(DATE).json
	forjar apply -f forjar-jetson-teardown.yaml --yes

bench-jetson-llamacpp:
	@echo "=== teardown before llama.cpp bench ==="
	forjar apply -f forjar-jetson-teardown.yaml --yes
	@echo "=== llama.cpp (isolated) ==="
	forjar apply -f forjar-jetson-llamacpp.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(JETSON_LLAMACPP) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--runtime-name llamacpp-jetson-isolated-c1 \
		--output results/jetson-serial-llamacpp-c1-$(DATE).json
	@echo "--- c=4 ---"
	probador llm load --url $(JETSON_LLAMACPP) --concurrency 4 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--runtime-name llamacpp-jetson-isolated-c4 \
		--output results/jetson-serial-llamacpp-c4-$(DATE).json
	forjar apply -f forjar-jetson-teardown.yaml --yes

bench-jetson-serial: bench-jetson-realizr bench-jetson-ollama bench-jetson-llamacpp
	@echo ""
	@echo "=== Serial Benchmark Complete ==="
	@echo "Results in results/jetson-serial-*-$(DATE).json"
	@echo "Compare: jq '{runtime: .runtime_name, tok_s: .tokens_per_sec, decode: .decode_tok_per_sec, p50: .latency_p50_ms}' results/jetson-serial-*-c1-$(DATE).json"

# ============================================================================
# Deep profiling (apr + realizar tools, 4090 only)
# ============================================================================

# Roofline analysis: compute-bound vs memory-bound, hardware efficiency %,
# hotspot breakdown, performance grade (A-F), Ollama comparison
profile-gpu:
	apr profile $(GGUF_MODEL) --perf-grade --ollama --granular --json \
		--warmup 3 --measure 10 --tokens 32 \
		--output results/profile-gpu-$(DATE).json 2>&1 | tee results/profile-gpu-$(DATE).txt
	apr profile $(GGUF_MODEL) --perf-grade --granular --format flamegraph \
		--output results/flamegraph-gpu-$(DATE).svg

# Per-brick timing with budget targets (rms_norm 1.5µs, attn 10µs, ffn 12.2µs)
bench-gpu:
	apr bench $(GGUF_MODEL) --iterations 10 --warmup 3 --max-tokens 32 --json \
		2>&1 | tee results/bench-gpu-$(DATE).json
	@echo ""
	@echo "=== Brick-level breakdown ==="
	apr bench $(GGUF_MODEL) --brick rms_norm --json 2>&1 | tee results/bench-brick-rms_norm-$(DATE).json
	apr bench $(GGUF_MODEL) --brick attn --json 2>&1 | tee results/bench-brick-attn-$(DATE).json
	apr bench $(GGUF_MODEL) --brick ffn --json 2>&1 | tee results/bench-brick-ffn-$(DATE).json

# ComputeBrick pipeline profiler — per-brick timing across all layers (headless)
cbtop-gpu:
	apr cbtop --model-path $(GGUF_MODEL) --headless --json \
		--warmup 10 --iterations 100 \
		--output results/cbtop-gpu-$(DATE).json 2>&1 | tee results/cbtop-gpu-$(DATE).txt

# 10-gate falsifiable QA checklist: golden output, throughput floor,
# Ollama parity, GPU vs CPU speedup, cross-format parity, tensor contracts
qa-gpu:
	apr qa $(GGUF_MODEL) --verbose --json \
		--iterations 10 --warmup 3 --max-tokens 32 \
		2>&1 | tee results/qa-gpu-$(DATE).json

# Per-request tracing via X-Trace-Level headers (requires realizar running on 8081)
trace-gpu:
	@echo "=== Brick-level trace (token operations) ==="
	curl -s -X POST $(GPU_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "X-Trace-Level: brick" \
		-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}' | python3 -m json.tool
	@echo ""
	@echo "=== Layer-level trace (per-layer timing) ==="
	curl -s -X POST $(GPU_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "X-Trace-Level: layer" \
		-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}' | python3 -m json.tool

# realizar internal benchmark suites (tensor ops, inference, cache, tokenizer, quantize)
realize-bench:
	realizar bench --runtime realizar --model $(GGUF_MODEL) --output results/realize-bench-$(DATE).json
	realizar bench --runtime ollama --url $(GPU_OLLAMA) --model $(OLLAMA_MODEL) --output results/realize-bench-ollama-$(DATE).json
	realizar bench --runtime llama-cpp --url $(GPU_LLAMACPP) --output results/realize-bench-llamacpp-$(DATE).json

# GPU utilization snapshot (memory, compute, power, clocks)
gpu-util:
	@nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.current.graphics --format=csv

# Full profiling pipeline: deploy, load test, then deep profile
full-gpu: deploy-gpu health-gpu load-gpu profile-gpu bench-gpu cbtop-gpu qa-gpu report

# ============================================================================
# Install (forjar-managed tooling)
# ============================================================================

install:
	forjar apply -f forjar-gpu.yaml --resource nsight-tools

# ============================================================================
# NVIDIA Nsight profiling (kernel-level GPU analysis)
# ============================================================================
# Requires: make install (nsight-systems + nsight-compute)
# These profile the apr serve process directly — NOT via HTTP.
# Start server first with make deploy-gpu, then attach.

NSYS_OPTS := --trace=cuda,nvtx --cuda-graph-trace=node --force-overwrite=true --export=sqlite
NCU_OPTS  := --set=full --force-overwrite

# nsys timeline: captures ALL CUDA kernels in a 5-second window during decode.
# Shows per-kernel duration, gaps between kernels, H2D/D2H transfers, graph replay.
# Output: results/nsys-apr-gpu-YYYYMMDD.nsys-rep (open with nsys-ui or nsys stats)
nsys-gpu:
	@echo "=== nsys: Profiling apr serve (GPU) for 5s ==="
	@echo "Sending warmup request..."
	@curl -sf -X POST $(GPU_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' > /dev/null
	@APR_PID=$$(pgrep -f 'apr serve.*8081' | head -1); \
	if [ -z "$$APR_PID" ]; then echo "ERROR: apr serve not running on 8081"; exit 1; fi; \
	echo "Attaching nsys to PID $$APR_PID..."; \
	nsys profile $(NSYS_OPTS) --duration=5 --output=results/nsys-apr-gpu-$(DATE) \
		--attach-to=$$APR_PID &; \
	sleep 1; \
	echo "Sending inference request during capture..."; \
	curl -sf -X POST $(GPU_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"default","messages":[{"role":"user","content":"Write a Rust function that checks if a number is prime."}],"max_tokens":128}' > /dev/null; \
	wait; \
	echo "=== nsys stats ===" ; \
	nsys stats --report cuda_gpu_kern_sum results/nsys-apr-gpu-$(DATE).nsys-rep 2>&1 | tee results/nsys-apr-gpu-kernels-$(DATE).txt

# ncu roofline: profiles individual GEMV kernel launches with full metrics.
# CANNOT profile inside CUDA graphs — must disable graph capture.
# Use CUDA_GRAPH=0 env var on the apr serve process.
# Output: results/ncu-apr-gpu-YYYYMMDD.ncu-rep (open with ncu-ui)
ncu-gpu:
	@echo "=== ncu: Per-kernel roofline (CUDA graph DISABLED) ==="
	@echo "NOTE: Restart apr serve with CUDA_GRAPH=0 for ncu profiling:"
	@echo "  CUDA_GRAPH=0 apr serve run $(GGUF_MODEL) --port 8081 --gpu"
	@APR_PID=$$(pgrep -f 'apr serve.*8081' | head -1); \
	if [ -z "$$APR_PID" ]; then echo "ERROR: apr serve not running on 8081"; exit 1; fi; \
	echo "Profiling 1 inference request (this is slow ~60s)..."; \
	ncu $(NCU_OPTS) --output=results/ncu-apr-gpu-$(DATE) \
		--target-processes=all --replay-mode=kernel \
		--launch-count=420 \
		-- curl -sf -X POST $(GPU_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}' > /dev/null; \
	echo "=== ncu summary ===" ; \
	ncu --import results/ncu-apr-gpu-$(DATE).ncu-rep --csv 2>&1 | head -50

# nsys timeline for Ollama (A/B comparison against apr)
nsys-ollama:
	@echo "=== nsys: Profiling Ollama (GPU) for 5s ==="
	@curl -sf -X POST $(GPU_OLLAMA)/api/generate \
		-d '{"model":"$(OLLAMA_MODEL)","prompt":"Hello","stream":false}' > /dev/null
	@OLLAMA_PID=$$(pgrep -f 'ollama.*serve' | head -1); \
	if [ -z "$$OLLAMA_PID" ]; then echo "ERROR: ollama not running"; exit 1; fi; \
	echo "Attaching nsys to PID $$OLLAMA_PID..."; \
	nsys profile $(NSYS_OPTS) --duration=5 --output=results/nsys-ollama-gpu-$(DATE) \
		--attach-to=$$OLLAMA_PID &; \
	sleep 1; \
	curl -sf -X POST $(GPU_OLLAMA)/api/generate \
		-d '{"model":"$(OLLAMA_MODEL)","prompt":"Write a Rust function that checks if a number is prime.","stream":false}' > /dev/null; \
	wait; \
	echo "=== nsys stats ===" ; \
	nsys stats --report cuda_gpu_kern_sum results/nsys-ollama-gpu-$(DATE).nsys-rep 2>&1 | tee results/nsys-ollama-gpu-kernels-$(DATE).txt

# nsys timeline for llama.cpp (A/B comparison)
nsys-llamacpp:
	@echo "=== nsys: Profiling llama.cpp (GPU) for 5s ==="
	@curl -sf -X POST $(GPU_LLAMACPP)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"default","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}' > /dev/null
	@LCPP_PID=$$(pgrep -f 'llama-server.*8083' | head -1); \
	if [ -z "$$LCPP_PID" ]; then echo "ERROR: llama-server not running on 8083"; exit 1; fi; \
	echo "Attaching nsys to PID $$LCPP_PID..."; \
	nsys profile $(NSYS_OPTS) --duration=5 --output=results/nsys-llamacpp-gpu-$(DATE) \
		--attach-to=$$LCPP_PID &; \
	sleep 1; \
	curl -sf -X POST $(GPU_LLAMACPP)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"default","messages":[{"role":"user","content":"Write a Rust function that checks if a number is prime."}],"max_tokens":128}' > /dev/null; \
	wait; \
	echo "=== nsys stats ===" ; \
	nsys stats --report cuda_gpu_kern_sum results/nsys-llamacpp-gpu-$(DATE).nsys-rep 2>&1 | tee results/nsys-llamacpp-gpu-kernels-$(DATE).txt

# ncu on Jetson: per-kernel bandwidth, register usage, occupancy
# Requires CUDA_GRAPH=0 (ncu can't profile inside CUDA graphs)
# Profiles a single decode inference request (~16 tokens)
ncu-jetson:
	@echo "=== Restarting apr on Jetson with CUDA_GRAPH=0 ==="
	-ssh jetson 'pkill -f "apr serve" 2>/dev/null; sleep 2; true'
	ssh jetson 'SKIP_PARITY_GATE=1 CUDA_GRAPH=0 nohup /home/noah/.cargo/bin/apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --gpu --host 0.0.0.0 --port 8081 --skip-contract > /tmp/apr-ncu.log 2>&1 & sleep 15; curl -sf http://127.0.0.1:8081/health >/dev/null && echo "HEALTHY" || (cat /tmp/apr-ncu.log | tail -20; echo "FAILED"; exit 1)'
	@echo "=== Warmup ==="
	@curl -sf -X POST $(JETSON_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-d '{"model":"default","messages":[{"role":"user","content":"Hi"}],"max_tokens":8}' > /dev/null
	@echo "=== ncu profiling (single request, ~60s) ==="
	ssh jetson 'ncu --set=roofline --kernel-name "mwv_dp4a_q4k_gemv|q6k_gemv|multi_warp" --launch-count 50 --target-processes all --force-overwrite -o /tmp/ncu-jetson-$(DATE) -- curl -sf -X POST http://127.0.0.1:8081/v1/chat/completions -H "Content-Type: application/json" -d '"'"'{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}'"'"' > /dev/null 2>&1'
	scp jetson:/tmp/ncu-jetson-$(DATE).ncu-rep results/ncu-jetson-$(DATE).ncu-rep
	@echo "=== Results ==="
	ssh jetson 'ncu --import /tmp/ncu-jetson-$(DATE).ncu-rep --csv --page raw 2>&1 | head -100'
	@echo ""
	@echo "=== Restarting apr normally (with CUDA graphs) ==="
	-ssh jetson 'pkill -f "apr serve" 2>/dev/null; sleep 2; true'
	ssh jetson 'SKIP_PARITY_GATE=1 CUBLAS_PREFILL=1 REALIZR_FREE_CPU_WEIGHTS=1 REALIZR_MAX_SEQ_LEN=2048 nohup /home/noah/.cargo/bin/apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --gpu --host 0.0.0.0 --port 8081 --skip-contract > /tmp/apr-gguf-gpu.log 2>&1 & sleep 15; curl -sf http://127.0.0.1:8081/health >/dev/null && echo "HEALTHY" || echo "FAILED"'

# BrickProfiler + nsys combined: run both for cross-validation
# BrickProfiler gives per-operation CPU-side timing (via --trace)
# nsys gives actual GPU kernel execution time (async, more accurate)
profile-kernels-gpu: nsys-gpu
	@echo ""
	@echo "=== BrickProfiler trace (CPU-side timing) ==="
	curl -sf -X POST $(GPU_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "X-Trace-Level: brick" \
		-d '{"model":"default","messages":[{"role":"user","content":"Write a Rust function that checks if a number is prime."}],"max_tokens":128}' | python3 -m json.tool
	@echo ""
	@echo "Compare: results/nsys-apr-gpu-kernels-$(DATE).txt (GPU kernel time)"
	@echo "     vs: BrickProfiler output above (CPU-side time including launch overhead)"

# ============================================================================
# Shared targets
# ============================================================================

report:
	probador llm report --results results/ --output performance.md --update-readme README.md
