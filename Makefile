# ============================================================================
# qwen-coder-deploy — Benchmark realizar vs ollama vs llama.cpp
# ============================================================================
# Targets:
#   Jetson (serial):     make bench-jetson-serial (isolated, one runtime at a time)
#   Jetson (parallel):   make deploy-jetson / make load-jetson (smoke tests only)
#   GPU (4090, profiling only): make deploy-gpu / make nsys-gpu / make profile-gpu
#   CPU (intel host):    make deploy / make test / make load
#   Scoring:             make score / make score-prod / make score-jetson / make score-gate
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

# --- Intel (CPU + WGPU remote host) ---
INTEL_HOST := 192.168.50.100
INTEL_REALIZAR := http://$(INTEL_HOST):8081
INTEL_OLLAMA   := http://$(INTEL_HOST):8082
INTEL_LLAMACPP := http://$(INTEL_HOST):8083
INTEL_WGPU     := http://$(INTEL_HOST):8081

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

# --- Yoga (primary benchmark target, RTX 4060 Laptop) ---
YOGA_HOST := 192.168.50.38
# NOTE: Use IP not hostname — yoga only resolves via SSH config, not DNS
YOGA_REALIZAR := http://$(YOGA_HOST):8081
YOGA_OLLAMA   := http://$(YOGA_HOST):8082
YOGA_LLAMACPP := http://$(YOGA_HOST):8083
YOGA_VLLM     := http://$(YOGA_HOST):8084

# Ollama requires exact model tag (not "default")
OLLAMA_MODEL := qwen2.5-coder:1.5b-instruct

# vLLM uses AWQ INT4 model (not GGUF — poor perf in vLLM)
VLLM_MODEL := Qwen/Qwen2.5-Coder-1.5B-Instruct-AWQ

GGUF_MODEL := /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

# Qwen 1.5B transformer layers (for per-layer decode time comparison)
QWEN_LAYERS := 28

.PHONY: deploy teardown test load report nightly health \
        deploy-gpu teardown-gpu test-gpu load-gpu health-gpu nightly-gpu \
        deploy-jetson teardown-jetson test-jetson load-jetson health-jetson nightly-jetson \
        bench-jetson-serial bench-jetson-realizr bench-jetson-ollama bench-jetson-llamacpp \
        bench-gpu-serial bench-gpu-realizr bench-gpu-llamacpp \
        deploy-yoga-realizr deploy-yoga-llamacpp deploy-yoga-ollama deploy-yoga-vllm teardown-yoga health-yoga \
        bench-yoga-realizr bench-yoga-llamacpp bench-yoga-ollama bench-yoga-vllm bench-yoga-serial \
        bench-yoga-prod bench-yoga-prod-realizr bench-yoga-prod-llamacpp bench-yoga-prod-ollama bench-yoga-prod-vllm \
        profile-gpu bench-gpu cbtop-gpu qa-gpu trace-gpu realize-bench \
        gpu-util full-gpu install \
        nsys-gpu ncu-gpu nsys-ollama nsys-llamacpp \
        profile-yoga profile-yoga-ci profile-yoga-trace profile-yoga-compare profile-yoga-full \
        build-wgpu deploy-wgpu start-wgpu test-wgpu stop-wgpu \
        score score-prod score-all score-json score-jetson score-gate \
        contract-lint contract-validate contract-score contract-falsify

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
# Intel WGPU targets (Radeon Pro W5700X, Vulkan)
# ============================================================================

WGPU_MODEL := /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

build-wgpu:  ## Build apr with WGPU feature
	cd ~/src/aprender && CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/aprender \
		cargo build --release -p apr-cli --bin apr \
		--features "apr-cli/inference,apr-cli/wgpu" --no-default-features

deploy-wgpu: build-wgpu  ## Deploy WGPU binary to intel
	ssh intel 'pkill -f "backend wgpu" 2>/dev/null; sleep 2; true'
	scp /mnt/nvme-raid0/targets/aprender/release/apr intel:~/.cargo/bin/apr

start-wgpu: deploy-wgpu  ## Start WGPU server on intel
	ssh intel 'nohup apr serve run $(WGPU_MODEL) --backend wgpu --port 8081 --host 0.0.0.0 > /tmp/wgpu-serve.log 2>&1 &'
	@echo "Waiting for WGPU startup (model dequant + GPU upload)..."
	@sleep 18
	@curl -sf $(INTEL_WGPU)/health >/dev/null && echo "WGPU ready" || echo "WGPU not ready — check /tmp/wgpu-serve.log"

test-wgpu:  ## Correctness test on WGPU endpoint
	@echo "=== WGPU Correctness ==="
	@for prompt in "What is 2+2?" "Capital of France?" "Write hello in Python"; do \
		echo -n "  \"$$prompt\": "; \
		curl -s --max-time 120 $(INTEL_WGPU)/v1/chat/completions \
			-H 'Content-Type: application/json' \
			-d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"$$prompt\"}],\"max_tokens\":32}" \
			| python3 -c "import json,sys; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'])" 2>/dev/null || echo "FAIL"; \
	done

stop-wgpu:  ## Stop WGPU server on intel
	ssh intel 'pkill -f "backend wgpu" 2>/dev/null; true'

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
	@echo "=== Starting apr ==="
	ssh jetson 'SKIP_PARITY_GATE=1 nohup apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --gpu --host 0.0.0.0 --port 8081 --skip-contract > /tmp/apr-gguf-gpu.log 2>&1 & sleep 15; curl -sf http://127.0.0.1:8081/health >/dev/null && echo "HEALTHY" || echo "FAILED"'

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
	ssh jetson 'SKIP_PARITY_GATE=1 nohup /home/noah/.cargo/bin/apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --gpu --host 0.0.0.0 --port 8081 --skip-contract > /tmp/apr-gguf-gpu.log 2>&1 & sleep 15; curl -sf http://127.0.0.1:8081/health >/dev/null && echo "HEALTHY" || echo "FAILED"'

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
# Yoga targets (PRIMARY benchmark target — RTX 4060 Laptop, ssh yoga)
# ============================================================================
# Spec: docs/specifications/perf-parity-spec.md
# Method: forjar setup/teardown, one runtime at a time, locked clocks
#
# Usage:
#   make bench-yoga-serial                  # All 3 runtimes, c=1 and c=4
#   make bench-yoga-realizr                 # realizr only (isolated)
#   make bench-yoga-llamacpp                # llama.cpp only (isolated)
#   make bench-yoga-ollama                  # ollama only (isolated)

deploy-yoga-realizr:
	forjar apply -f forjar-yoga-realizr.yaml --yes

deploy-yoga-llamacpp:
	forjar apply -f forjar-yoga-llamacpp.yaml --yes

deploy-yoga-ollama:
	forjar apply -f forjar-yoga-ollama.yaml --yes

deploy-yoga-vllm:
	forjar apply -f forjar-yoga-vllm.yaml --yes

teardown-yoga:
	forjar apply -f forjar-yoga-teardown.yaml --yes

health-yoga:
	@echo "Checking realizr (yoga)..."
	@curl -sf $(YOGA_REALIZAR)/health && echo " OK" || echo " FAIL"
	@echo "Checking ollama (yoga)..."
	@curl -sf $(YOGA_OLLAMA)/api/tags >/dev/null 2>&1 && echo " OK" || echo " FAIL"
	@echo "Checking llama.cpp (yoga)..."
	@curl -sf $(YOGA_LLAMACPP)/health && echo " OK" || echo " FAIL"
	@echo "Checking vLLM (yoga)..."
	@curl -sf $(YOGA_VLLM)/v1/models >/dev/null 2>&1 && echo " OK" || echo " FAIL"

bench-yoga-realizr:
	@echo "=== teardown before realizr bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== realizr (isolated, yoga) ==="
	forjar apply -f forjar-yoga-realizr.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(YOGA_REALIZAR) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name realizr-yoga-c1 \
		--output results/yoga-serial-realizr-c1-$(DATE).json
	@echo "--- c=4 ---"
	probador llm load --url $(YOGA_REALIZAR) --concurrency 4 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name realizr-yoga-c4 \
		--output results/yoga-serial-realizr-c4-$(DATE).json
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-llamacpp:
	@echo "=== teardown before llama.cpp bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== llama.cpp (isolated, yoga) ==="
	forjar apply -f forjar-yoga-llamacpp.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(YOGA_LLAMACPP) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name llamacpp-yoga-c1 \
		--output results/yoga-serial-llamacpp-c1-$(DATE).json
	@echo "--- c=4 ---"
	probador llm load --url $(YOGA_LLAMACPP) --concurrency 4 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name llamacpp-yoga-c4 \
		--output results/yoga-serial-llamacpp-c4-$(DATE).json
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-ollama:
	@echo "=== teardown before ollama bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== ollama (isolated, yoga) ==="
	forjar apply -f forjar-yoga-ollama.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(YOGA_OLLAMA) --model $(OLLAMA_MODEL) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name ollama-yoga-c1 \
		--output results/yoga-serial-ollama-c1-$(DATE).json
	@echo "--- c=4 ---"
	probador llm load --url $(YOGA_OLLAMA) --model $(OLLAMA_MODEL) --concurrency 4 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name ollama-yoga-c4 \
		--output results/yoga-serial-ollama-c4-$(DATE).json
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-vllm:
	@echo "=== teardown before vLLM bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== vLLM (isolated, yoga) ==="
	forjar apply -f forjar-yoga-vllm.yaml --yes --force
	@echo "--- c=1 ---"
	probador llm load --url $(YOGA_VLLM) --model $(VLLM_MODEL) --concurrency 1 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name vllm-yoga-c1 \
		--output results/yoga-serial-vllm-c1-$(DATE).json
	@echo "--- c=4 ---"
	probador llm load --url $(YOGA_VLLM) --model $(VLLM_MODEL) --concurrency 4 \
		--duration $(BENCH_DURATION) --warmup $(BENCH_WARMUP) --prompt-profile $(BENCH_PROFILE) \
		--stream true \
		--num-layers $(QWEN_LAYERS) \
		--runtime-name vllm-yoga-c4 \
		--output results/yoga-serial-vllm-c4-$(DATE).json
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-serial: bench-yoga-realizr bench-yoga-llamacpp bench-yoga-ollama bench-yoga-vllm
	@echo ""
	@echo "=== Yoga Serial Benchmark Complete ==="
	@echo "Results in results/yoga-serial-*-$(DATE).json"
	@echo "Compare c=1 decode:"
	@jq '{runtime: .runtime_name, decode_tok_s: .decode_tok_per_sec, ttft_p50_ms: .ttft_p50_ms, itl_p50_ms: .itl_p50_ms}' results/yoga-serial-*-c1-$(DATE).json 2>/dev/null || true

# ============================================================================
# Yoga production-realistic benchmarks (PMAT-157+ methodology)
# ============================================================================
# Medium prompt (~102 tok) + heterogeneous output (uniform:16,256) + streaming
# realizr/vLLM: c=1,4,8,16,32,64,128. llama.cpp: c=1,4,8,16,32. ollama: c=1,4,8,16,32.
# Results feed directly into probador llm score and gap decomposition model
#
# Usage:
#   make bench-yoga-prod                  # All 4 runtimes (realizr, llama.cpp, ollama, vLLM)
#   make bench-yoga-prod-realizr          # realizr only (isolated)
#   make bench-yoga-prod-llamacpp         # llama.cpp only (isolated)
#   make bench-yoga-prod-ollama           # ollama only (isolated)
#   make bench-yoga-prod-vllm             # vLLM only (isolated)

PROD_DURATION := 60s
PROD_WARMUP   := 5s
PROD_PROFILE  := medium
PROD_OUTPUT   := --max-tokens-distribution uniform:16,256

define run-prod-bench
	@echo "--- $(1) c=$(2) ---"
	probador llm load --url $(3) $(4) --concurrency $(2) \
		--duration $(PROD_DURATION) --warmup $(PROD_WARMUP) --prompt-profile $(PROD_PROFILE) \
		$(PROD_OUTPUT) --stream true \
		--runtime-name $(1) \
		--output results/$(1)-yoga-prod-c$(2)-$(DATE).json
endef

bench-yoga-prod-realizr:
	@echo "=== teardown before realizr prod bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== realizr production benchmark (yoga) ==="
	forjar apply -f forjar-yoga-realizr.yaml --yes --force
	$(call run-prod-bench,realizr,1,$(YOGA_REALIZAR),)
	$(call run-prod-bench,realizr,4,$(YOGA_REALIZAR),)
	$(call run-prod-bench,realizr,8,$(YOGA_REALIZAR),)
	$(call run-prod-bench,realizr,16,$(YOGA_REALIZAR),)
	$(call run-prod-bench,realizr,32,$(YOGA_REALIZAR),)
	$(call run-prod-bench,realizr,64,$(YOGA_REALIZAR),)
	$(call run-prod-bench,realizr,128,$(YOGA_REALIZAR),)
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-prod-llamacpp:
	@echo "=== teardown before llama.cpp prod bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== llama.cpp production benchmark (yoga) ==="
	forjar apply -f forjar-yoga-llamacpp.yaml --yes --force
	$(call run-prod-bench,llamacpp,1,$(YOGA_LLAMACPP),)
	$(call run-prod-bench,llamacpp,4,$(YOGA_LLAMACPP),)
	$(call run-prod-bench,llamacpp,8,$(YOGA_LLAMACPP),)
	$(call run-prod-bench,llamacpp,16,$(YOGA_LLAMACPP),)
	$(call run-prod-bench,llamacpp,32,$(YOGA_LLAMACPP),)
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-prod-vllm:
	@echo "=== teardown before vLLM prod bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== vLLM production benchmark (yoga) ==="
	forjar apply -f forjar-yoga-vllm.yaml --yes --force
	$(call run-prod-bench,vllm,1,$(YOGA_VLLM),--model $(VLLM_MODEL))
	$(call run-prod-bench,vllm,4,$(YOGA_VLLM),--model $(VLLM_MODEL))
	$(call run-prod-bench,vllm,8,$(YOGA_VLLM),--model $(VLLM_MODEL))
	$(call run-prod-bench,vllm,16,$(YOGA_VLLM),--model $(VLLM_MODEL))
	$(call run-prod-bench,vllm,32,$(YOGA_VLLM),--model $(VLLM_MODEL))
	$(call run-prod-bench,vllm,64,$(YOGA_VLLM),--model $(VLLM_MODEL))
	$(call run-prod-bench,vllm,128,$(YOGA_VLLM),--model $(VLLM_MODEL))
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-prod-ollama:
	@echo "=== teardown before ollama prod bench ==="
	-forjar apply -f forjar-yoga-teardown.yaml --yes
	@echo "=== ollama production benchmark (yoga) ==="
	forjar apply -f forjar-yoga-ollama.yaml --yes --force
	$(call run-prod-bench,ollama,1,$(YOGA_OLLAMA),--model $(OLLAMA_MODEL))
	$(call run-prod-bench,ollama,4,$(YOGA_OLLAMA),--model $(OLLAMA_MODEL))
	$(call run-prod-bench,ollama,8,$(YOGA_OLLAMA),--model $(OLLAMA_MODEL))
	$(call run-prod-bench,ollama,16,$(YOGA_OLLAMA),--model $(OLLAMA_MODEL))
	$(call run-prod-bench,ollama,32,$(YOGA_OLLAMA),--model $(OLLAMA_MODEL))
	-forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-prod: bench-yoga-prod-realizr bench-yoga-prod-llamacpp bench-yoga-prod-ollama bench-yoga-prod-vllm
	@echo ""
	@echo "=== Yoga Production Benchmark Complete ==="
	@echo "Results in results/*-yoga-prod-*-$(DATE).json"
	@echo ""
	@echo "Score with: probador llm score --results results/ --platform yoga"

# ============================================================================
# Yoga profiling (apr profile — internal roofline + hotspot analysis)
# ============================================================================
# Run ON yoga via SSH (apr profile needs direct GPU access, not HTTP)
# Complements probador (external latency) with internal bottleneck analysis

YOGA_GGUF := /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf
PROFILE_WARMUP := 3
PROFILE_MEASURE := 10
PROFILE_TOKENS := 32

profile-yoga: ## Roofline + hotspots + perf grade on yoga (requires realizr deployed)
	@echo "=== apr profile (yoga, roofline + hotspots) ==="
	ssh yoga 'apr profile $(YOGA_GGUF) \
		--perf-grade --granular --json \
		--warmup $(PROFILE_WARMUP) --measure $(PROFILE_MEASURE) --tokens $(PROFILE_TOKENS) \
		--output /tmp/profile-yoga-$(DATE).json' 2>&1 | tee results/profile-yoga-$(DATE).txt
	scp yoga:/tmp/profile-yoga-$(DATE).json results/profile-yoga-$(DATE).json 2>/dev/null || true
	@echo "=== flamegraph ==="
	ssh yoga 'apr profile $(YOGA_GGUF) --format flamegraph \
		--warmup $(PROFILE_WARMUP) --measure $(PROFILE_MEASURE) --tokens $(PROFILE_TOKENS) \
		--output /tmp/flamegraph-yoga-$(DATE).svg'
	scp yoga:/tmp/flamegraph-yoga-$(DATE).svg results/flamegraph-yoga-$(DATE).svg 2>/dev/null || true
	@echo "Results: results/profile-yoga-$(DATE).json, results/flamegraph-yoga-$(DATE).svg"

profile-yoga-ci: ## CI assertion mode — fail if below thresholds
	@echo "=== apr profile CI gate (yoga) ==="
	ssh yoga 'apr profile $(YOGA_GGUF) \
		--ci --assert-throughput 130 --assert-p99 50 \
		--warmup $(PROFILE_WARMUP) --measure $(PROFILE_MEASURE) --tokens $(PROFILE_TOKENS)'

profile-yoga-trace: ## Live brick trace via X-Trace-Level header (requires realizr running)
	@echo "=== Brick-level trace ==="
	@curl -s -X POST $(YOGA_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "X-Trace-Level: brick" \
		-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}' | python3 -m json.tool
	@echo ""
	@echo "=== Layer-level trace ==="
	@curl -s -X POST $(YOGA_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "X-Trace-Level: layer" \
		-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":16}' | python3 -m json.tool

profile-yoga-compare: ## Profile + compare against ollama baseline
	@echo "=== apr profile with ollama comparison ==="
	ssh yoga 'apr profile $(YOGA_GGUF) \
		--perf-grade --granular --ollama --json \
		--warmup $(PROFILE_WARMUP) --measure $(PROFILE_MEASURE) --tokens $(PROFILE_TOKENS)' \
		2>&1 | tee results/profile-yoga-compare-$(DATE).txt

profile-yoga-vs-llamacpp: ## Profile + compare against llama.cpp baseline (PMAT-056)
	@echo "=== apr profile with llama.cpp comparison ==="
	ssh yoga 'apr profile $(YOGA_GGUF) \
		--perf-grade --granular \
		--baseline-url http://127.0.0.1:8083 --baseline-model default --baseline-name llama.cpp \
		--warmup $(PROFILE_WARMUP) --measure $(PROFILE_MEASURE) --tokens $(PROFILE_TOKENS)' \
		2>&1 | tee results/profile-yoga-vs-llamacpp-$(DATE).txt

profile-yoga-vs-vllm: ## Profile + compare against vLLM baseline (PMAT-056)
	@echo "=== apr profile with vLLM comparison ==="
	ssh yoga 'apr profile $(YOGA_GGUF) \
		--perf-grade --granular \
		--baseline-url http://127.0.0.1:8084 --baseline-model $(VLLM_MODEL) --baseline-name vLLM \
		--warmup $(PROFILE_WARMUP) --measure $(PROFILE_MEASURE) --tokens $(PROFILE_TOKENS)' \
		2>&1 | tee results/profile-yoga-vs-vllm-$(DATE).txt

# Full profiling pipeline: deploy realizr, profile, trace, teardown
profile-yoga-full: deploy-yoga-realizr profile-yoga profile-yoga-trace teardown-yoga
	@echo "=== Full Yoga Profiling Complete ==="

# ============================================================================
# Provable contract gates
# ============================================================================
# Contracts: paged-kv-cache-v1, continuous-batching-v1, ptx-target-parity-v1,
#            kv-cache-equivalence-v1, performance-grading-v1, inference-pipeline-v1

contract-lint: ## Validate provable contracts (min-score 0.25, binding-aware)
	pv lint ../provable-contracts/contracts/ --binding ../provable-contracts/contracts/realizar/binding.yaml --min-score 0.25

contract-validate: ## Validate new paged KV + continuous batching contracts
	pv validate ../provable-contracts/contracts/paged-kv-cache-v1.yaml
	pv validate ../provable-contracts/contracts/continuous-batching-v1.yaml
	pv validate ../provable-contracts/contracts/performance-grading-v1.yaml

contract-score: ## Score realizr-relevant contracts
	@echo "=== Paged KV Cache ==="
	@pv score ../provable-contracts/contracts/paged-kv-cache-v1.yaml
	@echo "=== Continuous Batching ==="
	@pv score ../provable-contracts/contracts/continuous-batching-v1.yaml
	@echo "=== KV Cache Equivalence ==="
	@pv score ../provable-contracts/contracts/kv-cache-equivalence-v1.yaml
	@echo "=== Performance Grading ==="
	@pv score ../provable-contracts/contracts/performance-grading-v1.yaml
	@echo "=== Inference Pipeline ==="
	@pv score ../provable-contracts/contracts/inference-pipeline-v1.yaml

# Run static falsification tests from ptx-target-parity-v1.yaml
contract-falsify: ## PMAT-044: verify no hardcoded emit_ptx() in executor
	@echo "FALSIFY-PTP-001: No hardcoded emit_ptx in executor runtime path"
	@HITS=$$(grep -r '\.emit_ptx()' ../realizar/src/cuda/executor/ 2>/dev/null | wc -l); \
	if [ "$$HITS" -ne 0 ]; then echo "FAIL: $$HITS occurrences of .emit_ptx() in executor/"; exit 1; fi; \
	echo "  PASS (0 occurrences)"
	@echo "FALSIFY-PTP-002: CudaKernels uses device target"
	@HITS=$$(grep -c 'CudaKernels::new()' ../realizar/src/cuda/executor/core.rs 2>/dev/null); HITS=$${HITS:-0}; \
	if [ "$$HITS" != "0" ]; then echo "FAIL: CudaKernels::new() found in core.rs (should use with_target)"; exit 1; fi; \
	echo "  PASS (CudaKernels::with_target in use)"
	@echo "FALSIFY-PTP-003: generate_ptx threads target"
	@HITS=$$(grep -c 'fn generate_.*_ptx(kernel_type: &KernelType)' ../realizar/src/cuda/kernels_generate_gemm_cuda.rs 2>/dev/null); HITS=$${HITS:-0}; \
	if [ "$$HITS" != "0" ]; then echo "FAIL: $$HITS functions missing target param"; exit 1; fi; \
	echo "  PASS (all generate helpers accept target param)"
	@echo ""
	@echo "All static falsification tests passed."

# ============================================================================
# CORRECTNESS-013: Batched decode frozen slots instrumentation
# ============================================================================
# Five-whys: correctness_under_batching obligation NOT IMPLEMENTED (BIND-003)
# Falsification: FALSIFY-CB-006, CB-008, CB-009

correctness-013-deploy: ## Deploy realizr on yoga with ALL trace env vars
	@echo "=== CORRECTNESS-013: Deploy with tracing ==="
	forjar apply -f forjar-yoga-teardown.yaml --force 2>/dev/null || true
	@ssh noah@$(YOGA_HOST) 'pkill -f "apr serve" 2>/dev/null; sleep 2; pkill -9 -f "apr serve" 2>/dev/null; sleep 1; true'
	ssh noah@$(YOGA_HOST) 'SKIP_PARITY_GATE=1 PMAT051_TRACE=1 PREFILL_DETAIL_TRACE=1 \
		nohup /home/noah/.cargo/bin/apr serve \
		--model /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
		--gpu --host 0.0.0.0 --port 8081 > /tmp/apr-correctness013.log 2>&1 &'
	@echo "Waiting for health..."
	@sleep 15
	@curl -sf $(YOGA_REALIZAR)/health || (echo "FAIL: health check"; exit 1)
	@echo "Ready."

correctness-013-test: ## FALSIFY-CB-006: c=1 correctness + c=4 load (frozen slot detection)
	@echo "=== FALSIFY-CB-006: c=1 correctness baseline ==="
	probador llm test --config prompts/correctness.yaml --url $(YOGA_REALIZAR) \
		-o results/correctness-013-c1-$(DATE).json 2>&1 | tee results/correctness-013-c1-$(DATE).txt
	@echo ""
	@echo "=== FALSIFY-CB-008: c=4 load (15s, detect frozen slots) ==="
	probador llm load --url $(YOGA_REALIZAR) --duration 15 --concurrency 4 --warmup 2 --stream true \
		2>&1 | tee results/correctness-013-c4-$(DATE).txt
	@echo ""
	@echo "=== Server trace log (frozen slot evidence) ==="
	@ssh noah@$(YOGA_HOST) 'grep -E "token_ids=|frozen|batched_kv_lengths" /tmp/apr-correctness013.log | tail -30'

correctness-013-trace: ## X-Trace-Level: brick on c=1 request
	@echo "=== X-Trace-Level: brick (c=1) ==="
	@curl -s -X POST $(YOGA_REALIZAR)/v1/chat/completions \
		-H "Content-Type: application/json" \
		-H "X-Trace-Level: brick" \
		-d '{"model":"default","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":8}' | python3 -m json.tool

correctness-013-load: ## c=4 extended load test (60s) + full server log
	@echo "=== FALSIFY-CB-008: c=4 load test (60s) ==="
	probador llm load --url $(YOGA_REALIZAR) --duration 60 --concurrency 4 --warmup 5 --stream true \
		2>&1 | tee results/correctness-013-load-$(DATE).txt
	@echo ""
	@echo "=== Server log (last 60 lines) ==="
	@ssh noah@$(YOGA_HOST) 'tail -60 /tmp/apr-correctness013.log'

correctness-013-kv-verify: ## FALSIFY-CB-009: verify batched_kv_lengths after prefill
	@echo "=== FALSIFY-CB-009: KV cache population check ==="
	@ssh noah@$(YOGA_HOST) 'grep -E "batched_kv_lengths|Prefill done|decode step 0" /tmp/apr-correctness013.log | tail -20'

correctness-013-audit: ## pv audit on continuous-batching contract
	@echo "=== pv audit continuous-batching-v1.yaml ==="
	pv audit ../provable-contracts/contracts/continuous-batching-v1.yaml \
		--binding ../provable-contracts/contracts/realizar/binding.yaml

correctness-013-full: correctness-013-deploy correctness-013-test correctness-013-kv-verify correctness-013-audit ## Full CORRECTNESS-013 instrumentation pipeline
	@echo ""
	@echo "=== CORRECTNESS-013 Instrumentation Complete ==="
	@echo "Review: results/correctness-013-*-$(DATE).txt"
	@echo "Server log: ssh noah@$(YOGA_HOST) 'cat /tmp/apr-correctness013.log'"

# ============================================================================
# Shared targets
# ============================================================================

report:
	probador llm report --results results/ --output performance.md --update-readme README.md

# --- Scoring ---

score:
	@echo "=== Yoga RTX 4060L Scorecard ==="
	@for c in 1 4 8 16 32 64 128; do \
		echo ""; \
		echo "--- c=$$c ---"; \
		probador llm score --results results/ --platform yoga --concurrency $$c --format table; \
	done

score-prod:
	@echo "=== Yoga Production Scorecard (PMAT-177+ methodology) ==="
	@mkdir -p /tmp/yoga-prod-scoring
	@cp results/*yoga-prod*.json /tmp/yoga-prod-scoring/ 2>/dev/null || true
	@for c in 1 4 8 16 32 64 128; do \
		echo ""; \
		echo "--- c=$$c ---"; \
		probador llm score --results /tmp/yoga-prod-scoring/ --concurrency $$c --format table; \
	done
	@rm -rf /tmp/yoga-prod-scoring

score-all:
	probador llm score --results results/

score-json:
	@for c in 1 4 8 16 32 64 128; do \
		probador llm score --results results/ --platform yoga --concurrency $$c --format json --output results/scorecard-yoga-c$$c-$(DATE).json; \
	done

score-jetson:
	@for c in 1 4; do \
		echo ""; \
		echo "--- Jetson c=$$c ---"; \
		probador llm score --results results/ --platform jetson --concurrency $$c --format table; \
	done

score-gate:
	probador llm score --results results/ --platform yoga --concurrency 1 --fail-on-grade C

# PMAT-328: PyTorch canary testing (ground truth comparison)
canary-golden:
	ssh yoga 'source ~/venvs/vllm/bin/activate && python3 /tmp/canary_pytorch.py generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct --output /tmp/canary-golden.json'
	scp yoga:/tmp/canary-golden.json results/canary-golden.json

canary-yoga:
	scp scripts/canary_pytorch.py yoga:/tmp/canary_pytorch.py
	ssh yoga 'source ~/venvs/vllm/bin/activate && python3 /tmp/canary_pytorch.py compare --golden /tmp/canary-golden.json --url http://127.0.0.1:8081 --name realizr-gpu'

canary-cpu:
	scp scripts/canary_pytorch.py intel:/tmp/canary_pytorch.py
	scp results/canary-golden.json intel:/tmp/canary-golden.json
	ssh intel 'python3 /tmp/canary_pytorch.py compare --golden /tmp/canary-golden.json --url http://127.0.0.1:8081 --name realizr-cpu'
