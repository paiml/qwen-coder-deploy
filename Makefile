# ============================================================================
# qwen-coder-deploy — Benchmark realizar vs ollama vs llama.cpp
# ============================================================================
# Targets:
#   CPU (intel host):  make deploy / make test / make load
#   GPU (localhost):   make deploy-gpu / make test-gpu / make load-gpu
#
# Deep profiling (apr/realizar tools):
#   make profile-gpu    — Roofline analysis + hotspot breakdown
#   make bench-gpu      — Per-brick timing with budget targets
#   make cbtop-gpu      — ComputeBrick pipeline profiler (headless)
#   make qa-gpu         — 10-gate falsifiable QA checklist
#   make trace-gpu      — Per-request brick/layer tracing
#   make realize-bench  — realizar internal benchmark suites
#   make gpu-util       — nvidia-smi GPU utilization snapshot
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

# --- GPU (localhost, RTX 4090) ---
GPU_HOST := 127.0.0.1
GPU_REALIZAR := http://$(GPU_HOST):8081
GPU_OLLAMA   := http://$(GPU_HOST):8082
GPU_LLAMACPP := http://$(GPU_HOST):8083

# Ollama requires exact model tag (not "default")
OLLAMA_MODEL := qwen2.5-coder:1.5b-instruct-q4_K_M

GGUF_MODEL := /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf

.PHONY: deploy teardown test load report nightly health \
        deploy-gpu teardown-gpu test-gpu load-gpu health-gpu nightly-gpu \
        profile-gpu bench-gpu cbtop-gpu qa-gpu trace-gpu realize-bench \
        gpu-util full-gpu

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

deploy-gpu:
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
# Deep profiling (apr + realizar tools)
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
# Shared targets
# ============================================================================

report:
	probador llm report --results results/ --output performance.md --update-readme README.md
