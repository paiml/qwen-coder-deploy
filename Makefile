# ============================================================================
# qwen-coder-deploy — Benchmark realizar vs ollama vs llama.cpp
# ============================================================================
# Targets:
#   CPU (intel host):  make deploy / make test / make load
#   GPU (localhost):   make deploy-gpu / make test-gpu / make load-gpu
# ============================================================================

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

.PHONY: deploy teardown test load report nightly health \
        deploy-gpu teardown-gpu test-gpu load-gpu health-gpu nightly-gpu

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
	probador llm test --config prompts/correctness.yaml --url $(INTEL_OLLAMA) --runtime-name ollama-cpu --output results/ollama-cpu-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(INTEL_LLAMACPP) --runtime-name llamacpp-cpu --output results/llamacpp-cpu-correctness-$(DATE).json

load:
	probador llm load --url $(INTEL_REALIZAR) --concurrency 4 --duration 60s --runtime-name realizar-cpu --output results/realizar-cpu-load-$(DATE).json
	probador llm load --url $(INTEL_OLLAMA) --concurrency 4 --duration 60s --runtime-name ollama-cpu --output results/ollama-cpu-load-$(DATE).json
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
	probador llm test --config prompts/correctness.yaml --url $(GPU_OLLAMA) --runtime-name ollama-gpu --output results/ollama-gpu-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(GPU_LLAMACPP) --runtime-name llamacpp-gpu --output results/llamacpp-gpu-correctness-$(DATE).json

load-gpu:
	probador llm load --url $(GPU_REALIZAR) --concurrency 4 --duration 60s --runtime-name realizar-gpu --output results/realizar-gpu-load-$(DATE).json
	probador llm load --url $(GPU_OLLAMA) --concurrency 4 --duration 60s --runtime-name ollama-gpu --output results/ollama-gpu-load-$(DATE).json
	probador llm load --url $(GPU_LLAMACPP) --concurrency 4 --duration 60s --runtime-name llamacpp-gpu --output results/llamacpp-gpu-load-$(DATE).json

nightly-gpu: deploy-gpu health-gpu test-gpu load-gpu report

# ============================================================================
# Shared targets
# ============================================================================

report:
	probador llm report --results results/ --output performance.md --update-readme README.md
