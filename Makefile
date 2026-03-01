INTEL_HOST := 192.168.50.100
REALIZAR_URL := http://$(INTEL_HOST):8081
OLLAMA_URL := http://$(INTEL_HOST):8082
LLAMACPP_URL := http://$(INTEL_HOST):8083
DATE := $(shell date +%Y%m%d)

.PHONY: deploy teardown test load report nightly health

deploy:
	forjar apply

teardown:
	forjar apply -f forjar-teardown.yaml

health:
	@echo "Checking realizar..."
	@curl -sf $(REALIZAR_URL)/health && echo " OK" || echo " FAIL"
	@echo "Checking ollama..."
	@curl -sf $(OLLAMA_URL)/health && echo " OK" || echo " FAIL"
	@echo "Checking llama.cpp..."
	@curl -sf $(LLAMACPP_URL)/health && echo " OK" || echo " FAIL"

test:
	probador llm test --config prompts/correctness.yaml --url $(REALIZAR_URL) --runtime-name realizar --output results/realizar-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(OLLAMA_URL) --runtime-name ollama --output results/ollama-correctness-$(DATE).json
	probador llm test --config prompts/correctness.yaml --url $(LLAMACPP_URL) --runtime-name llamacpp --output results/llamacpp-correctness-$(DATE).json

load:
	probador llm load --url $(REALIZAR_URL) --concurrency 4 --duration 60s --runtime-name realizar --output results/realizar-load-$(DATE).json
	probador llm load --url $(OLLAMA_URL) --concurrency 4 --duration 60s --runtime-name ollama --output results/ollama-load-$(DATE).json
	probador llm load --url $(LLAMACPP_URL) --concurrency 4 --duration 60s --runtime-name llamacpp --output results/llamacpp-load-$(DATE).json

report:
	probador llm report --results results/ --output performance.md --update-readme README.md

nightly: deploy health test load report
