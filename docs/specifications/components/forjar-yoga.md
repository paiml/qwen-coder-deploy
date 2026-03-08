# Component: Forjar Templates (Yoga)

**Parent:** [perf-parity-spec.md](../perf-parity-spec.md)
**Status:** DONE — all templates created and verified
**Test target:** ssh yoga (192.168.50.38)

---

## Goal

Declarative forjar templates for deploying and tearing down each runtime
on yoga independently. All benchmark infrastructure flows through forjar.

---

## Machine: yoga

- GPU: NVIDIA RTX 4060 Laptop (sm_89, 24 SMs, 8GB GDDR6)
- CPU: x86_64
- OS: Ubuntu 22.04 with NVIDIA driver 590
- CUDA: 12.6 at `/usr/local/cuda` (NVIDIA repo), 11.5 at `/usr/bin` (apt, legacy)
- IP: 192.168.50.38 (static, LAN)
- SSH: `ssh yoga`
- Ports: 8081-8084 open (UFW allowed from 192.168.50.0/24)

---

## Templates

All templates live in this repo (qwen-coder-deploy).
Forjar configs reference `../infra` for machine definitions.

### Templates (all created and verified 2026-03-08)

| Template | Runtime | Port | Status |
|----------|---------|------|--------|
| `forjar-yoga-realizr.yaml` | apr serve (realizr) | 8081 | PASS |
| `forjar-yoga-llamacpp.yaml` | llama-server (CUDA 12.6, sm_89) | 8083 | PASS |
| `forjar-yoga-ollama.yaml` | ollama serve | 8082 | PASS |
| `forjar-yoga-teardown.yaml` | (all) | -- | PASS |

### Isolation Rules

1. **Never run two runtimes simultaneously** — VRAM contention corrupts benchmarks
2. **Always teardown before deploying next** — leftover processes skew results
3. **Lock GPU clocks** — prevents thermal throttling variance between runs
4. **Warmup period** — probador `--warmup 5` discards first 5 seconds

---

## Makefile Integration

```makefile
deploy-yoga-realizr:
	forjar apply -f forjar-yoga-realizr.yaml --yes

deploy-yoga-llamacpp:
	forjar apply -f forjar-yoga-llamacpp.yaml --yes

deploy-yoga-ollama:
	forjar apply -f forjar-yoga-ollama.yaml --yes

teardown-yoga:
	forjar apply -f forjar-yoga-teardown.yaml --yes

bench-yoga-realizr: deploy-yoga-realizr
	probador llm load --url http://yoga:8081 --model qwen \
		--duration 60 --concurrency 1 --warmup 5 --stream true
	$(MAKE) teardown-yoga

bench-yoga-all:
	$(MAKE) bench-yoga-realizr
	$(MAKE) bench-yoga-llamacpp
	$(MAKE) bench-yoga-ollama
```

---

## Provable Contracts

Every template includes `completion_check` on every resource. Additionally:

| Contract | Assertion | Templates |
|----------|-----------|-----------|
| `cuda-version-check` | `/usr/local/cuda/bin/nvcc` reports release 12+ | llamacpp |
| `ufw-bench-ports` | UFW allows 8081:8084/tcp from 192.168.50.0/24 | realizr, llamacpp, ollama |
| `lock-clocks` | GPU clocks locked at 1500MHz | realizr, llamacpp, ollama |
| Health checks | `curl -sf http://127.0.0.1:<port>/health` | realizr, llamacpp, ollama |

If any contract fails, `policy.failure: stop_on_first` halts the deploy.

---

## Pass Criteria

- All templates deploy successfully via `forjar apply` (all contracts pass)
- Health check passes within 10 seconds of deploy
- Teardown kills all processes cleanly
- No port conflicts between sequential deploys

---

## Issue Tracker

- **paiml/infra#1**: yoga CUDA 11.5 → 12.6 upgrade (apt → NVIDIA repo)
