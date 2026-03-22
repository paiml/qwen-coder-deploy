Continue implementing the next best step with PMAT work items from our spec. This is an autonomous loop — pick up where you left off, implement, measure, document, push.

## Process

1. **Identify next best step** using five-whys analysis:
   - Read the spec at `docs/specifications/gpu-performance-spec.md`
   - Check roadmap at `docs/roadmaps/roadmap.yaml`
   - Use `pmat query` (NOT grep) for all code search across trueno/realizar/renacer
   - Prioritize by measured ROI, not theoretical projection

2. **Implement** across all repos as needed:
   - `~/src/trueno` — CUDA kernels, PTX generation, CudaEvent API
   - `~/src/realizar` — inference engine, iteration scheduler, decode path
   - `~/src/renacer` — tracing infrastructure (renacer-core)
   - `~/src/aprender` — build workspace (apr-cli binary)
   - `~/src/qwen-coder-deploy` — benchmarks, spec, docs

3. **Build and deploy** to yoga (RTX 4060L):
   - Build: `cd ~/src/aprender && touch crates/apr-cli/src/main.rs && CARGO_TARGET_DIR=/mnt/nvme-raid0/targets/aprender cargo build --release -p apr-cli --bin apr --features "apr-cli/hf-hub,apr-cli/safetensors-compare,apr-cli/inference,apr-cli/cuda"`
   - Deploy: `scp /mnt/nvme-raid0/targets/aprender/release/apr yoga:~/.cargo/bin/apr`
   - Start: `ssh yoga 'SKIP_PARITY_GATE=1 CUDA_MAX_BATCH=32 ITERATION_SCHEDULER=1 nohup apr serve run /home/noah/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf --gpu --host 0.0.0.0 --port 8081 > /tmp/apr-gpu.log 2>&1 & sleep 15; curl -sf http://127.0.0.1:8081/health'`

4. **Measure** with production methodology:
   - Correctness: `probador llm test --config prompts/correctness.yaml --url http://192.168.50.38:8081`
   - Benchmark: `probador llm load --url http://192.168.50.38:8081 --concurrency N --duration 60 --warmup 5 --max-tokens-distribution uniform:16,256 --stream true --runtime-name realizr -o results/FILE.json`

5. **Document and push** after every measurement:
   - Update `performance.md`, `docs/specifications/gpu-performance-spec.md`, `README.md`, `docs/roadmaps/roadmap.yaml`
   - Ensure `pv lint` passes
   - Commit with `(Refs PMAT-XXX)` and `Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>`
   - Push to main (no branches)

6. **Falsify aggressively** — Popperian methodology:
   - Every projection must have a defined falsification condition
   - When measurement contradicts projection, document as FALSIFIED
   - Update all stale claims across all docs

## Critical Rules

- Use `pmat query` for ALL code search (NOT grep/glob)
- Apply five-whys on EVERY finding
- PTX must be pure ASCII (no em dashes, smart quotes, etc.)
- Multi-week PTX kernel work is IN SCOPE — implement megakernel, fused GEMM, etc.
- Include all repos: trueno, realizar, renacer, aprender, qwen-coder-deploy
- `pv lint` must pass before every push to qwen-coder-deploy

## Current State

- Spec: `docs/specifications/gpu-performance-spec.md` v5.34.0 (296 PMAT items)
- Production config: `CUDA_MAX_BATCH=32 ITERATION_SCHEDULER=1`
- Yoga IP: `192.168.50.38`, ports: realizr 8081, ollama 8082, llama.cpp 8083, vLLM 8084
- Spec: v5.37.0 (312 PMAT items)
- **GPU: 149/322/529/947/1600 tok/s at c=1/4/8/16/32** (+8.5% from PMAT-291/294)
- **CPU: 32.6 tok/s** (+91% from 17.1, gap 1.81x vs llama.cpp 59.0)
- Graph dispatch ON by default + Q8 cache. realizr beats llama.cpp at c=8 (+26%)
- 16 GPU kernel fusion approaches falsified. Architecture ceiling reached
- 20 CPU approaches tested: 7 confirmed, 13 falsified. IPC 1.59 vs 1.01
- Remaining: NUMA pinning, cuBLAS grouped GEMM, persistent kernel, or pivot to features
