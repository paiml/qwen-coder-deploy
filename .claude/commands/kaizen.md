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

- Spec: `docs/specifications/gpu-performance-spec.md` v5.28.0 (290 PMAT items)
- Production config: `CUDA_MAX_BATCH=32 ITERATION_SCHEDULER=1`
- Yoga IP: `192.168.50.38`, ports: realizr 8081, ollama 8082, llama.cpp 8083, vLLM 8084
- Asymptote: 1,511 tok/s (+71% from iter sched). GPU within 8% of vLLM (7.4 vs 6.8ms)
- Binding bottleneck: 430 CPU dispatch calls (~5ms non-GEMM at ~25us, ~0.6ms GEMM at ~3us)
- All kernel optimizations exhausted (PMAT-279-289): event sync, per-M graph, fused scatter, fused DP4A, non-GEMM fusion, megakernel, prefill chunking — ALL falsified or low ROI
- PMAT-290: −12% regression from trueno PTX disk caching — FIXED (pass CU_JIT_TARGET to cuLinkCreate)
- CB mostly complete (PMAT-088c/d). Graph safety resolved (CORRECTNESS-014)
- Remaining: cuBLAS grouped GEMM (CUDA 12.x), or accept current 0.44-0.51x vLLM at c=4-32
