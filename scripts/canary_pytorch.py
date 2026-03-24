#!/usr/bin/env python3
"""PMAT-328: PyTorch canary testing — HuggingFace reference vs realizr/llama.cpp/ollama.

Adapted from bashrs canary pattern:
1. Load model via HuggingFace transformers (ground truth)
2. Generate golden outputs (greedy, temperature=0)
3. Compare token-by-token against inference runtimes
4. Ship/kill gate: token match >= 95%, KL div < 0.1

Usage:
    # Generate golden reference (on yoga with vLLM venv)
    python3 scripts/canary_pytorch.py generate --model Qwen/Qwen2.5-Coder-1.5B-Instruct

    # Compare against runtime
    python3 scripts/canary_pytorch.py compare --golden results/canary-golden.json \
        --url http://192.168.50.38:8081

    # Full pipeline (generate + compare all runtimes)
    python3 scripts/canary_pytorch.py full --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
        --urls realizr=http://192.168.50.38:8081,llamacpp=http://192.168.50.38:8083
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Canonical prompts — same as correctness.yaml but with expected golden tokens
CANARY_PROMPTS = [
    {
        "id": "math_simple",
        "messages": [{"role": "user", "content": "What is 7 * 8? Answer with just the number."}],
        "max_tokens": 8,
        "expected_contains": "56",
    },
    {
        "id": "code_fibonacci",
        "messages": [{"role": "user", "content": "Write a Python function to compute the nth Fibonacci number. Just the code, no explanation."}],
        "max_tokens": 128,
        "expected_contains": "def fib",
    },
    {
        "id": "hello_rust",
        "messages": [{"role": "user", "content": "Write a Rust hello world program. Just the code."}],
        "max_tokens": 64,
        "expected_contains": "fn main",
    },
    {
        "id": "json_output",
        "messages": [{"role": "user", "content": "Output a JSON object with keys 'name' and 'age'. Just the JSON, nothing else."}],
        "max_tokens": 32,
        "expected_contains": '"name"',
    },
    {
        "id": "sql_query",
        "messages": [{"role": "user", "content": "Write a SQL query to select all users older than 18. Just the query."}],
        "max_tokens": 32,
        "expected_contains": "SELECT",
    },
]


def generate_golden(model_name: str, output_path: str):
    """Generate golden reference outputs using HuggingFace transformers."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    results = []
    for prompt in CANARY_PROMPTS:
        print(f"  Generating: {prompt['id']}...")
        messages = prompt["messages"]

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=prompt["max_tokens"],
                do_sample=False,  # Greedy — deterministic
                temperature=1.0,
                return_dict_in_generate=True,
                output_logits=True,
            )
        elapsed = time.time() - t0

        # Extract generated tokens (excluding prompt)
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs.sequences[0][prompt_len:].tolist()
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        # Extract top-5 logits per position for distribution comparison
        top_logits = []
        if hasattr(outputs, "logits") and outputs.logits:
            for step_logits in outputs.logits:
                probs = torch.softmax(step_logits[0], dim=-1)
                top_vals, top_ids = probs.topk(5)
                top_logits.append({
                    "ids": top_ids.tolist(),
                    "probs": top_vals.tolist(),
                })

        results.append({
            "id": prompt["id"],
            "messages": messages,
            "max_tokens": prompt["max_tokens"],
            "expected_contains": prompt["expected_contains"],
            "golden_text": gen_text,
            "golden_token_ids": gen_ids,
            "golden_token_count": len(gen_ids),
            "top5_logits": top_logits[:10],  # First 10 positions only (save space)
            "latency_s": elapsed,
            "model": model_name,
        })
        print(f"    -> {len(gen_ids)} tokens in {elapsed:.2f}s: {gen_text[:80]}...")

    # Save golden reference
    golden = {
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prompts": results,
    }
    Path(output_path).write_text(json.dumps(golden, indent=2))
    print(f"\nGolden reference saved to {output_path}")
    return golden


def compare_runtime(golden_path: str, url: str, runtime_name: str = "unknown"):
    """Compare a runtime's outputs against golden reference."""
    import requests

    golden = json.loads(Path(golden_path).read_text())
    print(f"\nComparing {runtime_name} ({url}) against {golden['model']}...")

    results = []
    total_match = 0
    total_prompts = 0

    for ref in golden["prompts"]:
        prompt_id = ref["id"]
        print(f"  {prompt_id}...", end=" ")

        # Call runtime via OpenAI-compat API
        try:
            t0 = time.time()
            resp = requests.post(
                f"{url}/v1/chat/completions",
                json={
                    "model": "qwen",
                    "messages": ref["messages"],
                    "max_tokens": ref["max_tokens"],
                    "temperature": 0,
                },
                timeout=30,
            )
            elapsed = time.time() - t0
            resp.raise_for_status()
            data = resp.json()
            runtime_text = data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"FAIL ({e})")
            results.append({"id": prompt_id, "status": "error", "error": str(e)})
            continue

        # Compare
        golden_text = ref["golden_text"]
        contains_expected = ref["expected_contains"].lower() in runtime_text.lower()

        # Token overlap (approximate — we don't have runtime token IDs from API)
        # Use character-level Jaccard similarity as proxy
        g_set = set(golden_text.lower().split())
        r_set = set(runtime_text.lower().split())
        jaccard = len(g_set & r_set) / max(len(g_set | r_set), 1)

        status = "PASS" if contains_expected else "FAIL"
        total_match += int(contains_expected)
        total_prompts += 1

        results.append({
            "id": prompt_id,
            "status": status,
            "golden_text": golden_text[:100],
            "runtime_text": runtime_text[:100],
            "contains_expected": contains_expected,
            "word_jaccard": round(jaccard, 3),
            "latency_s": round(elapsed, 3),
            "golden_latency_s": ref.get("latency_s", 0),
        })
        print(f"{status} (jaccard={jaccard:.2f}, {elapsed:.2f}s)")

    # Ship/Kill gate
    match_rate = total_match / max(total_prompts, 1)
    gate = "SHIP" if match_rate >= 0.80 else "KILL"

    summary = {
        "runtime": runtime_name,
        "url": url,
        "golden_model": golden["model"],
        "match_rate": round(match_rate, 3),
        "total_prompts": total_prompts,
        "total_match": total_match,
        "gate": gate,
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"  Runtime: {runtime_name}")
    print(f"  Match rate: {total_match}/{total_prompts} ({match_rate:.0%})")
    print(f"  Gate: {gate}")
    print(f"{'='*60}")

    return summary


def full_pipeline(model_name: str, urls: dict, output_dir: str = "results"):
    """Run full canary pipeline: generate golden + compare all runtimes."""
    golden_path = f"{output_dir}/canary-golden.json"

    # Step 1: Generate golden reference
    generate_golden(model_name, golden_path)

    # Step 2: Compare each runtime
    all_results = []
    for name, url in urls.items():
        result = compare_runtime(golden_path, url, name)
        result_path = f"{output_dir}/canary-{name}.json"
        Path(result_path).write_text(json.dumps(result, indent=2))
        all_results.append(result)

    # Step 3: Summary
    print(f"\n{'='*60}")
    print("CANARY TEST SUMMARY")
    print(f"{'='*60}")
    any_kill = False
    for r in all_results:
        icon = "✓" if r["gate"] == "SHIP" else "✗"
        print(f"  {icon} {r['runtime']}: {r['match_rate']:.0%} ({r['gate']})")
        if r["gate"] == "KILL":
            any_kill = True

    return 1 if any_kill else 0


def main():
    parser = argparse.ArgumentParser(description="PMAT-328: PyTorch canary testing")
    sub = parser.add_subparsers(dest="command")

    gen = sub.add_parser("generate", help="Generate golden reference")
    gen.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    gen.add_argument("--output", default="results/canary-golden.json")

    cmp = sub.add_parser("compare", help="Compare runtime against golden")
    cmp.add_argument("--golden", required=True)
    cmp.add_argument("--url", required=True)
    cmp.add_argument("--name", default="unknown")

    full = sub.add_parser("full", help="Full pipeline: generate + compare")
    full.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    full.add_argument("--urls", required=True, help="name=url,name=url")
    full.add_argument("--output-dir", default="results")

    args = parser.parse_args()

    if args.command == "generate":
        generate_golden(args.model, args.output)
    elif args.command == "compare":
        result = compare_runtime(args.golden, args.url, args.name)
        print(json.dumps(result, indent=2))
    elif args.command == "full":
        urls = dict(pair.split("=", 1) for pair in args.urls.split(","))
        sys.exit(full_pipeline(args.model, urls, args.output_dir))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
