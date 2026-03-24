#!/bin/bash
# PMAT-329: Cross-backend parity gate
# Validates q4k-cross-backend-parity-v1.yaml contract
#
# Tests factual prompts (MUST match) and creative prompts (MAY diverge).
# Exit 0 = SHIP, Exit 1 = KILL
#
# Usage: ./scripts/canary_parity_gate.sh GPU_URL CPU_URL

set -uo pipefail

GPU_URL="${1:-http://192.168.50.38:8081}"
CPU_URL="${2:-http://192.168.50.100:8081}"

echo "PMAT-329: Cross-Backend Parity Gate"
echo "===================================="
echo "GPU: $GPU_URL"
echo "CPU: $CPU_URL"
echo ""

FACTUAL_PASS=0
FACTUAL_TOTAL=0
CREATIVE_MATCH=0
CREATIVE_TOTAL=0

check_prompt() {
    local type="$1"  # factual or creative
    local prompt="$2"
    local expected="$3"

    local gpu_out cpu_out
    gpu_out=$(curl -s --max-time 15 "$GPU_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":16,\"temperature\":0}" \
        2>/dev/null | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")

    cpu_out=$(curl -s --max-time 15 "$CPU_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"$prompt\"}],\"max_tokens\":16,\"temperature\":0}" \
        2>/dev/null | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['choices'][0]['message']['content'])" 2>/dev/null || echo "ERROR")

    local match="NO"
    [[ "$gpu_out" == "$cpu_out" ]] && match="YES"

    local contains_expected="NO"
    if echo "$gpu_out" | grep -qi "$expected" 2>/dev/null; then
        if echo "$cpu_out" | grep -qi "$expected" 2>/dev/null; then
            contains_expected="YES"
        fi
    fi

    if [[ "$type" == "factual" ]]; then
        FACTUAL_TOTAL=$((FACTUAL_TOTAL + 1))
        if [[ "$contains_expected" == "YES" ]]; then
            FACTUAL_PASS=$((FACTUAL_PASS + 1))
            echo "  [FACTUAL] PASS: $prompt"
            [[ "$match" == "NO" ]] && echo "    WARN: outputs differ but both contain '$expected'"
        else
            echo "  [FACTUAL] FAIL: $prompt"
            echo "    GPU: $gpu_out"
            echo "    CPU: $cpu_out"
            echo "    Expected: $expected"
        fi
    else
        CREATIVE_TOTAL=$((CREATIVE_TOTAL + 1))
        [[ "$match" == "YES" ]] && CREATIVE_MATCH=$((CREATIVE_MATCH + 1))
        echo "  [CREATIVE] $prompt → match=$match"
        [[ "$match" == "NO" ]] && echo "    GPU: $gpu_out" && echo "    CPU: $cpu_out"
    fi
}

echo "--- Factual prompts (MUST match expected content) ---"
check_prompt factual "What is 7 * 8? Answer with just the number." "56"
check_prompt factual "What is the capital of France? One word." "Paris"
check_prompt factual "What language is fn main used in? One word." "Rust"

echo ""
echo "--- Creative prompts (MAY diverge) ---"
check_prompt creative "What is your name?" ""
check_prompt creative "List 3 colors." ""
check_prompt creative "Output a JSON with name and age." ""

echo ""
echo "===================================="
echo "Factual: $FACTUAL_PASS/$FACTUAL_TOTAL"
echo "Creative match: $CREATIVE_MATCH/$CREATIVE_TOTAL"

if [[ $FACTUAL_PASS -eq $FACTUAL_TOTAL ]]; then
    echo "Gate: SHIP (all factual prompts match)"
    exit 0
else
    echo "Gate: KILL (factual divergence detected!)"
    exit 1
fi
