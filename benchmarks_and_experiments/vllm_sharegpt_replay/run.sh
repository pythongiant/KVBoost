#!/usr/bin/env bash
#
# vLLM ShareGPT Replay Benchmark — Prefix Cache
#
# Replays ShareGPT conversations through vLLM with prefix caching enabled
# and saves results + the TTFT-vs-turn-number plot.
#
# Usage:
#   ./run.sh                        # default 500 conversations
#   ./run.sh --n-conversations 200  # quick smoke-test
#   ./run.sh --model meta-llama/Llama-3.2-3B
#   ./run.sh --no-prefix-caching    # ablation: disable prefix cache
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="${*}"

echo "========================================"
echo "  VLLM SHAREGPT REPLAY BENCHMARK"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "========================================"
echo ""

# Prefix caching is ALWAYS enabled — do not pass --no-prefix-caching here.
# This keeps the benchmark fair: vLLM prefix cache is the feature under test.
python "${SCRIPT_DIR}/run_sharegpt_vllm.py" \
    --n-samples 500 \
    --min-turns 3 \
    --max-turns 8 \
    --max-context-tokens 6000 \
    --model Qwen/Qwen2.5-3B \
    --max-new-tokens 128 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --output "${SCRIPT_DIR}/results/vllm_sharegpt_replay.json" \
    ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "  Results: ${SCRIPT_DIR}/results/vllm_sharegpt_replay.json"
echo "  Plot:    ${SCRIPT_DIR}/results/vllm_sharegpt_ttft_vs_turn.png"
echo "========================================"
