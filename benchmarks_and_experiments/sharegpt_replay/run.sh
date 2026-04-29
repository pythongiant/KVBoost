#!/usr/bin/env bash
#
# ShareGPT Replay Benchmark — KVBoost vs Baseline
#
# Replays 1000 real ShareGPT conversations with the best-performing KVBoost
# config and saves results + the TTFT-vs-turn-number plot.
#
# Usage:
#   ./run.sh                        # default 1000 conversations
#   ./run.sh --n-conversations 200  # quick smoke-test
#   ./run.sh --model meta-llama/Llama-3.2-3B
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="${*}"

echo "========================================"
echo "  SHAREGPT REPLAY BENCHMARK"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "========================================"
echo ""

python "${SCRIPT_DIR}/run_sharegpt.py" \
    --recompute-strategy cacheblend \
    --chunk-boundary-window 16 \
    --overlap-k 16 \
    --sink-tokens 32 \
    --max-cache-bytes 1.5e9 \
    --recency-window-chunks 8 \
    --n-samples 500 \
    --max-context-tokens 6000 \
    --min-turns 3 \
    --max-turns 8 \
    --model meta-llama/Llama-3.2-3B \
    --max-new-tokens 128 \
    --output "${SCRIPT_DIR}/results/sharegpt_replay.json" \
    ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "  Results: ${SCRIPT_DIR}/results/sharegpt_replay.json"
echo "  Plot:    ${SCRIPT_DIR}/results/sharegpt_ttft_vs_turn.png"
echo "========================================"
