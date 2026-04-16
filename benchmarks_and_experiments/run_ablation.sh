#!/usr/bin/env bash
#
# Run bug localization benchmark with both recompute strategies sequentially.
# Each run gets its own checkpoint and output file, so they can be resumed
# independently. --skip-baseline-logits cuts ~33% runtime per run.
#
# Usage:
#   ./run_ablation.sh                    # defaults: 50 samples
#   ./run_ablation.sh --n-samples 100    # pass extra args to both runs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="${*}"

echo "========================================"
echo "  ABLATION: selective vs cacheblend"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "========================================"
echo ""

echo "--- [1/2] selective (default) ---"
python "${SCRIPT_DIR}/long_bench_arena.py" \
    --recompute-strategy selective \
    --skip-baseline-logits \
    ${EXTRA_ARGS}

echo ""
echo "--- [2/2] cacheblend ---"
python "${SCRIPT_DIR}/long_bench_arena.py" \
    --recompute-strategy cacheblend \
    --skip-baseline-logits \
    ${EXTRA_ARGS}

echo ""
echo "========================================"
echo "  Results:"
echo "    ${SCRIPT_DIR}/results/bug_localization.json            (selective)"
echo "    ${SCRIPT_DIR}/results/bug_localization_cacheblend.json (cacheblend)"
echo "========================================"
