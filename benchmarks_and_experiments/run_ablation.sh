#!/usr/bin/env bash
#
# Ablation benchmark: test each continuity feature independently and combined.
#
# Configurations:
#   1. baseline-recompute  — selective recompute only (current default)
#   2. adaptive-only       — adaptive chunk boundaries (window=16)
#   3. overlap-only        — overlap encoding (k=16)
#   4. sink-only           — attention sinks (s=32)
#   5. all-features        — adaptive + overlap + sink combined
#   6. cacheblend-all      — cacheblend recompute + all features
#
# Usage:
#   ./run_ablation.sh                    # defaults: 50 samples
#   ./run_ablation.sh --n-samples 100    # pass extra args to all runs
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXTRA_ARGS="${*}"

echo "========================================"
echo "  ABLATION: continuity features"
echo "  Extra args: ${EXTRA_ARGS:-<none>}"
echo "========================================"
echo ""

COMMON="--skip-baseline-logits ${EXTRA_ARGS}"


# echo ""
# echo "--- [1/6] all-features (selective + adaptive + overlap + sink) ---"
# python "${SCRIPT_DIR}/long_bench_arena.py" \
#     --recompute-strategy selective \
#     --chunk-boundary-window 16 \
#     --overlap-k 16 \
#     --sink-tokens 32 \
#     --output "${SCRIPT_DIR}/results/ablation_all_selective.json" \
#     ${COMMON}


# =====================================================================================
#   CHECKPOINT STATE — Current Progress Summary
# =====================================================================================

#   Checkpoint: 820 samples processed

#   Bucket         N  Base Acc    KV Acc    Delta   p-value   Result
#   ---------- ----- --------- --------- -------- --------- --------
#   0-512        200   100.0%   100.0%   +0.0%       N/A     PASS
#   512-1K       200    99.5%    98.0%   -1.5%     0.250     PASS
#   1K-2K        200    99.0%    96.0%   -3.0%     0.031     FAIL
#   2K-4K        200    98.5%    92.0%   -6.5%     0.001     FAIL
#   4K+           20   100.0%    90.0%  -10.0%     0.500     PASS
#   ALL          820    99.3%    96.3%   -2.9%     0.000     FAIL <<<

#   Bucket         N  Avg Ctx  Base TTFT    KV TTFT  Speedup  KV Reuse
#   ---------- ----- -------- ---------- ---------- -------- ---------
#   0-512        200     298       387ms       243ms     2.3x      21%
#   512-1K       200     732      4521ms      2309ms     3.5x      33%
#   1K-2K        200    1455      9134ms      5255ms     4.1x      40%
#   2K-4K        200    2812     26075ms     10909ms    11.2x      44%
#   4K+           20    5321     73791ms     39428ms    29.0x      42%
#   ALL          820    1422     11584ms      5527ms     5.8x      35% <<<

# =====================================================================================

# echo ""
# echo "--- [2/6] adaptive-only (boundary_window=16) ---"
# python "${SCRIPT_DIR}/long_bench_arena.py" \
#     --recompute-strategy selective \
#     --chunk-boundary-window 16 \
#     --output "${SCRIPT_DIR}/results/ablation_adaptive_only.json" \
#     ${COMMON}

# echo ""
# echo "--- [3/6] overlap-only (overlap_k=16) ---"
# python "${SCRIPT_DIR}/long_bench_arena.py" \
#     --recompute-strategy selective \
#     --overlap-k 16 \
#     --output "${SCRIPT_DIR}/results/ablation_overlap_only.json" \
#     ${COMMON}

# echo ""
# echo "--- [4/6] sink-only (sink_tokens=32) ---"
# python "${SCRIPT_DIR}/long_bench_arena.py" \
#     --recompute-strategy selective \
#     --sink-tokens 32 \
#     --output "${SCRIPT_DIR}/results/ablation_sink_only.json" \
#     ${COMMON}

echo ""
echo "--- [5/6] cacheblend + all-features ---"
python "${SCRIPT_DIR}/long_bench_arena.py" \
    --recompute-strategy cacheblend \
    --chunk-boundary-window 16 \
    --overlap-k 16 \
    --sink-tokens 32 \
    --output "${SCRIPT_DIR}/results/ablation_all_cacheblend.json" \
    ${COMMON}

echo "--- [6/6] baseline-recompute (selective, no new features) ---"
python "${SCRIPT_DIR}/long_bench_arena.py" \
    --recompute-strategy selective \
    --output "${SCRIPT_DIR}/results/ablation_baseline_recompute.json" \
    ${COMMON}

echo ""
echo "========================================"
echo "  Results:"
echo "    ${SCRIPT_DIR}/results/ablation_baseline_recompute.json"
echo "    ${SCRIPT_DIR}/results/ablation_adaptive_only.json"
echo "    ${SCRIPT_DIR}/results/ablation_overlap_only.json"
echo "    ${SCRIPT_DIR}/results/ablation_sink_only.json"
echo "    ${SCRIPT_DIR}/results/ablation_all_selective.json"
echo "    ${SCRIPT_DIR}/results/ablation_all_cacheblend.json"
echo "========================================"
