#!/usr/bin/env bash
# Launch kvboost in its BEST setup for the coding benchmark — showcases the
# features the benchmark measures: KV reuse (faster TTFT) + OOM recovery, with
# the recent correctness/perf fixes all active.
#
# Run this, then in another shell:
#   python bench_coding.py --backend kvboost --url http://localhost:9000 \
#       --model "$MODEL" --mode both --out kvboost.json
# Stop it (Ctrl-C) before launching vLLM — one model fits the GPU at a time.
#
# Override via env: MODEL=... PORT=... MAX_CACHE_BYTES=... ./start_kvboost.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-9000}"
# KV-cache budget for cross-request chunk reuse. Size to (free VRAM after
# weights). On a 14.6 GiB card with a 3B fp16 model (~6 GiB) → ~4 GiB leaves
# headroom for prefill activations + the live request. Lower for the OOM-
# stress run to make the planner's adaptation more visible (e.g. 1.5e9).
MAX_CACHE_BYTES="${MAX_CACHE_BYTES:-4e9}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.15}"

echo "kvboost (best setup)"
echo "  model:            $MODEL"
echo "  port:             $PORT"
echo "  recompute:        cacheblend_sparse  (faithful selective recompute)"
echo "  kv-cache-bits:    8                  (int8 KV → 2× reuse capacity)"
echo "  max-cache-bytes:  $MAX_CACHE_BYTES"
echo "  oom planning:     on (safety_margin=$SAFETY_MARGIN)"
echo

# Why each flag:
#   --recompute-strategy cacheblend_sparse
#       Faithful CacheBlend: recompute only high-deviation tokens layer-by-
#       layer (paper's 2.2-3.3× TTFT), not the full-forward variant. This is
#       the "faster TTFT on reused context" feature. Falls back to plain
#       cacheblend automatically on unsupported architectures.
#   --kv-cache-bits 8
#       int8 KV cache: ~2× the cached-chunk capacity (more cross-request
#       reuse) and lower memory pressure, negligible quality cost.
#   --max-cache-bytes
#       Cross-request chunk-cache budget — bigger = more reuse, bounded by VRAM.
#   OOM planner (on by default) + --planner-safety-margin
#       Per-request peak prediction → picks chunk_size/kv_bits that fit, or a
#       clean HTTP 413. This is the "OOM recovery" feature. Add --auto-truncate
#       to truncate-and-complete oversized prompts instead of 413.
#   --max-batch-size 1
#       The benchmark replays sequentially (single GPU worker); 1 avoids
#       pointless batch-window latency. Raise for concurrent throughput tests.
#   (automatic, no flag: O(n) incremental detok, chunked CacheBlend forward,
#    streaming usage emission for input-throughput, planner cost probe.)
exec python -m kvboost.server \
    --model "$MODEL" \
    --dtype float16 \
    --recompute-strategy cacheblend_sparse \
    --kv-cache-bits 8 \
    --max-cache-bytes "$MAX_CACHE_BYTES" \
    --planner-safety-margin "$SAFETY_MARGIN" \
    --max-batch-size 1 \
    --host 0.0.0.0 \
    --port "$PORT"

# ── Optional add-ons (uncomment to enable) ───────────────────────────────────
# Speculative decoding to lift DECODE throughput (where vLLM's continuous
# batching otherwise leads). Needs a small same-family draft model and ~1 GiB
# extra VRAM; --speculative-tree turns on the SpecBlock-inspired tree variant
# with cost-aware per-request mode selection:
#     --speculative-draft-model Qwen/Qwen2.5-0.5B-Instruct \
#     --speculative-tree \
#
# Oversized-prompt policy for the OOM ramp: complete-by-truncation instead of
# a clean 413 reject:
#     --auto-truncate
