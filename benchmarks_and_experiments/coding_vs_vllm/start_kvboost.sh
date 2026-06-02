#!/usr/bin/env bash
# Launch kvboost in its BEST setup for the coding benchmark — showcases the
# features the benchmark measures (KV reuse → faster TTFT, OOM recovery) AND
# the throughput levers (FlashAttention-2, tree speculative decoding) added to
# close the gap to vLLM on an RTX 3060 (Ampere, 12 GB, ~360 GB/s).
#
# Run this, then in another shell:
#   python bench_coding.py --backend kvboost --url http://localhost:9000 \
#       --model "$MODEL" --mode both --out kvboost.json
# Stop it (Ctrl-C) before launching vLLM — one model fits the GPU at a time.
#
# Override via env: MODEL=... DRAFT=... PORT=... MAX_CACHE_BYTES=... NO_SPEC=1

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
# Small same-family drafter for speculative decoding (the decode-throughput
# lever). ~1 GB fp16; set NO_SPEC=1 to disable (e.g. to free VRAM for the
# OOM-headroom run, since the draft model lowers the context ceiling).
DRAFT="${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-9000}"
# KV-cache budget for cross-request chunk reuse. On a 12 GB 3060: ~6 GB model
# + ~1 GB draft leaves ~5 GB → 2.5 GB cache keeps activation headroom. Lower
# for the OOM-stress run to make the planner's adaptation more visible (~1e9).
MAX_CACHE_BYTES="${MAX_CACHE_BYTES:-2.5e9}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.15}"

SPEC_ARGS=(--speculative-draft-model "$DRAFT" --speculative-tree)
if [[ "${NO_SPEC:-0}" == "1" ]]; then SPEC_ARGS=(); fi

echo "kvboost (best setup — RTX 3060)"
echo "  model:            $MODEL"
echo "  draft:            ${NO_SPEC:+<disabled>}${NO_SPEC:-$DRAFT}"
echo "  port:             $PORT"
echo "  attention:        flash_attention_2 (auto-fallback to sdpa)"
echo "  recompute:        cacheblend_sparse  (faithful selective recompute)"
echo "  kv-cache-bits:    8                  (int8 KV → 2× reuse capacity)"
echo "  max-cache-bytes:  $MAX_CACHE_BYTES"
echo "  speculative:      ${NO_SPEC:+off}${NO_SPEC:-tree (auto mode-select)}"
echo "  oom planning:     on (safety_margin=$SAFETY_MARGIN)"
echo

# Why each flag:
#   --attn-impl auto
#       Tries FlashAttention-2 (Ampere wheel; faster, lower-memory prefill →
#       better TTFT and input throughput), silently falls back to sdpa if the
#       FA2 wheel isn't installed. Use --attn-impl flash_attention_2 to REQUIRE
#       it (errors loudly if missing) once you've confirmed the wheel.
#   --speculative-tree --speculative-draft-model
#       SpecBlock-inspired tree speculative decoding — the decode-throughput
#       lever. On bandwidth-bound hardware (3060), accepting several tokens per
#       target forward amortizes the per-token weight read → multiplies decode
#       tok/s. Auto mode-selector picks none/flat/tree per request.
#   --recompute-strategy cacheblend_sparse
#       Faithful CacheBlend: recompute only high-deviation tokens. The "faster
#       TTFT on reused context" feature. NOTE: on a pure shared-PREFIX workload
#       (this coding benchmark), --recompute-strategy none reuses prefix KV at
#       ~zero cost like vLLM prefix caching; cacheblend_sparse's edge is the
#       OUT-OF-ORDER RAG workload (bench_hf.py). Try both.
#   --kv-cache-bits 8
#       int8 KV STORAGE → ~2× cached-chunk capacity + less memory pressure.
#       (Note: it dequants to fp16 for compute, so it adds reuse capacity, not
#       decode bandwidth — that lever is weight quant, see below.)
#   OOM planner (on by default) + --planner-safety-margin
#       Per-request peak prediction → fitting chunk_size/kv_bits or clean 413.
#   (automatic: O(n) detok, chunked CacheBlend forward, streaming usage,
#    static decode input buffers.)
exec python -m kvboost.server \
    --model "$MODEL" \
    --dtype float16 \
    --attn-impl auto \
    --recompute-strategy cacheblend_sparse \
    --kv-cache-bits 8 \
    --max-cache-bytes "$MAX_CACHE_BYTES" \
    --planner-safety-margin "$SAFETY_MARGIN" \
    --max-batch-size 1 \
    "${SPEC_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"

# ── Optional add-ons (uncomment / set env to enable) ─────────────────────────
# WEIGHT QUANTIZATION (the biggest raw decode lever on a 3060): point --model
# at an AWQ/GPTQ Int4 checkpoint — transformers loads it with Marlin int4 GEMM
# on Ampere automatically (~4× less weight bandwidth → up to ~4× decode ceiling,
# 60→~240 tok/s for 3B). No extra flag; the engine detects quantized weights:
#     MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ ./start_kvboost.sh
#
# torch.compile (--compile): CUDA graphs + fusion erase per-token launch
# overhead → faster DECODE. CAVEAT: it recompiles per new PREFILL length, so it
# can HURT TTFT on this varying-prompt benchmark and adds a one-time first-
# request compile cost. Best for decode-bound / fixed-shape serving, not the
# TTFT ramp. Add:  --compile
#
# Oversized-prompt policy for the OOM ramp — complete-by-truncation vs 413:
#     --auto-truncate
