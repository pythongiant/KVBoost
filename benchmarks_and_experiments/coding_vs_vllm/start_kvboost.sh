#!/usr/bin/env bash
# Launch kvboost in its FASTEST setup on the RTX 3060 — speed over fairness.
# This stacks every working throughput/latency lever for the coding benchmark:
#   * Marlin int4 weight quant  (AWQ model)      — biggest decode-bandwidth lever
#   * Tree speculative decoding (draft model)    — multi-token/step decode lever
#   * INT8 SageAttention prefill (Triton 'sage') — faster TTFT, self-checks → SDPA
#   * recompute=none                             — zero-cost shared-prefix reuse
#   * int8 KV storage + OOM planner              — more reuse capacity, no crashes
#
# NOTE this is NOT a fair vs-vLLM config (int4 + spec). For an apples-to-apples
# run, point vLLM at the SAME AWQ checkpoint (it also uses Marlin) — see below.
#
# Run this, then in another shell:
#   python bench_coding.py --backend kvboost --url http://localhost:9000 \
#       --model "$MODEL" --mode both --out kvboost.json
# Stop it (Ctrl-C) before launching vLLM — one model fits the GPU at a time.
#
# Override via env:
#   MODEL=... PORT=... MAX_CACHE_BYTES=... SPEC=0 (disable spec) DRAFT=...
#   ATTN=flashinfer (decode-attn instead of sage) RECOMPUTE=cacheblend_sparse

set -euo pipefail

# int4 (Marlin) by default — the single biggest decode lever on Ampere (~4× less
# weight bandwidth). Override MODEL=Qwen/Qwen2.5-3B-Instruct for plain fp16.
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct-AWQ}"
PORT="${PORT:-9000}"
# KV-cache budget for cross-request chunk reuse. The int4 model is only ~2 GB
# (vs ~6 GB fp16) so on a 12 GB 3060 there's far more room for cache → bigger
# reuse hit-rate → better TTFT. The OOM planner backstops an over-large value.
MAX_CACHE_BYTES="${MAX_CACHE_BYTES:-5e9}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.15}"
ATTN="${ATTN:-sage}"
RECOMPUTE="${RECOMPUTE:-none}"

# Tree speculative decoding is ON by default here (it's a speed setup). Needs
# the ~1 GB fp16 draft model + VRAM. Disable with SPEC=0 for a no-spec run.
SPEC_ARGS=()
if [[ "${SPEC:-1}" == "1" ]]; then
    SPEC_ARGS=(--speculative-draft-model "${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}" \
               --speculative-tree)
    SPEC_DESC="tree (draft ${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct})"
else
    SPEC_DESC="off (SPEC=0)"
fi

echo "kvboost (FASTEST setup — RTX 3060, speed over fairness)"
echo "  model:            $MODEL  (int4 Marlin GEMM if AWQ/GPTQ)"
echo "  port:             $PORT"
echo "  attention:        $ATTN  (INT8 SageAttention prefill; self-check → SDPA)"
echo "  recompute:        $RECOMPUTE  (zero-cost shared-prefix reuse = fastest TTFT)"
echo "  kv-cache-bits:    8                  (int8 KV → 2× reuse capacity)"
echo "  max-cache-bytes:  $MAX_CACHE_BYTES"
echo "  speculative:      $SPEC_DESC"
echo "  oom planning:     on (safety_margin=$SAFETY_MARGIN)"
echo

# Why each flag (impact order on a 3060):
#   MODEL=...-AWQ  (the #1 raw lever)
#       int4 weight quant → transformers loads the AWQ/Marlin int4 GEMM CUDA
#       kernels on Ampere automatically (~4× less weight bandwidth → up to ~4×
#       the decode ceiling). The 3 GB→~2 GB model also frees VRAM for KV cache.
#   --speculative-tree (+ draft)  (the #2 decode lever)
#       SpecBlock-inspired tree speculative decoding — verifies several drafted
#       tokens per target step; auto mode-select per request. Decode throughput.
#   --attn-impl sage  (prefill lever; pairs with spec)
#       INT8 SageAttention prefill via Triton (INT8 tensor-core QK^T on sm_86;
#       no nvcc/flash-attn build). Decode (q_len==1) delegates to SDPA. One-time
#       numerical self-check vs SDPA → permanent SDPA fallback on mismatch, so
#       worst case is the SDPA baseline, never wrong. Watch the log for
#       "sage self-check passed". Set ATTN=flashinfer instead to accelerate
#       single-token DECODE attention (better when SPEC=0 + long context).
#   --recompute-strategy none  (fastest TTFT on shared prefix)
#       Reuses prefix KV at ~zero cost (like vLLM prefix caching) — lossless on
#       this coding benchmark's shared prefix. Set RECOMPUTE=cacheblend_sparse
#       for the OUT-OF-ORDER multiturn/RAG workload (faithful selective recompute
#       where moved chunks would otherwise go stale).
#   --kv-cache-bits 8
#       int8 KV STORAGE → ~2× cached-chunk capacity. (Dequants to fp16 for
#       compute — adds reuse capacity, not decode bandwidth; that's weight quant.)
#   OOM planner (on) + --planner-safety-margin
#       Per-request peak prediction → fits chunk_size/kv_bits or a clean 413.
exec python -m kvboost.server \
    --model "$MODEL" \
    --dtype float16 \
    --attn-impl "$ATTN" \
    --recompute-strategy "$RECOMPUTE" \
    --chunk-boundary-window 32 \
    --kv-cache-bits 8 \
    --max-cache-bytes "$MAX_CACHE_BYTES" \
    --planner-safety-margin "$SAFETY_MARGIN" \
    --max-batch-size 1 \
    "${SPEC_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"


# ── Optional add-ons / alternatives ──────────────────────────────────────────
# FAIR int4-vs-int4 comparison: run vLLM on the SAME AWQ checkpoint (it also
# uses Marlin) so the only difference is the engine, not the weights:
#     MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ ./start_vllm.sh
#
# DISABLE the aggressive levers (toward the old fair-vs-vLLM baseline):
#     MODEL=Qwen/Qwen2.5-3B-Instruct SPEC=0 ATTN=auto RECOMPUTE=cacheblend_sparse ./start_kvboost.sh
#
# FLASHINFER decode-attention (use instead of sage when SPEC=0): ATTN=flashinfer.
# Routes only the single-token DECODE step through FlashInfer's CUDA kernel
# (SDPA prefill + fallback, one-time self-check). Helps most at long context
# where KV reads dominate. Needs `pip install flashinfer-python`.
#
# CUDA-GRAPH DECODE (--cuda-graph-decode): LEFT OFF here on purpose — it caused
# recompile thrash on this box and was removed from this setup (commit
# "Remove cuda graph decode"). It targets the per-token launch overhead (~36 of
# ~56 ms/token) and stacks with int4, so if you've since fixed the re-capture
# thrash it's a big decode-latency win — add it back and validate output vs a
# run without it:
#     ... --cuda-graph-decode
#
# MULTI-TURN CacheBlend run (where CacheBlend beats vLLM prefix caching): the
# --mode multiturn workload reshuffles in-context files each turn (same files,
# OUT OF ORDER) — prefix caching misses, CacheBlend reuses. Use the faithful
# recompute path + content-aligned chunking (already on via --chunk-boundary-
# window 32 so a moved file still chunks identically):
#     RECOMPUTE=cacheblend_sparse ./start_kvboost.sh
#   then: python bench_coding.py --backend kvboost --url http://localhost:9000 \
#             --model "$MODEL" --mode multiturn --out kvboost_mt.json
#
# Oversized-prompt policy for the OOM ramp — complete-by-truncation vs 413:
#     --auto-truncate
