#!/usr/bin/env bash
# Launch kvboost in its FASTEST setup on the RTX 3060 — speed over fairness.
# Stacks every working throughput/latency lever for the coding benchmark:
#   * Marlin int4 weight quant (AWQ model)    — biggest decode-bandwidth lever (~4×)
#   * Tree speculative decoding (draft model) — multi-token/step decode lever
#   * recompute=none                          — zero-cost shared-prefix reuse (TTFT)
#   * int8 KV storage + OOM planner           — more reuse capacity, no crashes
#
# AWQ LOADING (important on this box): the plain *resident* AWQ load goes through
# transformers' AWQ quantizer, which pulls in a `gptqmodel` that's mismatched
# with the installed transformers ("module 'transformers.utils.hub' has no
# attribute 'create_repo'") and crashes on import. kvboost's OWN AWQ loader
# sidesteps it: `--awq-streaming --streaming-mode full_resident` keeps all int4
# weights ON the GPU (no DMA) and uses the Marlin int4 GEMM, never touching
# transformers' AWQ path. So for an AWQ/GPTQ MODEL this script auto-uses that
# loader. Tradeoff: the streaming path owns attention, so --attn-impl (incl. our
# Triton 'sage' kernel) is IGNORED there → SDPA prefill. int4 ≫ sage, so this is
# still the fastest config that actually launches. An fp16 MODEL uses the
# resident path + sage instead. (To regain sage WITH int4, repair the env — see
# the foot of this file — and force RESIDENT=1.)
#
# Run this, then in another shell:
#   python bench_coding.py --backend kvboost --url http://localhost:9000 \
#       --model "$MODEL" --mode both --out kvboost.json
# Stop it (Ctrl-C) before launching vLLM — one model fits the GPU at a time.
#
# Override via env:
#   MODEL=... PORT=... MAX_CACHE_BYTES=... SPEC=0 (disable spec) DRAFT=...
#   ATTN=flashinfer  RECOMPUTE=cacheblend_sparse  STREAMING_MODE=...  QUANT_KERNEL=marlin
#   RESIDENT=1 (force the transformers resident load even for an AWQ model)

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
STREAMING_MODE="${STREAMING_MODE:-full_resident}"   # all weights on GPU, no DMA
QUANT_KERNEL="${QUANT_KERNEL:-auto}"                 # auto = probe Marlin first

# Tree speculative decoding is ON by default here (it's a speed setup). The
# draft MUST be an AWQ checkpoint: kvboost's DraftModel always loads it through
# StreamingCausalLM (which bypasses transformers' AWQ quantizer), so a plain
# fp16 draft fails with "No AWQ quantization config found". Qwen2.5-1.5B-AWQ
# (~1 GB) is the draft the repo's other benchmarks use. Disable with SPEC=0.
SPEC_ARGS=()
if [[ "${SPEC:-1}" == "1" ]]; then
    SPEC_ARGS=(--speculative-draft-model "${DRAFT:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}" \
               --speculative-tree)
    SPEC_DESC="tree (draft ${DRAFT:-Qwen/Qwen2.5-1.5B-Instruct-AWQ})"
else
    SPEC_DESC="off (SPEC=0)"
fi

# Choose the load path. AWQ/GPTQ → kvboost's streaming loader (bypasses the
# broken transformers AWQ quantizer); fp16 → resident path + sage.
case "$MODEL" in
    *AWQ*|*awq*|*GPTQ*|*gptq*|*Int4*|*int4*|*INT4*)
        if [[ "${RESIDENT:-0}" == "1" ]]; then
            LOAD_ARGS=(--attn-impl "$ATTN")
            ATTN_DESC="$ATTN (resident AWQ — needs a repaired gptqmodel/transformers env)"
        else
            LOAD_ARGS=(--awq-streaming --streaming-mode "$STREAMING_MODE" \
                       --streaming-quant-kernel "$QUANT_KERNEL")
            ATTN_DESC="SDPA (AWQ streaming owns attention; --attn-impl ignored)"
        fi
        ;;
    *)
        LOAD_ARGS=(--attn-impl "$ATTN")
        ATTN_DESC="$ATTN (INT8 SageAttention prefill; self-check → SDPA)"
        ;;
esac

echo "kvboost (FASTEST setup — RTX 3060, speed over fairness)"
echo "  model:            $MODEL  (int4 Marlin GEMM if AWQ/GPTQ)"
echo "  port:             $PORT"
echo "  load path:        ${LOAD_ARGS[*]}"
echo "  attention:        $ATTN_DESC"
echo "  recompute:        $RECOMPUTE  (zero-cost shared-prefix reuse = fastest TTFT)"
echo "  kv-cache-bits:    8                  (int8 KV → 2× reuse capacity)"
echo "  max-cache-bytes:  $MAX_CACHE_BYTES"
echo "  speculative:      $SPEC_DESC"
echo "  oom planning:     on (safety_margin=$SAFETY_MARGIN)"
echo

# Why each flag (impact order on a 3060):
#   MODEL=...-AWQ  (the #1 raw lever)
#       int4 weight quant → Marlin int4 GEMM on Ampere (~4× less weight bandwidth
#       → up to ~4× the decode ceiling). Loaded here via --awq-streaming
#       --streaming-mode full_resident (kvboost's own AWQ→Marlin loader, all
#       weights resident on GPU) to dodge the broken transformers AWQ quantizer.
#   --speculative-tree (+ draft)  (the #2 decode lever)
#       SpecBlock-inspired tree speculative decoding — verifies several drafted
#       tokens per target step; auto mode-select per request. Decode throughput.
#   --recompute-strategy none  (fastest TTFT on shared prefix)
#       Reuses prefix KV at ~zero cost (like vLLM prefix caching) — lossless on
#       this benchmark's shared prefix. Set RECOMPUTE=cacheblend_sparse for the
#       OUT-OF-ORDER multiturn/RAG workload (faithful selective recompute).
#   --kv-cache-bits 8
#       int8 KV STORAGE → ~2× cached-chunk capacity. (Dequants to fp16 for
#       compute — adds reuse capacity, not decode bandwidth; that's weight quant.)
#   OOM planner (on) + --planner-safety-margin
#       Per-request peak prediction → fits chunk_size/kv_bits or a clean 413.
exec python -m kvboost.server \
    --model "$MODEL" \
    --dtype float16 \
    "${LOAD_ARGS[@]}" \
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
# FASTEST-that-launches (default): AWQ int4 via kvboost's streaming loader. If
# the streaming AWQ load itself errors (rare on a pure-transformer fp16 AWQ like
# Qwen2.5), fall back to fp16 — you keep spec + sage, lose only int4:
#     MODEL=Qwen/Qwen2.5-3B-Instruct ./start_kvboost.sh
#
# REGAIN sage WITH int4 (resident AWQ): repair the env so transformers' AWQ path
# works, then force the resident load:
#     pip install -U "gptqmodel" "transformers"        # realign the two, OR
#     pip uninstall -y gptqmodel && pip install autoawq # AWQ via autoawq, not gptqmodel
#     RESIDENT=1 ./start_kvboost.sh
#   (Confirm in the logs you see "sage self-check passed".)
#
# FAIR int4-vs-int4 comparison: run vLLM on the SAME AWQ checkpoint (Marlin too):
#     MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ ./start_vllm.sh
#
# FLASHINFER decode-attention (fp16 model, use with SPEC=0): ATTN=flashinfer.
# Routes only single-token DECODE through FlashInfer's CUDA kernel (SDPA prefill
# + fallback, one-time self-check). Helps most at long context.
#
# CUDA-GRAPH DECODE (--cuda-graph-decode): LEFT OFF here on purpose — it caused
# recompile thrash on this box and was removed from this setup (commit "Remove
# cuda graph decode"). It targets per-token launch overhead (~36 of ~56 ms/token)
# and stacks with int4, so if you've fixed the re-capture thrash, add it back and
# validate output vs a run without it:
#     ... --cuda-graph-decode
#
# MULTI-TURN CacheBlend run (where CacheBlend beats vLLM prefix caching): the
# --mode multiturn workload reshuffles in-context files each turn (same files,
# OUT OF ORDER). Use the faithful recompute path (content-aligned chunking is
# already on via --chunk-boundary-window 32):
#     RECOMPUTE=cacheblend_sparse ./start_kvboost.sh
#   then: python bench_coding.py --backend kvboost --url http://localhost:9000 \
#             --model "$MODEL" --mode multiturn --out kvboost_mt.json
#
# Oversized-prompt policy for the OOM ramp — complete-by-truncation vs 413:
#     --auto-truncate
