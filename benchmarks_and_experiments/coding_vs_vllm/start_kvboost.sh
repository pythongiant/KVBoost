#!/usr/bin/env bash
# Launch kvboost in its BEST setup for the coding benchmark — a FAIR match to
# vLLM's usual setup: fp16, FlashAttention-2, NO speculative decoding (vLLM's
# usual setup runs none either). The kvboost features on show are KV reuse
# (faster TTFT, cacheblend_sparse) and OOM recovery (the OOM planner).
#
# Run this, then in another shell:
#   python bench_coding.py --backend kvboost --url http://localhost:9000 \
#       --model "$MODEL" --mode both --out kvboost.json
# Stop it (Ctrl-C) before launching vLLM — one model fits the GPU at a time.
#
# Override via env: MODEL=... PORT=... MAX_CACHE_BYTES=... SPEC=1 (opt-in spec)

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-9000}"
# KV-cache budget for cross-request chunk reuse. On a 12 GB 3060: ~6 GB fp16
# model leaves ~6 GB → 3 GB cache keeps activation headroom. Lower for the
# OOM-stress run to make the planner's adaptation more visible (~1e9).
MAX_CACHE_BYTES="${MAX_CACHE_BYTES:-3e9}"
SAFETY_MARGIN="${SAFETY_MARGIN:-0.15}"

# Speculative decoding is OFF by default to keep the vLLM comparison fair
# (vLLM's usual setup runs no speculation). Opt in with SPEC=1 only if you
# WANT the kvboost-with-spec vs vanilla-vLLM comparison.
SPEC_ARGS=()
if [[ "${SPEC:-0}" == "1" ]]; then
    SPEC_ARGS=(--speculative-draft-model "${DRAFT:-Qwen/Qwen2.5-0.5B-Instruct}" \
               --speculative-tree)
fi

echo "kvboost (best setup — RTX 3060, fair vs vLLM)"
echo "  model:            $MODEL"
echo "  port:             $PORT"
echo "  attention:        flash_attention_2 (auto-fallback to sdpa)"
echo "  recompute:        cacheblend_sparse  (faithful selective recompute)"
echo "  kv-cache-bits:    8                  (int8 KV → 2× reuse capacity)"
echo "  max-cache-bytes:  $MAX_CACHE_BYTES"
echo "  speculative:      ${SPEC:+tree (opt-in via SPEC=1)}${SPEC:-off (fair vs vLLM)}"
echo "  oom planning:     on (safety_margin=$SAFETY_MARGIN)"
echo

# Why each flag:
#   --attn-impl auto
#       Tries FlashAttention-2 (Ampere wheel; faster, lower-memory prefill →
#       better TTFT and input throughput), silently falls back to sdpa if the
#       FA2 wheel isn't installed. vLLM uses an FA2-class kernel too, so this
#       keeps the comparison fair. Use --attn-impl flash_attention_2 to REQUIRE
#       it (errors loudly if missing) once you've confirmed the wheel.
#   --recompute-strategy cacheblend_sparse
#       Faithful CacheBlend: recompute only high-deviation tokens. The "faster
#       TTFT on reused context" feature. NOTE: on a pure shared-PREFIX workload
#       (this coding benchmark), --recompute-strategy none reuses prefix KV at
#       ~zero cost like vLLM prefix caching; cacheblend_sparse's edge is the
#       OUT-OF-ORDER RAG workload (bench_hf.py). Try both.
#   --kv-cache-bits 8
#       int8 KV STORAGE → ~2× cached-chunk capacity + less memory pressure.
#       (Dequants to fp16 for compute, so it adds reuse capacity, not decode
#       bandwidth — that lever is weight quant, see below.)
#   OOM planner (on by default) + --planner-safety-margin
#       Per-request peak prediction → fitting chunk_size/kv_bits or clean 413.
#   (automatic: O(n) detok, chunked CacheBlend forward, streaming usage,
#    static decode input buffers.)
exec python -m kvboost.server \
    --model "$MODEL" \
    --dtype float16 \
    --attn-impl auto \
    --recompute-strategy cacheblend_sparse \
    --chunk-boundary-window 32\
    --kv-cache-bits 8 \
    --max-cache-bytes "$MAX_CACHE_BYTES" \
    --planner-safety-margin "$SAFETY_MARGIN" \
    --max-batch-size 1 \
    "${SPEC_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT"
    

# ── Optional add-ons (set env / uncomment to enable) ─────────────────────────
# SPECULATIVE DECODING (NOT fair vs a no-spec vLLM — opt-in only): SPEC=1
# enables SpecBlock-inspired tree speculative decoding (decode-throughput
# lever; auto mode-select per request). Needs the ~1 GB draft model + VRAM:
#     SPEC=1 ./start_kvboost.sh
#
# WEIGHT QUANTIZATION — Marlin int4 (the biggest raw decode lever): point
# --model at an AWQ/GPTQ Int4 checkpoint — transformers loads it with the
# AWQ/Marlin int4 GEMM CUDA kernels on Ampere automatically (~4× less weight
# bandwidth → up to ~4× decode ceiling):
#     MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ ./start_kvboost.sh
# FAIR int4-vs-int4: run vLLM on the SAME AWQ checkpoint (it also uses Marlin):
#     MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ ./start_vllm.sh
#
# FLASHINFER decode-attention (real CUDA decode-attention kernel; helps most at
# long context where KV reads dominate): --attn-impl flashinfer. Routes only the
# decode step through FlashInfer, SDPA for prefill + fallback, with a one-time
# numerical self-check. Needs `pip install flashinfer-python` (install_deps.sh).
#
# CUDA-GRAPH DECODE (--cuda-graph-decode): THE fix for the dominant decode cost
# on a 3060 — the per-token launch overhead (~36 of the ~56 ms/token). Captures
# the single-token decode step against a static KV cache and replays it; reuse-
# based prefill is preserved. Self-checked vs eager (multi-step greedy) with
# eager fallback, so a bad capture degrades to correct-but-slow, never wrong.
# Expected: decode ~17 -> ~40+ tok/s. Validate on the box (compare output to a
# run without the flag). STACKS with Marlin int4 (MODEL=...-AWQ) → ~17 -> ~90+:
#     --cuda-graph-decode
#     MODEL=Qwen/Qwen2.5-3B-Instruct-AWQ ... --cuda-graph-decode   # both levers
#
# torch.compile (--compile): alternative CUDA-graph path via torch; recompiles
# per new PREFILL length so it can HURT this varying-prompt TTFT benchmark.
# Ignored if --cuda-graph-decode is set. Prefer --cuda-graph-decode.
#
# MULTI-TURN CacheBlend run (where CacheBlend should BEAT vLLM prefix caching):
# the bench --mode multiturn workload reshuffles the in-context files each turn,
# so the same files recur OUT OF ORDER — prefix caching misses, CacheBlend
# reuses. For kvboost to actually match a repositioned file, add content-aligned
# chunking so a moved file still chunks identically (else fixed 128-tok cuts
# misalign and miss):
#     ./start_kvboost.sh   # add: --chunk-boundary-window 32
#   then: python bench_coding.py --backend kvboost --url http://localhost:9000 \
#             --model "$MODEL" --mode multiturn --out kvboost_mt.json
#
# Oversized-prompt policy for the OOM ramp — complete-by-truncation vs 413:
#     --auto-truncate
