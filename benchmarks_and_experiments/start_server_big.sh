#!/usr/bin/env bash
# Launch kvboost-server tuned for a "big model on a small GPU" setup:
#
#   target = Qwen2.5-32B-Instruct-AWQ   (~19 GB packed — won't fit resident)
#   draft  = Qwen2.5-1.5B-Instruct-AWQ  (resident, drives speculative decoding)
#   GPU    = 12 GB (e.g. RTX 3060 12 GB; any Ampere+ with ≥12 GB works)
#
# Streaming math
# --------------
# Qwen2.5-32B has 64 hidden layers. With keep_first_k=9 + keep_last_k=9 we hold
# 18 layers resident; the other 46 stream their projection weights from pinned
# host RAM per forward. Embeddings + LM head + layernorms stay resident.
# Combined with the 1.5B-AWQ draft, KV cache, and CUDA scratch, this fits in
# ~10-11 GB of VRAM, leaving margin for prompt growth.
#
# Honest expectations on RTX 3060 12 GB (Ampere sm_86, PCIe 4.0)
# --------------------------------------------------------------
# Single request, warm:
#   decode_only_tok/s : 2.5 - 4.0   (verify-bound, what speculative delivers)
#   wall  tok/s       : 2.0 - 2.8   (after draft overhead)
#   TTFT (warm prefix): 0.5 - 2 s
#   TTFT (cold prefix): 30 - 80 s   (one-time per unique 1k-2k chunk)
#
# Concurrency wins are limited — streaming is PCIe-bound, so 4 concurrent
# clients don't run 4× faster. Expect maybe 1.5-2× throughput at concurrency=4
# before PCIe saturates.
#
# Faster alternatives if 32B isn't a hard requirement:
#   - Qwen2.5-14B-AWQ      (~8 GB packed, fully resident, 15-25 tok/s)
#   - Qwen2.5-7B-AWQ       (~4.5 GB packed, fully resident, 30-45 tok/s)
# Both are MUCH faster but obviously worse outputs.
#
# Usage
# -----
#   ./start_server_big.sh                          # 32B + 1.5B draft, port 8000
#   PORT=8001 ./start_server_big.sh                # different port
#   TARGET=Qwen/Qwen2.5-32B-Instruct-AWQ \
#     DRAFT=Qwen/Qwen2.5-1.5B-Instruct-AWQ \
#     ./start_server_big.sh                        # override models
#   NO_SPEC=1 ./start_server_big.sh                # disable speculative
#
# Pair with the parallel client
# -----------------------------
#   python sharegpt_3way/run_kvboost_server.py \
#       --server-url http://localhost:8000 \
#       --concurrency 4 --n-samples 500

set -euo pipefail

# ── Models ─────────────────────────────────────────────────────────────────
TARGET="${TARGET:-Qwen/Qwen2.5-32B-Instruct-AWQ}"
DRAFT="${DRAFT:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}"

# ── Streaming residency ────────────────────────────────────────────────────
# 9+9=18 resident layers is the published sweet spot for 32B-AWQ on 12 GB
# (README:473). Drop to 4+4 if you have 8 GB; bump to 14+14 if you have 16 GB.
KEEP_FIRST_K="${KEEP_FIRST_K:-9}"
KEEP_LAST_K="${KEEP_LAST_K:-9}"
# 4 slots gives full async prefetch overlap — wraps PCIe transfers behind
# compute. Drop to 2 if you hit VRAM pressure.
N_STAGING_SLOTS="${N_STAGING_SLOTS:-4}"
# `marlin` is the INT4 tensor-core kernel — only path that gets close to peak
# on Ampere+. `auto` falls back silently to slower kernels if Marlin doesn't
# load; forcing the choice surfaces the failure at startup instead.
QUANT_KERNEL="${QUANT_KERNEL:-marlin}"

# ── Speculative decoding ───────────────────────────────────────────────────
# At ~50% acceptance gamma=5 commits 3+ tokens per cycle, amortizing the
# per-cycle target DMA. Without spec a 32B-streamed forward is ~1 tok/cycle.
GAMMA="${GAMMA:-5}"
SPEC_MODE="${SPEC_MODE:-greedy}"

# ── KV cache ───────────────────────────────────────────────────────────────
# 1.5 GB cache. Bump if you have headroom and run high-reuse workloads.
MAX_CACHE_BYTES="${MAX_CACHE_BYTES:-1.5e9}"
KV_BITS="${KV_BITS:-16}"
# Selective beats cacheblend on speed; cacheblend wins on accuracy at >80%
# reuse. For "usable everyday" the selective tradeoff is right.
RECOMPUTE="${RECOMPUTE:-selective}"

# ── Batching ───────────────────────────────────────────────────────────────
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-4}"
BATCH_WINDOW_MS="${BATCH_WINDOW_MS:-20}"
MAX_QUEUE_SIZE="${MAX_QUEUE_SIZE:-64}"

# ── Server ─────────────────────────────────────────────────────────────────
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WARM_TEXT="${WARM_TEXT:-You are a helpful assistant. Be concise and accurate.}"

# ── Build the command ──────────────────────────────────────────────────────
ARGS=(
  --model               "${TARGET}"
  --awq-streaming
  --streaming-mode      partial_resident
  --keep-first-k        "${KEEP_FIRST_K}"
  --keep-last-k         "${KEEP_LAST_K}"
  --streaming-quant-kernel "${QUANT_KERNEL}"
  --recompute-strategy  "${RECOMPUTE}"
  --chunk-size          128
  --kv-cache-bits       "${KV_BITS}"
  --max-cache-bytes     "${MAX_CACHE_BYTES}"
  --max-batch-size      "${MAX_BATCH_SIZE}"
  --batch-window-ms     "${BATCH_WINDOW_MS}"
  --max-queue-size      "${MAX_QUEUE_SIZE}"
  --workers             1
  --always-warm         "${WARM_TEXT}"
  --host                "${HOST}"
  --port                "${PORT}"
)

# Optional: dedicated AWQ staging-slots arg if the server CLI exposes it
# (older versions consume it via env). Pass through if your build has it.
[[ -n "${N_STAGING_SLOTS:-}" ]] && export KVBOOST_N_STAGING_SLOTS="${N_STAGING_SLOTS}"

if [[ -z "${NO_SPEC:-}" ]]; then
  ARGS+=(
    --speculative-draft-model "${DRAFT}"
    --speculative-gamma       "${GAMMA}"
    --speculative-mode        "${SPEC_MODE}"
  )
fi

# ── ulimit: AWQ streaming pins ~12-15 GB of host RAM ───────────────────────
# Containers often cap RLIMIT_MEMLOCK at 64 MB which makes pinned-host
# allocation silently fall back to pageable + synchronous H2D (kills overlap).
# Try to raise it; ignore failure if we don't have permission.
if ! ulimit -l unlimited 2>/dev/null; then
  cur="$(ulimit -l 2>/dev/null || echo unknown)"
  echo "[warn] could not raise RLIMIT_MEMLOCK (currently: ${cur})." >&2
  echo "[warn] If you see 'mlock failed' or async prefetch turns off, run as" >&2
  echo "[warn] root or set: sudo prlimit --memlock=unlimited:unlimited --pid=\$\$" >&2
fi

echo "════════════════════════════════════════════════════════════════════"
echo "  kvboost-server — big model on small GPU"
echo "════════════════════════════════════════════════════════════════════"
echo "  target          = ${TARGET}"
echo "  draft           = $([ -z "${NO_SPEC:-}" ] && echo "${DRAFT} (gamma=${GAMMA})" || echo "(disabled)")"
echo "  residency       = partial_resident, keep_first/last=${KEEP_FIRST_K}/${KEEP_LAST_K}"
echo "  staging slots   = ${N_STAGING_SLOTS}"
echo "  quant kernel    = ${QUANT_KERNEL}  (forced; will error if not buildable)"
echo "  recompute       = ${RECOMPUTE}"
echo "  KV bits / cache = ${KV_BITS} / ${MAX_CACHE_BYTES} bytes"
echo "  batching        = max=${MAX_BATCH_SIZE}, window=${BATCH_WINDOW_MS}ms, queue=${MAX_QUEUE_SIZE}"
echo "  listening       = http://${HOST}:${PORT}"
echo "════════════════════════════════════════════════════════════════════"
echo "  Expected on RTX 3060 12 GB: 2.0-2.8 wall tok/s with spec, ~0.9 without."
echo "  First request will be slow (cold cache + first DMA cycle); subsequent"
echo "  requests with the same prefix hit the warm path."
echo "════════════════════════════════════════════════════════════════════"

exec kvboost-server "${ARGS[@]}"
