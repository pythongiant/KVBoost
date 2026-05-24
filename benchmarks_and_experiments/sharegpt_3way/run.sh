#!/usr/bin/env bash
#
# ShareGPT 3-way benchmark — KVBoost vs vLLM vs llama.cpp
#
# Defaults are tuned for an RTX 3060 12 GB: 7B AWQ target + 1.5B AWQ draft,
# all three backends running 4-bit-class weights (AWQ for the PyTorch stacks,
# Q4_K_M GGUF for llama.cpp). This is the apples-to-apples comparison.
#
# To run the original fp16 config on a bigger GPU, set PROFILE=fp16-big.
#
# Common overrides:
#   ./run.sh                                # all three, default 100 samples
#   ./run.sh --n-samples 50                 # quick smoke test
#   ONLY=kvboost ./run.sh                   # one backend only
#   ONLY=kvboost,vllm ./run.sh              # subset
#   PROFILE=fp16-big ./run.sh               # 7B fp16 (needs ≥24 GB)
#   N_SAMPLES=500 ./run.sh                  # full 500-conversation run
#   LLAMACPP_MODEL=/path/target.gguf LLAMACPP_DRAFT=/path/draft.gguf ./run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${*}"
ONLY="${ONLY:-kvboost,vllm,llamacpp}"
PROFILE="${PROFILE:-awq-12gb}"

# ── Profile: model ids + memory budgets ────────────────────────────────
case "${PROFILE}" in
  awq-12gb)
    KVBOOST_MODEL="${KVBOOST_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
    KVBOOST_DRAFT="${KVBOOST_DRAFT:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}"
    KVBOOST_EXTRA=(
        --awq-streaming
        --streaming-mode partial_resident
        --keep-first-k 1024 --keep-last-k 1024
        --kv-cache-bits 8
        --max-cache-bytes 1.5e9
    )
    VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
    VLLM_DRAFT="${VLLM_DRAFT:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
    VLLM_MAX_LEN="${VLLM_MAX_LEN:-4096}"
    LLAMACPP_MODEL="${LLAMACPP_MODEL:-${HOME}/models/qwen2.5-7b-instruct-q4_k_m.gguf}"
    LLAMACPP_DRAFT="${LLAMACPP_DRAFT:-${HOME}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
    LLAMACPP_CTX="${LLAMACPP_CTX:-4096}"
    MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-4096}"
    N_SAMPLES="${N_SAMPLES:-100}"
    GAMMA="${GAMMA:-4}"
    ;;
  fp16-big)
    KVBOOST_MODEL="${KVBOOST_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    KVBOOST_DRAFT="${KVBOOST_DRAFT:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}"  # DraftModel is always AWQ-streamed
    KVBOOST_EXTRA=()
    VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
    VLLM_DRAFT="${VLLM_DRAFT:-Qwen/Qwen2.5-1.5B-Instruct}"
    VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.90}"
    VLLM_MAX_LEN="${VLLM_MAX_LEN:-8192}"
    LLAMACPP_MODEL="${LLAMACPP_MODEL:-${HOME}/models/qwen2.5-7b-instruct-q4_k_m.gguf}"
    LLAMACPP_DRAFT="${LLAMACPP_DRAFT:-${HOME}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"
    LLAMACPP_CTX="${LLAMACPP_CTX:-8192}"
    MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-6000}"
    N_SAMPLES="${N_SAMPLES:-500}"
    GAMMA="${GAMMA:-5}"
    ;;
  *)
    echo "Unknown PROFILE='${PROFILE}'. Valid: awq-12gb | fp16-big" >&2
    exit 1
    ;;
esac

# ── Shared workload shape — identical across all three backends ────────
MIN_TURNS="${MIN_TURNS:-3}"
MAX_TURNS="${MAX_TURNS:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5}"

mkdir -p "${SCRIPT_DIR}/results"

echo "=========================================="
echo "  PROFILE=${PROFILE}"
echo "  n_samples=${N_SAMPLES}  turns=${MIN_TURNS}-${MAX_TURNS}"
echo "  max_context_tokens=${MAX_CONTEXT_TOKENS}  max_new_tokens=${MAX_NEW_TOKENS}"
echo "  gamma=${GAMMA}"
echo "=========================================="

run_backend() {
    local name="$1"; shift
    if [[ ",${ONLY}," != *",${name},"* ]]; then
        echo ">>> skip ${name} (ONLY=${ONLY})"
        return
    fi
    echo ""
    echo "========================================"
    echo "  ${name}"
    echo "========================================"
    "${PYTHON}" "${SCRIPT_DIR}/$1" "${@:2}" ${EXTRA_ARGS}
}

run_backend kvboost run_kvboost.py \
    --n-samples "${N_SAMPLES}" \
    --min-turns "${MIN_TURNS}" \
    --max-turns "${MAX_TURNS}" \
    --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --gamma "${GAMMA}" \
    --model "${KVBOOST_MODEL}" \
    --draft-model "${KVBOOST_DRAFT}" \
    --progress-every "${PROGRESS_EVERY}" \
    --save-output-text \
    "${KVBOOST_EXTRA[@]}"

run_backend vllm run_vllm.py \
    --n-samples "${N_SAMPLES}" \
    --min-turns "${MIN_TURNS}" \
    --max-turns "${MAX_TURNS}" \
    --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --gamma "${GAMMA}" \
    --model "${VLLM_MODEL}" \
    --draft-model "${VLLM_DRAFT}" \
    --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
    --max-model-len "${VLLM_MAX_LEN}" \
    --progress-every "${PROGRESS_EVERY}" \
    --save-output-text

run_backend llamacpp run_llamacpp.py \
    --n-samples "${N_SAMPLES}" \
    --min-turns "${MIN_TURNS}" \
    --max-turns "${MAX_TURNS}" \
    --max-context-tokens "${MAX_CONTEXT_TOKENS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --gamma "${GAMMA}" \
    --model-path "${LLAMACPP_MODEL}" \
    --draft-model-path "${LLAMACPP_DRAFT}" \
    --tokenizer-id "Qwen/Qwen2.5-7B-Instruct" \
    --n-ctx "${LLAMACPP_CTX}" \
    --n-gpu-layers -1 \
    --progress-every "${PROGRESS_EVERY}" \
    --save-output-text

echo ""
echo "========================================"
echo "  3-way comparison"
echo "========================================"
"${PYTHON}" "${SCRIPT_DIR}/compare.py"

echo ""
echo "Results:    ${SCRIPT_DIR}/results/{kvboost,vllm,llamacpp}.json"
echo "Plot:       ${SCRIPT_DIR}/results/3way_summary.png"
