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
    LLAMACPP_REPO="${LLAMACPP_REPO:-Qwen/Qwen2.5-7B-Instruct-GGUF}"
    LLAMACPP_FILE="${LLAMACPP_FILE:-qwen2.5-7b-instruct-q4_k_m.gguf}"
    LLAMACPP_DRAFT_REPO="${LLAMACPP_DRAFT_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
    LLAMACPP_DRAFT_FILE="${LLAMACPP_DRAFT_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
    LLAMACPP_CTX="${LLAMACPP_CTX:-4096}"
    MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-4096}"
    N_SAMPLES="${N_SAMPLES:-500}"
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
    LLAMACPP_REPO="${LLAMACPP_REPO:-Qwen/Qwen2.5-7B-Instruct-GGUF}"
    LLAMACPP_FILE="${LLAMACPP_FILE:-qwen2.5-7b-instruct-q4_k_m.gguf}"
    LLAMACPP_DRAFT_REPO="${LLAMACPP_DRAFT_REPO:-Qwen/Qwen2.5-1.5B-Instruct-GGUF}"
    LLAMACPP_DRAFT_FILE="${LLAMACPP_DRAFT_FILE:-qwen2.5-1.5b-instruct-q4_k_m.gguf}"
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

# Resolved GGUF paths: respect explicit LLAMACPP_MODEL/LLAMACPP_DRAFT overrides
# (back-compat with the older interface), else derive from REPO/FILE pair.
GGUF_DIR="${GGUF_DIR:-${HOME}/models}"
LLAMACPP_MODEL="${LLAMACPP_MODEL:-${GGUF_DIR}/${LLAMACPP_FILE}}"
LLAMACPP_DRAFT="${LLAMACPP_DRAFT:-${GGUF_DIR}/${LLAMACPP_DRAFT_FILE}}"

# ── GGUF auto-download (vLLM-style: point at HF repo, fetch on demand) ─
# vLLM resolves HF repos transparently when you pass --model. llama.cpp
# expects on-disk paths, so we mirror that ergonomics here: if the GGUF
# isn't present, download it before launching the runner.
ensure_gguf() {
    local repo="$1" filename="$2" dest="$3"
    if [[ -f "${dest}" ]]; then
        return 0
    fi
    mkdir -p "$(dirname "${dest}")"
    echo ">>> GGUF missing: ${dest}"
    echo "    Fetching ${filename} from ${repo} ..."
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download "${repo}" "${filename}" \
            --local-dir "$(dirname "${dest}")" \
            --local-dir-use-symlinks False
    else
        "${PYTHON}" - <<PY
from huggingface_hub import hf_hub_download
import os, shutil
p = hf_hub_download(repo_id="${repo}", filename="${filename}")
dest = "${dest}"
os.makedirs(os.path.dirname(dest), exist_ok=True)
if os.path.realpath(p) != os.path.realpath(dest):
    shutil.copy(p, dest)
print("ok:", dest)
PY
    fi
    if [[ ! -f "${dest}" ]]; then
        echo "ERROR: failed to fetch ${filename}; expected at ${dest}" >&2
        return 1
    fi
}

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

if [[ ",${ONLY}," == *",llamacpp,"* ]]; then
    ensure_gguf "${LLAMACPP_REPO}"       "${LLAMACPP_FILE}"       "${LLAMACPP_MODEL}"
    ensure_gguf "${LLAMACPP_DRAFT_REPO}" "${LLAMACPP_DRAFT_FILE}" "${LLAMACPP_DRAFT}"
fi

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
