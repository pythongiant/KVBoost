#!/usr/bin/env bash
#
# ShareGPT 3-way benchmark — KVBoost vs vLLM vs llama.cpp
#
# Same workload (500 ShareGPT conversations, multi-turn, Qwen2.5 7B target +
# 1.5B draft, gamma=5) across three serving stacks. Runs them sequentially
# so each backend has the GPU to itself.
#
# Usage:
#   ./run.sh                                # all three, default 500 samples
#   ./run.sh --n-samples 50                 # quick smoke test
#   ONLY=kvboost ./run.sh                   # one backend only
#   ONLY=kvboost,vllm ./run.sh              # subset
#   LLAMACPP_MODEL=/path/target.gguf LLAMACPP_DRAFT=/path/draft.gguf ./run.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"
EXTRA_ARGS="${*}"
ONLY="${ONLY:-kvboost,vllm,llamacpp}"

# llama.cpp GGUF paths — override via env if your local layout differs.
LLAMACPP_MODEL="${LLAMACPP_MODEL:-${HOME}/models/qwen2.5-7b-instruct-q4_k_m.gguf}"
LLAMACPP_DRAFT="${LLAMACPP_DRAFT:-${HOME}/models/qwen2.5-1.5b-instruct-q4_k_m.gguf}"

mkdir -p "${SCRIPT_DIR}/results"

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
    --n-samples 500 \
    --min-turns 3 \
    --max-turns 8 \
    --max-context-tokens 6000 \
    --max-new-tokens 128 \
    --gamma 5 \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --draft-model "Qwen/Qwen2.5-1.5B-Instruct"

run_backend vllm run_vllm.py \
    --n-samples 500 \
    --min-turns 3 \
    --max-turns 8 \
    --max-context-tokens 6000 \
    --max-new-tokens 128 \
    --gamma 5 \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --draft-model "Qwen/Qwen2.5-1.5B-Instruct" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192

run_backend llamacpp run_llamacpp.py \
    --n-samples 500 \
    --min-turns 3 \
    --max-turns 8 \
    --max-context-tokens 6000 \
    --max-new-tokens 128 \
    --gamma 5 \
    --model-path "${LLAMACPP_MODEL}" \
    --draft-model-path "${LLAMACPP_DRAFT}" \
    --tokenizer-id "Qwen/Qwen2.5-7B-Instruct" \
    --n-ctx 8192 \
    --n-gpu-layers -1

echo ""
echo "========================================"
echo "  3-way comparison"
echo "========================================"
"${PYTHON}" "${SCRIPT_DIR}/compare.py"

echo ""
echo "Results:    ${SCRIPT_DIR}/results/{kvboost,vllm,llamacpp}.json"
echo "Plot:       ${SCRIPT_DIR}/results/3way_summary.png"
