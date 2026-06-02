#!/usr/bin/env bash
# Launch vLLM in its USUAL serving setup for the coding benchmark — the
# standard OpenAI server with prefix caching (vLLM's cross-request reuse) and
# continuous batching (its default). Matched model + dtype to kvboost so the
# comparison is fair.
#
# Run this AFTER stopping the kvboost server (one model fits the GPU at a
# time), then in another shell:
#   python bench_coding.py --backend vllm --url http://localhost:8001 \
#       --model "$MODEL" --mode both --out vllm.json
#   # ... use the SAME --dataset/--n/--n-files/--contexts/--corpus-size as the
#   #     kvboost run so both backends see identical prompts.
#
# Override via env: MODEL=... PORT=... GPU_MEM_UTIL=... MAX_MODEL_LEN=...

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
PORT="${PORT:-8001}"
# vLLM pre-allocates this fraction of total VRAM for weights + its paged KV
# pool. 0.85 is the common production value.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
# Max admitted context. 32768 covers the throughput/TTFT workload. For the OOM
# ramp: a HIGH value (e.g. 131072) admits long prompts so they hit the runtime
# KV ceiling (real OOM); a LOW value makes vLLM reject over-length prompts with
# a graceful 400 instead (the benchmark scores that as success, not failure).
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

echo "vLLM (usual setup)"
echo "  model:                 $MODEL"
echo "  port:                  $PORT"
echo "  prefix caching:        on        (vLLM cross-request reuse)"
echo "  gpu-memory-utilization: $GPU_MEM_UTIL"
echo "  max-model-len:         $MAX_MODEL_LEN"
echo

# Why each flag:
#   --enable-prefix-caching  vLLM's reuse mechanism — the matched counterpart
#                            to kvboost's chunk-reuse/CacheBlend (reuses an
#                            exact shared *prefix* across requests).
#   --gpu-memory-utilization standard memory budget; matched to leave the same
#                            class of headroom kvboost gets.
#   --max-model-len          admitted context length (see note above re: OOM).
#   --dtype float16          matched to kvboost.
# Continuous batching is vLLM's default and stays on — it's why vLLM usually
# leads raw decode throughput; the benchmark reports that honestly.
exec vllm serve "$MODEL" \
    --dtype float16 \
    --enable-prefix-caching \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --host 0.0.0.0 \
    --port "$PORT"
