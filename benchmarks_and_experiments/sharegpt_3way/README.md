# ShareGPT 3-way benchmark — KVBoost vs vLLM vs llama.cpp

Same workload, three serving stacks. The workload is **500 multi-turn
ShareGPT conversations** replayed turn-by-turn with growing history,
served by a **Qwen2.5-7B-Instruct target + Qwen2.5-1.5B-Instruct draft**
speculative pair (γ=5 draft tokens per verification round).

## What each stack does

| Stack    | Prefix reuse                | Speculative                                  |
|----------|-----------------------------|----------------------------------------------|
| KVBoost  | CacheBlend chunk KV reuse   | `SpeculativeConfig` (target verifier + draft) |
| vLLM     | Automatic prefix caching    | `speculative_config` / `speculative_model`    |
| llama.cpp| Implicit KV retention (n_past) | `Llama(draft_model=...)`                  |

## Metrics captured (per turn → aggregated)

- **TTFT** — p50 / p90 / p99 / mean
- **ITL** — inter-token latency p50 / p90 (proxy for steady-state decode)
- **Decode tok/s** — output tokens per second during decode
- **Output tok/s (wall)** — output tokens / total wall time
- **Request throughput (rps)** — requests / total wall time
- **Cache hit ratio** — cached tokens / prompt tokens
- **Spec acceptance rate** — KVBoost only (vLLM doesn't expose it publicly,
  llama.cpp doesn't either; their speedup shows up in ITL/decode-tps)
- **Per-turn TTFT** — the "money chart" showing prefix-reuse benefit
  compounding across conversation turns

## Layout

```
sharegpt_3way/
├── _common.py          # shared loader, metrics, checkpointing
├── run_kvboost.py      # KVBoost runner
├── run_vllm.py         # vLLM runner
├── run_llamacpp.py     # llama-cpp-python runner
├── compare.py          # 3-way table + plot from the JSONs
├── run.sh              # orchestrator
└── results/
    ├── kvboost.json
    ├── vllm.json
    ├── llamacpp.json
    └── 3way_summary.png
```

## Quick start

```bash
# Default 500 samples × 3 backends.
./run.sh

# Smoke test (50 samples).
./run.sh --n-samples 50

# One backend only.
ONLY=kvboost ./run.sh

# Just the comparison (after backends ran independently).
python compare.py
```

## Prereqs

### KVBoost
Already in this repo. Needs the HF Qwen weights (auto-downloaded on first
run). Loading 7B + 1.5B in fp16 wants ~18 GB VRAM combined.

### vLLM
```bash
pip install vllm>=0.6
```
Speculative decoding kwargs changed across vLLM versions — the runner
tries the new `speculative_config={...}` form first and falls back to the
legacy `speculative_model=` / `num_speculative_tokens=` form.

### llama.cpp
```bash
# CUDA build
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-binary llama-cpp-python
```
Then point at GGUF files for both models. Q4_K_M is a good default:
```bash
# Pre-download from HF
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ~/models
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir ~/models

LLAMACPP_MODEL=~/models/qwen2.5-7b-instruct-q4_k_m.gguf \
LLAMACPP_DRAFT=~/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
./run.sh
```

If your installed `llama-cpp-python` doesn't expose the `draft_model`
kwarg, the runner falls back to non-speculative llama.cpp and logs a
warning — the prefix-cache comparison still holds.

## Output JSON schema

```jsonc
{
  "backend": "kvboost" | "vllm" | "llamacpp",
  "model": "...",
  "draft_model": "...",
  "config": { "gamma": 5, "max_new_tokens": 128, ... },
  "wall_s": 1234.5,
  "metrics": {
    "n_conversations": 500,
    "n_turns_total":   2734,
    "overall": {
      "ttft_p50_ms": ..., "ttft_p90_ms": ..., "ttft_p99_ms": ...,
      "itl_p50_ms": ..., "itl_p90_ms": ...,
      "decode_tps_mean": ...,
      "avg_cache_hit_ratio": ...,
      "request_throughput_rps": ...,
      "output_token_throughput": ...,
      "spec_acceptance_rate": ...
    },
    "by_turn": { "0": {...}, "1": {...}, ... }
  }
}
```

## Notes on fairness

- **Same prompt formatting** across backends (`Human: ...\nAssistant:`),
  so the tokenizer-side input is identical. Each backend internally
  re-tokenizes with its own tokenizer; we record both the HF-tokenized
  `history_tokens` and the engine-reported `prompt_tokens` per turn.
- **Greedy decoding** (`temperature=0`) so speculative decoding produces
  bit-identical output to non-spec decoding under the same target model.
- **No batching** — each turn is one request, sequential. This matches
  the "interactive chat" use case where prefix reuse matters most.
- **Speculative acceptance** is reported only for KVBoost (vLLM and
  llama.cpp don't surface per-request acceptance counters). Their
  speculative speedup is still visible in ITL and decode-tps.
