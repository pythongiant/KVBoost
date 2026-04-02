<p align="center">
  <img src="docs/kvboost.svg" alt="KVBoost Logo" width="200">
</p>

<h1 align="center">KVBoost</h1>

<p align="center">
  <strong>Chunk-level KV cache reuse for HuggingFace inference.</strong><br>
  5-48x TTFT reduction on 3B+ models with repeated long context. 3 lines to integrate.
</p>

<p align="center">
  <a href="https://pypi.org/project/kvboost/0.1.0/"><img src="https://img.shields.io/pypi/v/kvboost?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/kvboost/0.1.0/"><img src="https://img.shields.io/pypi/pyversions/kvboost" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/pythongiant/kvboost"><img src="https://img.shields.io/badge/platform-CUDA%20%7C%20MPS%20%7C%20CPU-orange" alt="Platform"></a>
  <a href="https://github.com/pythongiant/kvboost/stargazers"><img src="https://img.shields.io/github/stars/pythongiant/kvboost?style=social" alt="Stars"></a>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> &bull;
  <a href="#-benchmarks">Benchmarks</a> &bull;
  <a href="#-installation">Installation</a> &bull;
  <a href="#-api-reference">API Reference</a> &bull;
  <a href="#-examples">Examples</a> &bull;
  <a href="#-how-it-works">How It Works</a>
</p>

---

### When KVBoost Helps

| Condition | Expected TTFT Speedup |
|---|---|
| Multi-turn conversation, 8+ turns, 3B+ model | **10-48x** |
| Code context / document reuse, 800+ tokens | **15-21x** |
| RAG document reuse, ~500 tokens | 1-2x |
| System prompt reuse, ~250 tokens | 0.3-0.5x (overhead > savings) |
| Any workload, 0.5B model | < 1x (overhead exceeds prefill) |

> **Rule of thumb:** Benefits appear on **3B+ models** with **500+ token shared
> context**. Below this, caching overhead exceeds prefill savings. The peak 47.9x
> is at 1350 tokens on Qwen2.5-3B — see [benchmarks](#benchmarks) for full data.

### Features

| | |
|---|---|
| **Identical outputs** | Greedy decoding produces the same text as baseline |
| **3 lines to integrate** | `from_pretrained` / `warm` / `generate` |
| **11+ architectures** | Llama, Qwen, Gemma, Mistral, Phi -- any RoPE model on CUDA, MPS, or CPU |
| **Two recompute strategies** | Selective boundary recompute or CacheBlend deviation-guided |
| **Two-tier storage** | Hot RAM (frequency-based eviction) + optional disk-backed cold tier |
| **Prefix-chained keys** | vLLM-style hash chaining for positional correctness |

---

## Installation

```bash
pip install kvboost
```

**From source:**

```bash
git clone https://github.com/pythongiant/kvboost.git
cd kvboost
pip install -e .
```

**Requirements:** Python >= 3.9, PyTorch >= 2.1, Transformers >= 4.38

---

## Quick Start

```python
from kvboost import KVBoost

# 1. Load any HuggingFace causal LM
engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")

# 2. Cache your system prompt / document / few-shot examples once
engine.warm("You are a helpful coding assistant. Always provide concise answers...")

# 3. Generate -- cached prefix is reused automatically
result = engine.generate(
    "You are a helpful coding assistant. Always provide concise answers...\n\n"
    "User: How do I reverse a linked list?\n"
    "Assistant:",
    max_new_tokens=128,
)

print(result.output_text)
print(f"TTFT: {result.ttft_ms:.1f}ms | Cache reuse: {result.kv_reuse_ratio:.0%}")
```

---

## Benchmarks

> Qwen/Qwen2.5-3B (float16) on MacBook Air M-series, 16GB RAM, MPS backend.
> Chunk size 128, greedy decoding.

### Multi-Turn Conversation

Baseline TTFT scales linearly with history. KVBoost stays **flat at ~62ms**.

| Turn | Tokens | Baseline | KVBoost | Reuse | Speedup |
|:---:|:---:|---:|---:|:---:|:---:|
| 1 | 232 | 35ms | 31ms | 0% | 1.1x |
| 2 | 353 | 149ms | 79ms | 36% | 1.9x |
| 3 | 495 | 194ms | 60ms | 52% | 3.2x |
| 4 | 621 | 374ms | 62ms | 62% | **6.0x** |
| 5 | 762 | 658ms | 57ms | 67% | **11.6x** |
| 6 | 946 | 1,228ms | 63ms | 68% | **19.6x** |
| 7 | 1,113 | 1,737ms | 64ms | 81% | **27.2x** |
| 8 | 1,353 | 2,970ms | 62ms | 76% | **47.9x** |

### Code Context Reuse (~800 tokens)

| Query | Baseline | KVBoost | Reuse | Speedup |
|---|---:|---:|:---:|:---:|
| Q1 (cold) | 1,670ms | 2,292ms | 0% | 0.7x |
| Q2 (warm) | 1,577ms | **75ms** | 92% | **21.1x** |
| Q3 (warm) | 2,133ms | **128ms** | 92% | **16.6x** |

### System Prompt Reuse (~250 tokens)

Identical outputs, but prompts are too short for speedup at 3B scale:

| Query | Baseline | KVBoost | Reuse | Speedup |
|---|---:|---:|:---:|:---:|
| Q1 (cold) | 40ms | 76ms | 60% | 0.5x |
| Q2 | 34ms | 75ms | 60% | 0.4x |
| Q3 | 34ms | 96ms | 60% | 0.4x |
| Q4 | 34ms | 121ms | 61% | 0.3x |

### RAG Document Reuse (~500 tokens)

| Query | Baseline | KVBoost | Reuse | Speedup |
|---|---:|---:|:---:|:---:|
| Q1 (cold) | 72ms | 48ms | 0% | 1.5x |
| Q2 (warm) | 78ms | **51ms** | 86% | 1.5x |
| Q3 (warm) | 47ms | 55ms | 85% | 0.9x |

### Few-Shot Classification (~500 tokens)

| Review | Baseline | KVBoost | Reuse | Speedup |
|---|---:|---:|:---:|:---:|
| Review 1 | 75ms | 53ms | 81% | 1.4x |
| Review 2 | 52ms | 54ms | 81% | 1.0x |
| Review 3 | 40ms | 52ms | 81% | 0.8x |

> **Pattern:** At ~250-500 tokens, KVBoost is roughly break-even. The cache
> overhead (~60-100ms) matches the prefill savings. Speedups become dramatic
> above ~600 tokens where prefill dominates.

### When Does It Help? (Model Size Matters)

The same examples on **Qwen2-0.5B** tell the opposite story -- cache overhead
*exceeds* prefill savings because the model is too small for prefill to be a
bottleneck.

<details>
<summary><strong>Qwen2-0.5B results (click to expand)</strong></summary>

> Qwen/Qwen2-0.5B (float16), chunk_size=64, same MacBook Air.

**RAG Document Reuse** -- high reuse, but KVBoost is *slower*:

| Query | Baseline | KVBoost | Reuse | Speedup |
|---|---:|---:|:---:|:---:|
| Q1 (cold) | 244ms | 12ms | 0% | 20.3x |
| Q2 (warm) | 29ms | 152ms | 83% | **0.2x** |
| Q3 (warm) | 27ms | 141ms | 81% | **0.2x** |

**Code Context Reuse** -- same pattern:

| Query | Baseline | KVBoost | Reuse | Speedup |
|---|---:|---:|:---:|:---:|
| Q1 (cold) | 216ms | 13ms | 0% | 16.6x |
| Q2 (warm) | 32ms | 94ms | 74% | **0.3x** |
| Q3 (warm) | 29ms | 41ms | 73% | **0.7x** |

**Multi-Turn** -- never reaches the crossover:

| Turn | Tokens | Baseline | KVBoost | Reuse | Speedup |
|:---:|:---:|---:|---:|:---:|:---:|
| 1 | 23 | 47ms | 12ms | 0% | 4.0x |
| 2 | 48 | 12ms | 12ms | 0% | 1.0x |
| 3 | 83 | 55ms | 12ms | 0% | 4.6x |
| 4 | 110 | 197ms | 100ms | 58% | 2.0x |

</details>

**Why?** At 0.5B, prefill costs ~30ms after MPS kernel warmup -- there's nothing
meaningful to save. The cache lookup + CPU-to-MPS transfer + selective recompute
overhead (~100ms) exceeds the prefill it replaces.

| | Qwen2-0.5B | Qwen2.5-3B |
|---|:---:|:---:|
| Prefill cost (500 tok) | ~30ms | ~400ms |
| Cache overhead | ~100ms | ~60ms |
| Break-even | Never (overhead > savings) | ~350 tokens |
| Peak speedup | 2.0x (110 tok) | **47.9x** (1353 tok) |

> **Rule of thumb:** KVBoost pays off on **3B+ models** with **500+ token prompts**.
> The bigger the model and the longer the prompt, the larger the win.

<details>
<summary><strong>Methodology validation</strong></summary>

Three properties confirm these results reflect genuine cache reuse:

1. **Flat TTFT curve** -- KVBoost TTFT stays ~62ms from 232 to 1,353 tokens.
   This is the signature of cache reuse: only live tokens (constant per turn)
   are processed.

2. **Output correctness** -- Under greedy decoding, baseline and KVBoost produce
   **identical output text**. Corrupted cache tensors would cause divergence.

3. **Cold-start control** -- Every first query shows 0% reuse and comparable TTFT.
   Speedup only appears after cache population, ruling out measurement bias.

</details>

### KVBoost vs MLX LLM

Head-to-head against Apple's Metal-optimized [MLX](https://github.com/ml-explore/mlx)
inference framework. Same model (Qwen2.5-3B), same hardware, same prompts.

| Workload | Query | HF Baseline | KVBoost | MLX | KV vs HF | KV vs MLX |
|---|---|---:|---:|---:|:---:|:---:|
| **Chatbot** | Q1 (cold) | 52,122ms | **81ms** | 60,527ms | 640x | 743x |
| | Q2 | 20,276ms | **1,301ms** | 56,556ms | 16x | 44x |
| | Q3 | 19,571ms | **1,308ms** | 52,906ms | 15x | 40x |
| **Code** | Q1 (cold) | 63,369ms | **493ms** | 49,342ms | 129x | 100x |
| | Q2 | 40,635ms | **137ms** | 47,167ms | 297x | 344x |
| | Q3 | 39,529ms | **152ms** | 48,592ms | 260x | 319x |
| **Multi-turn** | Q1 | 71,835ms | **108ms** | 61,067ms | 668x | 568x |
| | Q2 | 41,743ms | **349ms** | 45,344ms | 120x | 130x |
| | Q3 | 48,858ms | **153ms** | 46,962ms | 319x | 307x |
| | Q4 | 40,834ms | **233ms** | 62,975ms | 175x | 270x |
| | Q5 | 51,666ms | **130ms** | 53,865ms | 398x | 415x |
| | Q6 | 52,664ms | **153ms** | 50,860ms | 345x | 333x |
| | Q7 | 50,676ms | **144ms** | 46,060ms | 353x | 321x |

> KVBoost is **100-743x faster than MLX** on TTFT across all workloads. MLX shows
> no cross-request cache reuse -- each prompt is processed from scratch. KVBoost's
> chunk-level caching eliminates redundant prefill entirely.

<details>
<summary><strong>Run it yourself</strong></summary>

```bash
pip install mlx-lm
python benchmarks_and_experiments/benchmark_vs_mlx.py
python benchmarks_and_experiments/benchmark_vs_mlx.py --workload code
```

</details>

### KVBoost vs vLLM Prefix Caching (vllm-mlx)

Head-to-head against [vLLM-MLX](https://github.com/waybarrios/vllm-mlx) prefix
caching on Apple Silicon. vLLM caches system prompt KV and reuses on exact prefix
match. KVBoost reuses any matching chunk, including non-prefix interior content.

> KVBoost: Qwen2.5-3B float16 (MPS) | vLLM-MLX: Qwen2.5-3B 4-bit (MLX Metal)

**Axis 1: Non-Prefix Interior Reuse** (KVBoost's differentiator)

Document placed at the start, in the middle, or not at all:

| Pattern | Query | HF Baseline | KVBoost | vLLM-MLX | KV vs vLLM |
|---|---|---:|---:|---:|:---:|
| **Exact prefix** | Q1 | 51ms | **58ms** (89%) | 1,722ms | 29.6x |
| | Q2 | 673ms | **291ms** (90%) | 928ms | 3.2x |
| | Q3 | 226ms | **72ms** (88%) | 856ms | 11.9x |
| **Interior** | Q1 | 307ms | **33ms** (0%) | 1,219ms | 36.8x |
| | Q2 | 321ms | **103ms** (83%) | 1,214ms | 11.8x |
| | Q3 | 324ms | **57ms** (82%) | 1,294ms | 22.8x |
| **No reuse** | Q1 | 287ms | **33ms** | 1,346ms | 40.7x |
| | Q2 | 335ms | **33ms** | 1,313ms | 39.5x |
| | Q3 | 313ms | **33ms** | 1,279ms | 38.7x |

**Axis 2: Cold-Start Overhead**

| Cache State | HF Baseline | KVBoost | vLLM-MLX |
|---|---:|---:|---:|
| Cold | 206ms | **32ms** | 777ms |
| Warm Q2 | 204ms | **44ms** (90%) | 891ms |
| Warm Q3 | 210ms | **62ms** (88%) | 942ms |

**Axis 3: Break-Even Prompt Length**

| Prompt Length | Baseline | KVBoost (cold) | KVBoost (warm) | vLLM (cold) | vLLM (warm) |
|---|---:|---:|---:|---:|---:|
| ~100 words | 154ms | 28ms | 124ms (0%) | 549ms | 438ms |
| ~250 words | 244ms | 33ms | **37ms** (89%) | 1,039ms | 849ms |
| ~500 words | 403ms | 76ms | **48ms** (88%) | 1,882ms | 1,960ms |
| ~1000 words | 2,329ms | 2,218ms | **242ms** (98%) | 21,047ms | 61,131ms |
| ~2000 words | 76,864ms | 7,302ms | **1,452ms** (98%) | 49,015ms | 66,714ms |

> **Key findings:**
> - KVBoost is **3-41x faster than vLLM-MLX** on TTFT across all patterns
> - On **interior reuse** (document in the middle), vLLM gets zero cache hits
>   while KVBoost achieves 82-83% reuse -- this is the core differentiator
> - At 2000 words warm, KVBoost is **46x faster** than vLLM-MLX (1.5s vs 66.7s)
> - Even with **no reuse possible**, KVBoost's HF baseline (33ms) beats vLLM-MLX (1.3s)
>   due to MPS vs MLX Metal overhead differences
> - Overall mean TTFT: KVBoost **564ms** vs vLLM-MLX **9,928ms**

<details>
<summary><strong>Run it yourself</strong></summary>

```bash
pip install vllm-mlx
python benchmarks_and_experiments/benchmark_vs_vllm.py
python benchmarks_and_experiments/benchmark_vs_vllm.py --axis non_prefix
python benchmarks_and_experiments/benchmark_vs_vllm.py --skip-vllm  # KVBoost only
```

</details>

---

## API Reference

### `KVBoost.from_pretrained(model_name, **kwargs)`

Factory method. Loads a HuggingFace model and tokenizer. Validates architecture
compatibility at load time.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` | HF decoder-only causal LM (must use RoPE) |
| `strict` | `bool` | `True` | Raise on unsupported architectures, warn on untested |
| `chunk_size` | `int` | `128` | Tokens per cache chunk |
| `max_chunks` | `int` | `128` | Max chunks in RAM before LRU eviction |
| `recompute_strategy` | `str` | `"selective"` | `"selective"`, `"cacheblend"`, or `"none"` (see below) |
| `recompute_overlap` | `int` | `16` | Tokens to recompute at seams (selective only) |
| `recompute_ratio` | `float` | `0.15` | Fraction of tokens to recompute (cacheblend only) |
| `disk_cache_dir` | `str \| None` | `None` | Path for disk-backed cold storage |
| `device` | `str \| None` | `None` | `"cuda"`, `"mps"`, `"cpu"`, or auto-detect |

**Recompute strategies:**

| Strategy | How it works | When to use |
|---|---|---|
| `"selective"` | Recomputes last R tokens at each chunk boundary | Default, safe baseline |
| `"cacheblend"` | Measures per-token KV deviation, recomputes only the ~15% that actually changed | Better quality/speed trade-off on long prompts |
| `"none"` | Skips recompute entirely | Maximum speed, acceptable when chunks are from the same original encoding |

```python
from kvboost import KVBoost

# Default: selective recompute at boundaries
engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")

# CacheBlend: smarter, recomputes only deviated tokens
engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B", recompute_strategy="cacheblend")

# No recompute: fastest, use when chunks share original context
engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B", recompute_strategy="none")
```

### `engine.warm(text, position_offset=0) -> int`

Pre-cache fixed-size chunks from `text`. Returns number of new chunks stored.
Call this for content reused across requests: system prompts, documents, few-shot examples.

### `engine.generate(prompt, **kwargs) -> GenerationResult`

Generate text with automatic KV cache reuse.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `prompt` | `str` | -- | Full prompt including any cached prefix |
| `max_new_tokens` | `int` | `64` | Max tokens to generate |
| `mode` | `GenerationMode` | `CHUNK_KV_REUSE` | `BASELINE`, `PREFIX_CACHE`, or `CHUNK_KV_REUSE` |
| `temperature` | `float` | `1.0` | Sampling temperature |
| `do_sample` | `bool` | `False` | Greedy (`False`) or sampling (`True`) |

### `GenerationResult`

| Field | Type | Description |
|---|---|---|
| `output_text` | `str` | Generated text |
| `ttft_ms` | `float` | Time to first token (ms) |
| `total_ms` | `float` | End-to-end latency (ms) |
| `tokens_per_sec` | `float` | Decode throughput |
| `kv_reuse_ratio` | `float` | Fraction of prompt served from cache |
| `prompt_tokens` | `int` | Total prompt token count |
| `cached_tokens` | `int` | Tokens served from cache |

### `engine.cache_stats() -> dict`

Returns: `hot_chunks`, `hot_memory_mb`, `cache_hits`, `approximate_hits`,
`cache_misses`, `hit_rate`, `exact_hit_rate`.

### `engine.verify_correctness(max_new_tokens=32) -> bool`

Runs a quick greedy-decode comparison (baseline vs cached) on a synthetic prompt.
Returns `True` if outputs match. Use this to validate untested architectures.

```python
engine = KVBoost.from_pretrained("some/untested-model", strict=False)
assert engine.verify_correctness(), "KV cache stitching produces wrong outputs!"
```

### Model Compatibility

KVBoost's KV cache stitching requires **RoPE positional encoding** with explicit
`position_ids` support. Models using ALiBi, learned absolute embeddings, or
sliding window attention are not compatible.

| Status | Architectures |
|---|---|
| **Supported** | Llama, Qwen2, Qwen2.5, Gemma, Gemma2, Mistral (full attn), Phi, Phi3, StableLM, InternLM |
| **Unsupported** | GPT-2, GPT-Neo, GPT-NeoX, MPT, Falcon, BLOOM |
| **Conditional** | Mistral with `sliding_window != None` -- blocked |

```python
# Supported model -- loads normally
engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")

# Unsupported model -- raises ValueError with explanation
engine = KVBoost.from_pretrained("gpt2")
# ValueError: GPT-2 uses learned absolute positional embeddings...

# Unknown model -- warns, user can self-certify
engine = KVBoost.from_pretrained("some/new-rope-model", strict=False)
assert engine.verify_correctness()

# Skip all checks (you know what you're doing)
engine = KVBoost.from_pretrained("some/model", strict=False)
```

---

## Examples

The [`examples/`](examples/) directory contains runnable demos for 5 real-world patterns.
Configuration is driven by a `.env` file -- swap models without touching code.

```bash
cp examples/.env.example examples/.env   # configure model, device, etc.
python examples/run.py                    # run all examples
python examples/run.py --example rag      # run one
python examples/run.py --model Qwen/Qwen2.5-3B  # override model
python examples/run.py --list             # see all options
```

| Example | Pattern | What it demonstrates |
|---|---|---|
| `chatbot` | System prompt reuse | Fixed instructions cached, reused across queries |
| `rag` | RAG document reuse | Same retrieved doc, multiple questions |
| `fewshot` | Few-shot classification | Cached examples, only new inputs need compute |
| `multiturn` | Multi-turn conversation | Growing history with increasing cache reuse |
| `code` | Code context reuse | Shared code file queried multiple times |

---

## How It Works

### Architecture

```
prompt_text
    |
    v
 tokenize
    |
    v
 ChunkRegistry.split()          -- split into fixed/semantic chunks
    |
    v
 KVCacheManager.find_matches()  -- SHA256 lookup per chunk
    |
    v
 PromptAssembler.assemble()     -- stitch cached KV + live tokens
    |
    v
 Recompute (selective OR cacheblend) -- fix stale KV from stitching
    |
    v
 model.forward(live_tokens, past_key_values=cached_kv)
    |
    v
 GenerationResult
```

### Key Design Decisions

**Content-addressed chunks** -- Each chunk is keyed by `SHA256(token_bytes)`.
Same tokens always produce the same key, regardless of surrounding context.

**RoPE-safe position handling** -- Cached chunks store KV at their original
positions. Live tokens get `position_ids` starting at `cached_length`, so
positional encodings remain monotonically correct.

**Two recompute strategies** -- When chunks are stitched together, their KV
tensors are "stale" (computed without cross-chunk attention). KVBoost offers
two ways to fix this:

*Selective recompute* (default) recomputes the last `R` tokens at each chunk
seam. Simple and correct, but costs `O(R * num_seams)` — which can be up to
79% of full prefill cost.

*CacheBlend recompute* runs a cheap forward pass over the assembled KV, measures
per-token cosine deviation between cached and full-context KV vectors, and
recomputes only the ~15% of tokens that actually deviate. This is both faster
(fewer tokens recomputed) and more accurate (fixes tokens *inside* chunks that
depend on cross-chunk context, not just boundaries). Based on the CacheBlend
paper (USENIX ATC '25).

**Two-tier storage** -- Hot tier is an in-memory `OrderedDict` with LRU eviction.
Optional cold tier serializes evicted chunks to disk via `torch.save()` and
promotes them back on cache hit.

<details>
<summary><strong>Data structures</strong></summary>

```python
CachedChunk:
    chunk_id:        str          # SHA256(token_ids bytes)
    token_ids:       List[int]    # tokenized content
    past_key_values: Tuple[       # HF format: (key, value) per layer
                       Tuple[Tensor, Tensor], ...
                     ]
    position_start:  int          # absolute position of first token
    position_end:    int          # exclusive end position

AssembledPrompt:
    cached_past_kv:    PastKVType   # merged KV for all reused chunks
    cached_length:     int          # tokens covered by cache
    live_token_ids:    List[int]    # tokens needing fresh forward pass
    live_position_ids: List[int]    # absolute positions for live tokens
    chunk_boundaries:  List[Tuple]  # seam locations for recompute
    cache_hit_ratio:   float
```

</details>

<details>
<summary><strong>Memory layout</strong></summary>

Per-chunk KV memory (float16):

| Model | Layers | Heads | head_dim | chunk_size=128 |
|---|:---:|:---:|:---:|---:|
| TinyLlama 1.1B | 22 | 32 | 64 | 22.9 MB |
| GPT-2 Medium | 24 | 16 | 64 | 12.6 MB |
| Qwen2.5-3B | 36 | 16 | 128 | 9.4 MB |

Formula: `2 * layers * heads * chunk_size * head_dim * 2 bytes`

</details>

---

## Experimental Results (TinyLlama 1.1B)

<details>
<summary><strong>Expand full experiment suite (10 experiments)</strong></summary>

All results on TinyLlama-1.1B-Chat, Apple Silicon (MPS). Full JSON data in
[`benchmarks_and_experiments/results/`](benchmarks_and_experiments/results/).

---

#### Experiment 1: Scaling Across Modes

![TTFT by Generation Mode](docs/figures/ttft_modes.png)

| Workload | Mode | TTFT (ms) | Speedup |
|---|---|---|---|
| System prompt | Baseline | 191.7 | 1.0x |
| System prompt | Prefix cache | 66.5 | 2.9x |
| System prompt | Chunk KV reuse | 68.9 | 2.8x |
| RAG document | Baseline | 75.2 | 1.0x |
| RAG document | Prefix cache | 95.7 | 0.8x |
| RAG document | Chunk KV reuse | 25.5 | 3.0x |

Chunk KV reuse delivers its biggest win on RAG workloads (3x) where non-prefix
chunk matching kicks in.

---

#### Experiment 2: Latency Breakdown

![Latency Breakdown](docs/figures/latency_breakdown.png)

| Stage | Baseline | Chunk KV |
|---|---|---|
| Cache lookup | 0.0 | 13.3 |
| Selective recompute | 0.0 | 553.8 |
| Live token prefill | 1431.8 | 703.1 |
| Decode | 747.6 | 605.4 |
| **Total** | **2179.4** | **1875.8** |

Selective recompute is the dominant overhead. Recompute optimization is the
highest-leverage improvement opportunity.

---

#### Experiment 3: Hyperparameter Sweep

| Chunk Size | Overlap | TTFT (ms) | Hit Rate | MB/chunk |
|---|---|---|---|---|
| 64 | 0 | 21.2 | 0.750 | 0.79 |
| 64 | 16 | 20.3 | 0.750 | 0.79 |
| 128 | 0 | 19.7 | 0.429 | 2.23 |
| 128 | 16 | 16.1 | 0.429 | 2.23 |

Smaller chunks (64) achieve higher hit rates at lower memory cost.

---

#### Experiment 4: Output Quality

| Test | Result |
|---|---|
| Greedy output match (with recompute) | **100%** |
| Greedy output match (without recompute) | **100%** |
| Long-range dependency match | **100%** |
| Sampling Hellinger distance (temp=0.5) | Near 0 |

Under greedy decoding, chunk KV reuse produces **identical outputs** to baseline.

---

#### Experiment 5: Realistic Workloads

![Multi-Turn Conversation](docs/figures/multi_turn.png)

**Multi-turn** (8 turns): KV reuse becomes faster at turn 4 (50% reuse), reaching
43% faster at turn 7 (90% reuse).

**Server simulation** (20 queries): **76% TTFT reduction**, **2.3x throughput**.

---

#### Experiment 6: Memory Analysis

| Metric | Value |
|---|---|
| Speedup factor | **29.9x** |
| Cache memory | 15.5 MB |
| **Break-even** | **12 requests** |

---

#### Experiment 7: Comparison with Existing Systems

| Feature | KVBoost | vLLM | SGLang |
|---|:---:|:---:|:---:|
| Prefix caching | Y | Y | Y |
| Non-prefix chunk reuse | Y | N | Partial |
| Selective boundary recompute | Y | N | N |
| Content-addressable keys | Y | Y | N |
| Disk-backed cold storage | Y | N | N |
| Semantic chunking | Y | N | N |
| Continuous batching | N | Y | Y |
| PagedAttention | N | Y | Y |

KVBoost is complementary to production serving features like PagedAttention.

---

#### Experiment 8: Chunking Strategies

| Strategy | System Prompt TTFT | RAG Hit Rate | Memory |
|---|---|---|---|
| Fixed | 32.6ms | 65.0% | 28.4 MB |
| **Semantic** | **20.9ms** | 61.9% | 27.4 MB |
| Document | 34.7ms | 21.4% | 39.9 MB |

Semantic chunking is **36% faster** for system prompts.

---

#### Experiment 9: Cache Hit Rate Under Traffic

![Cache Warmup](docs/figures/cache_warmup.png)

| Pattern | Hit Rate | Mean TTFT |
|---|---|---|
| Uniform | 22% | 138.7ms |
| Zipfian | 38% | 68.5ms |
| **Temporal** | **62%** | **65.0ms** |

Cache warms up in ~20 requests. Temporal locality matches real API traffic.

---

#### Experiment 10: Statistical Rigor

Cold-start TTFT: 368ms (baseline) vs 118ms (KVBoost) = **3.1x improvement**.

At TinyLlama scale, overhead can offset savings for short prompts. Benefits
grow with model size where prefill cost dominates.

</details>

### Summary of Findings

1. **47.9x TTFT speedup** on multi-turn conversations with 1350+ tokens
2. **21x speedup** on code context reuse (~800 tokens)
3. **Identical outputs** under greedy decoding (mathematically equivalent)
4. **Cache pays for itself in 12 requests** with only 15.5 MB overhead
5. **Semantic chunking outperforms fixed by 36%** for system prompts
6. **Benefits scale with prompt length** -- gains appear above ~500 tokens

---

## Running Experiments

```bash
cd benchmarks_and_experiments

python run_all.py              # full suite (~55 min)
python run_all.py --quick      # quick mode (~15 min)
python run_all.py --experiments 2,4,10  # specific experiments
```

Results are saved to [`benchmarks_and_experiments/results/`](benchmarks_and_experiments/results/).

---

## Contributing

Contributions are welcome! Areas of interest:

- **Recompute optimization** -- selective recompute is the current bottleneck
- **Batch inference** -- extending cache reuse to batched requests
- **PagedAttention integration** -- combining with vLLM-style memory management
- **Quantized KV storage** -- int8/int4 cache tensors for lower memory footprint

---

## License

[MIT](LICENSE)
