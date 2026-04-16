<p align="center">
  <img src="docs/kvboost.svg" alt="KVBoost Logo" width="200">
</p>

<h1 align="center">KVBoost</h1>

<p align="center">
  <strong>Chunk-level KV cache reuse for HuggingFace inference.</strong><br>
  5-48x TTFT reduction on 3B+ models with repeated long context. 3 lines to integrate.
</p>

<p align="center">
  <a href="https://pypi.org/project/kvboost/"><img src="https://img.shields.io/pypi/v/kvboost?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/kvboost/"><img src="https://img.shields.io/pypi/pyversions/kvboost" alt="Python"></a>
  <a href="https://kvboost.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/kvboost" alt="Docs"></a>
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
  <a href="#how-it-works">How it works</a> &bull;
  <a href="https://kvboost.readthedocs.io/en/latest/">Docs</a>
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
---

## How it works

### What normally happens inside an LLM

When you send a prompt to a language model, the model reads every token
before it can write anything back. Internally, each layer of the model
computes two tensors for every token: a **key** and a **value** (K and V).
These K/V tensors are what the model uses to "remember" earlier parts of
the text when deciding what comes next. The full set of them is called the
KV cache.

For a 3B-parameter model reading 1,000 tokens, that first read (called
prefill) takes roughly 1-3 seconds on a MacBook. The K/V tensors are
computed, used to generate the first output token, and then kept around so
the model doesn't have to re-read the prompt for subsequent tokens. Each
new output token just adds one more K/V pair to the cache. That part is
fast.

The problem is what happens on the *next request*. You send the same system
prompt plus a different question. The model throws away everything from last
time and reads the entire prompt again from scratch. Another 1-3 seconds of
prefill, even though 90% of the prompt is identical. Multiply that by
hundreds of requests and you're spending most of your GPU time re-reading
text the model has already seen.

### What KVBoost changes

KVBoost saves those K/V tensors after each request and reuses them on the
next one. The mechanics of how it does that have a few moving parts, because
"just save and reload" has correctness problems that will silently produce
wrong outputs if you're not careful.

### Step 1: Split the prompt into chunks

`ChunkRegistry.split()` in [chunk_registry.py](src/kvboost/chunk_registry.py)
walks through the token list and cuts it into fixed-size blocks (default 128
tokens). A 1,000-token prompt becomes 7 full chunks plus a 104-token tail.

### Step 2: Hash each chunk (two hashes, not one)

Each chunk gets two identifiers, computed in [models.py](src/kvboost/models.py):

```
prefix_hash  = SHA256(previous_chunk's_hash + this_chunk's_token_bytes)
content_hash = SHA256(this_chunk's_token_bytes)
```

Why two? Suppose the sentence "The transformer architecture uses
self-attention" appears as chunk 3 in conversation A and chunk 1 in
conversation B. The tokens are identical, so the content hash is the same.
But the prefix hash is different because conversation A's hash includes
chunks 1 and 2 chained before it.

This matters because the K/V tensors for that sentence in conversation A
were computed with the model having already read conversation A's earlier
text. Those tensors encode "what these tokens mean, given everything before
them." Loading them into conversation B, where the preceding text is
completely different, would be wrong.

The prefix hash is the primary lookup key. It only matches when the tokens
*and* all preceding chunks are identical. The content hash is a fallback.
It matches on the tokens alone but flags the result as "approximate" so the
engine knows the stored data needs full correction, not just light touch-up.

### Step 3: Look up what's already cached

`KVCacheManager.find_matching_chunks()` in
[cache_manager.py](src/kvboost/cache_manager.py) tries the prefix hash
first. If that misses, it checks the content hash via a secondary index.
The result comes back wrapped in a `ChunkMatch` object that carries an
`approximate` flag (True if it was a content-hash fallback).

The cache itself is a Python `OrderedDict`. When it fills up, eviction is
frequency-based: chunks that appeared in many requests (your system prompt)
have a high count and stay put. Chunks that appeared once (a one-off
document) stay at count 1 and get evicted first.

### Step 4: Separate cached tokens from live tokens

`PromptAssembler` in [prompt_assembler.py](src/kvboost/prompt_assembler.py)
takes the cache lookup results and splits the prompt into two regions: the
prefix covered by cache hits (stored K/V data exists) and the "live" tail
(new tokens that the model hasn't seen before).

If chunks 1-7 all hit cache and only the last 104 tokens are new, those 104
tokens are the only ones the model needs to process. The cached K/V tensors
for the first 896 tokens get loaded from memory instead of recomputed.

### Step 5: Fix the stitching errors

This is the part that makes the difference between "works" and "produces
subtly wrong text."

Each cached chunk was processed independently when it was first created.
Token 129 (first token of chunk 2) never attended to token 1 (first token of
chunk 1) during that original computation. Its K/V values reflect a model
that only saw tokens 1-128, not the full prompt. When you stitch chunks 1
and 2 together and hand them to the model as if they were one continuous
sequence, those values at the boundaries are slightly off.

KVBoost has two ways to correct this, configured via `recompute_strategy`:

**`"selective"`** (the default) re-runs the model on the last 16 tokens at
each chunk boundary, this time with all preceding chunks visible. The
corrected K/V values replace the stale ones. Simple, but it only fixes
boundary tokens. A token in the middle of chunk 3 that happens to depend
on something in chunk 1 won't get corrected.

**`"cacheblend"`** takes a different approach. It runs one forward pass
through the entire stitched K/V, computes the cosine distance between each
token's stored values and what the values would be with full context, and
recomputes only the ~15% of tokens with the highest deviation. This catches
problems inside chunks, not just at edges. The implementation is in
[cacheblend.py](src/kvboost/cacheblend.py).

If any chunk was an approximate match (content hash hit, not prefix hash),
CacheBlend runs automatically regardless of your configured strategy. When
the position encodings are wrong, boundary-only repair isn't enough.

### Step 6: Run the model on the live tokens only

The corrected cached K/V and the live suffix tokens go into a single
`model.forward()` call in [engine.py](src/kvboost/engine.py). HuggingFace
models accept a `past_key_values` argument that tells them "pretend you
already processed this many tokens." The model reads the live tokens,
attends to the cached K/V as context, and produces the first z∑output token.
From there, autoregressive decoding continues token by token as normal.

After generation finishes, `_store_prompt_chunks()` saves any chunks that
weren't already in cache. So the next request with overlapping text will
hit cache without needing an explicit `warm()` call.

### Why it produces identical outputs

Under greedy decoding (temperature=0, always pick the highest-probability
token), the K/V tensors from a cached-and-corrected path are mathematically
equivalent to the K/V tensors from a full re-read. The argmax token at
every step is the same. The benchmarks verify this by running both paths
on the same prompts and comparing outputs token by token.

Under sampling (temperature > 0), the outputs aren't identical because
sampling is inherently random. But the probability distributions are the
same, which you can verify by measuring KL divergence between the two
paths' logit distributions.

### Where the data lives

Cached K/V tensors sit in a Python dict in CPU RAM by default. When the
model needs them, they're moved to the GPU.

If you set `kv_cache_bits=8`, the tensors get compressed to int8 before
storage. Keys are quantized per-channel, values per-token (the asymmetry
from the KIVI paper, ICML 2024). This halves RAM usage with near-zero
accuracy loss. `kv_cache_bits=4` is available for 4x compression but
should be validated with `verify_correctness()` first.

When the in-memory cache fills up, evicted chunks are written to a single
pre-allocated binary file on disk. A JSON index maps chunk hashes to byte
offsets in that file. When a disk-tier chunk gets a cache hit, it's read
back and promoted to RAM.

> Full API docs: [kvboost.readthedocs.io](https://kvboost.readthedocs.io/en/latest/)


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
## License

[MIT](LICENSE)
