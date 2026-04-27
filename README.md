<p align="center">
  <img src="docs/kvboost.svg" alt="KVBoost Logo" width="200">
</p>

<h1 align="center">KVBoost</h1>

<p align="center">
  <strong>Chunk-level KV cache reuse for HuggingFace inference.</strong><br>
  Reuse KV tensors across requests that share long prefixes. Drop-in on any HF causal LM.
</p>

<p align="center">
  <a href="https://pypi.org/project/kvboost/"><img src="https://img.shields.io/pypi/v/kvboost?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/kvboost/"><img src="https://img.shields.io/pypi/pyversions/kvboost" alt="Python"></a>
  <a href="https://kvboost.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/kvboost" alt="Docs"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
  <a href="https://github.com/pythongiant/kvboost"><img src="https://img.shields.io/badge/platform-CUDA%20%7C%20MPS%20%7C%20CPU-orange" alt="Platform"></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#benchmarks">Benchmarks</a> &bull;
  <a href="#how-it-works">How it works</a> &bull;
  <a href="#when-kvboost-helps-and-when-it-doesnt">When it helps</a> &bull;
  <a href="#api-reference">API</a> &bull;
  <a href="https://kvboost.readthedocs.io/en/latest/">Docs</a>
</p>


## Quick start

```bash
pip install kvboost
```

```python
from kvboost import KVBoost

engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")

# Warm the shared prefix once
engine.warm("You are a helpful coding assistant. Always be concise...")

# Subsequent generates reuse cached chunks automatically
result = engine.generate(
    "You are a helpful coding assistant. Always be concise...\n\n"
    "User: How do I reverse a linked list?\nAssistant:",
    max_new_tokens=128,
)

print(result.output_text)
print(f"TTFT: {result.ttft_ms:.1f} ms | reuse: {result.kv_reuse_ratio:.0%}")
```

From source:

```bash
git clone https://github.com/pythongiant/kvboost.git
cd kvboost
pip install -e .
```

Requirements: Python ≥ 3.9, PyTorch ≥ 2.1, Transformers ≥ 4.38.

---

## How it works

The core idea is one sentence: **split the prompt into fixed-size chunks,
hash them, and on the next request load the K/V tensors for chunks you
have already computed instead of recomputing them.** Everything else is
making that produce correct outputs.

### 1. Chunking

[`chunk_registry.py`](src/kvboost/chunk_registry.py) splits the token
stream into fixed-size blocks (default 128). A 1000-token prompt becomes
7 full chunks plus a 104-token tail. With `--chunk-boundary-window=16`
the cut point slides up to ±16 tokens to avoid splitting mid-sentence,
which reduces seam error on natural-language prompts.

### 2. Two-level hashing

Each chunk gets two keys (see [`models.py`](src/kvboost/models.py)):

```
prefix_hash  = SHA256(previous_chunk.prefix_hash || this_chunk.tokens)
content_hash = SHA256(this_chunk.tokens)
```

The prefix hash only matches when the tokens *and every preceding chunk*
are identical — this is the case where stored K/V is directly usable.
The content hash is a fallback: the tokens match but the history doesn't,
so the stored K/V is approximately right but needs heavier correction.

### 3. Lookup and assembly

[`KVCacheManager.find_matching_chunks()`](src/kvboost/cache_manager.py)
tries prefix hash, then falls back to content hash, and flags approximate
matches. [`PromptAssembler`](src/kvboost/prompt_assembler.py) then splits
the prompt into a cached prefix (K/V loaded from memory) and a live
suffix (tokens the model still has to process).

Cache storage is an `OrderedDict` in CPU RAM with frequency-based
eviction; frequently-reused chunks (your system prompt) stay resident,
one-off chunks get evicted first. Overflow spills to a pre-allocated
binary file via [`disk_tier.py`](src/kvboost/disk_tier.py).

### 4. Seam repair

This is the part that makes stitching correct. Each cached chunk was
originally computed without seeing the chunks now preceding it in the new
prompt, so its K/V values are slightly wrong at the boundaries.

KVBoost has two strategies (`recompute_strategy=`):

- **`selective`** (default) re-runs the model on the last `R` tokens at
  each seam with the preceding cached context visible, and overwrites the
  stale K/V. Cheap but only fixes the boundary.
  ([`selective_recompute.py`](src/kvboost/selective_recompute.py))
- **`cacheblend`** does one forward pass, measures per-token cosine
  deviation vs. what the K/V would be with full context, and recomputes
  only the ~15% most-deviated tokens. Catches mid-chunk errors selective
  misses. ([`cacheblend.py`](src/kvboost/cacheblend.py))

Approximate (content-hash) matches force CacheBlend regardless of the
chosen strategy — position encodings are wrong in that case and
boundary-only repair is not enough.

Two optional continuity features stack on top of either strategy:

- `--overlap-k=16`: each chunk re-encodes the last K tokens of the
  previous chunk, so seam tokens always see K tokens of real preceding
  context at store time.
- `--sink-tokens=32`: always keep the first N tokens (the "attention
  sink") fully fresh, since many attention heads anchor on them.

### 5. Forward pass

The corrected cached K/V and the live suffix go into a single
`model.forward(past_key_values=...)` call in
[`engine.py`](src/kvboost/engine.py). Autoregressive decoding then
proceeds normally. After generation, any newly-seen chunks are written
back to the cache so the next request with overlapping text hits without
an explicit `warm()`.

### 6. Correctness guarantees

Under **greedy decoding**, the cached-and-corrected path is designed to
produce the argmax-equivalent token at every step — which matches what
the benchmark's `cosine = 1.000` columns show on the KV-side logits.
Despite this, *task* accuracy still drifts by a few points at high reuse.
Why? Because "argmax matches at step 1" does not guarantee "full
generation matches" — small K/V perturbations can tilt later tokens onto
a different branch. The accuracy-by-reuse table is the ground truth;
treat the logit-cosine metric as a necessary but not sufficient check.

Under **sampling** (temperature > 0), outputs differ run-to-run by
construction; the meaningful check is distributional (KL between logit
distributions), not token-identity.

### Optional: KV quantization

`kv_cache_bits=8` quantizes cached tensors (per-channel for K,
per-token for V — the KIVI-paper asymmetry) for ~2× RAM savings with
minimal accuracy loss. `kv_cache_bits=4` is available for 4× but you
should validate it with `verify_correctness()` on your workload before
trusting it.


## API reference

Minimum surface:

```python
KVBoost.from_pretrained(
    model_name_or_path: str,
    recompute_strategy: Literal["selective", "cacheblend", "none"] = "selective",
    chunk_size: int = 128,
    kv_cache_bits: Optional[Literal[4, 8]] = None,
    device: Optional[str] = None,          # "cuda" | "mps" | "cpu"
    ...
) -> KVBoost

engine.warm(text: str) -> WarmResult
engine.generate(prompt: str, max_new_tokens: int = ..., **kwargs) -> GenerationResult
engine.verify_correctness(prompts: list[str], ...) -> CorrectnessReport
```

`GenerationResult` exposes `output_text`, `ttft_ms`, `total_ms`,
`kv_reuse_ratio`, and the token-level traces used by the benchmarks.

Full docs: [kvboost.readthedocs.io](https://kvboost.readthedocs.io/en/latest/)


---

## Benchmarks

Results on **Qwen/Qwen2.5-3B**, **500 bug-localization samples** (LongBench, max 6 000 context tokens).
Each backend ran in an isolated process for a clean GPU state. Accuracy measured as exact-match on 4-choice multiple-choice questions.

KVBoost config: `cacheblend` strategy, 1.5 GB cache, recency window 8, boundary window 16, overlap-k 16, sink tokens 32.

### Latency — Time to First Token

![COLD vs WARM TTFT](docs/figures/cold_warm_ttft.png)

| Backend | TTFT mean | TTFT p95 | COLD mean | WARM mean | Throughput | vs Baseline |
|---|---|---|---|---|---|---|
| **KVBoost** | **142 ms** | 506 ms | 222 ms | **63 ms** | 11.7 tok/s | **4.49×** |
| vLLM (prefix cache) | 166 ms | 653 ms | 269 ms | **62 ms** | 13.2 tok/s | 3.86× |
| Baseline (HF) | 639 ms | 1 705 ms | 639 ms | 640 ms | 4.7 tok/s | 1.00× |

COLD = first query in a pair (no cached KVs). WARM = second query after the diff prefix is cached from the first.

KVBoost WARM TTFT is **3.5× faster than its own COLD** and **10.1× faster than Baseline**.
Both caching backends reach nearly identical WARM latency (~62–63 ms); KVBoost has a lower overall mean because its COLD path (222 ms) is faster than vLLM's (269 ms) due to chunk-level partial cache hits on first access.

![Speedup vs Baseline](docs/figures/speedup_summary.png)

![TTFT CDF](docs/figures/ttft_cdf.png)

The CDF shows that KVBoost's advantage is consistent across percentiles, not just at the mean — even the p95 warm latency (101 ms) is far below the baseline median (440 ms).

![TTFT by Context Length](docs/figures/ttft_by_bucket.png)

KVBoost's chunk-level partial cache hits let it outperform vLLM on COLD queries at every context-length bucket, because even a first-time request can hit cached chunks from earlier requests with overlapping text.

### Accuracy

![Accuracy vs KV Reuse](docs/figures/accuracy_vs_reuse.png)

| Backend | Overall | COLD | WARM | Avg KV reuse (warm) |
|---|---|---|---|---|
| **KVBoost** | **99.2%** | 99.2% | 99.2% | **72.9%** |
| vLLM (prefix cache) | 99.1% | 99.4% | 98.8% | — |
| Baseline (HF) | 99.1% | 99.2% | 99.0% | — |

Cold accuracy spread across backends is **0.2 pp**, confirming all three backends process identical inputs.
KVBoost WARM accuracy matches COLD exactly (99.2%) despite 72.9% average KV reuse — the CacheBlend seam repair produces no measurable quality degradation. The accuracy-by-reuse chart confirms this holds even at the 80–100% reuse bucket.

### KV Reuse Distribution (KVBoost, warm queries only)

![KV Reuse Distribution](docs/figures/kv_reuse_distribution.png)

| Reuse bucket | Share of warm queries |
|---|---|
| 80–100% | 49% |
| 60–80% | 25% |
| 40–60% | 16% |
| 20–40% | 10% |
| 0–20% | 0% |

49% of warm queries reuse more than 80% of their diff prefix from cache. Average: **72.9%**.

### GPU Memory

| Backend | Peak mean | Peak p95 | COLD mean | WARM mean |
|---|---|---|---|---|
| **KVBoost** | 6 126 MB | 6 495 MB | 6 140 MB | 6 111 MB |
| Baseline (HF) | 6 141 MB | 6 517 MB | 6 140 MB | 6 141 MB |

KVBoost warm queries use ~29 MB less peak memory than cold queries, as cached chunks skip the full prefill activation spike.
vLLM peak memory is managed internally by its engine and is not tracked via `torch.cuda.max_memory_allocated`.

---

## License

[MIT](LICENSE)
