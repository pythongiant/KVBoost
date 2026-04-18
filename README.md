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

---

## TL;DR

- **What it is**: a library that caches the per-chunk KV tensors of your HF
  model and reuses them across requests that share prefixes.
- **What it gives you**: lower TTFT on repeated long context. Exact
  magnitude depends on reuse ratio, context length, and model size.
- **What it costs you**: a small accuracy tax on long-context reasoning
  tasks, and extra RAM for the cache. Quantified below.
- **What it is not**: a free speedup. Short prompts, small models, and
  prompts with zero overlap will get slower, not faster.

---

## Headline results

All numbers come from [`benchmarks_and_experiments/`](benchmarks_and_experiments/)
and are reproducible from the scripts in that directory. We report what we
actually measured, including the cases where KVBoost lost.

### Bug-localization arena (Qwen2.5-3B, 720 samples, selective recompute + all continuity features)

Dataset: `JetBrains-Research/lca-bug-localization`, 4-way multiple choice
over code diffs (100–3K tokens).

| Metric                   | Baseline HF | KVBoost | Delta |
|--------------------------|-------------|---------|-------|
| Accuracy                 | 99.58%      | 97.36%  | **−2.22 pp** |
| TTFT (mean)              | 8331 ms     | 4209 ms | **1.98× faster** |
| Total latency (mean)     | 13476 ms    | 14445 ms| **0.93× (7% slower)** |
| Logit cosine (min / mean)| —           | 1.000 / 1.000 | identical under greedy |

McNemar: 0 samples where baseline was wrong and KVBoost was right;
16 samples where baseline was right and KVBoost was wrong. The accuracy
cost is real and one-sided.

**What this means.** Prefill gets meaningfully faster (2× TTFT). Total
wall-clock gets slightly worse here because this benchmark generates short
MC answers, so the decode phase dominates and KVBoost's extra stitching
work shows up. For workloads that generate longer answers, TTFT savings
translate directly into total speedup.

### Accuracy vs. KV reuse (same run)

Accuracy is close to baseline at low reuse and degrades as reuse climbs.
This is the honest picture: the more of the prompt you serve from cached
chunks, the more likely a downstream token depends on context the seam
repair didn't fully reconstruct.

| Reuse bucket | N   | Accuracy | TTFT speedup |
|--------------|-----|----------|--------------|
| ~0.0         | 360 | 99.44%   | 1.24×        |
| ~0.2         | 16  | 100.00%  | 0.90×        |
| ~0.3         | 15  | 100.00%  | 1.62×        |
| ~0.4         | 26  | 100.00%  | 3.25×        |
| ~0.5         | 34  | 100.00%  | 2.61×        |
| ~0.6         | 46  | 100.00%  | 3.08×        |
| ~0.7         | 66  | 95.45%   | 3.56×        |
| ~0.8         | 81  | 91.36%   | 4.04×        |
| ~0.9         | 73  | 90.41%   | 7.57×        |
| ~1.0         | 3   | 100.00%  | 29.23×       |

### Context-length scaling (earlier 820-sample checkpoint, same workload)

Speedup and reuse both grow with context length; the short-context bucket
is essentially free of accuracy loss, and the hit concentrates in the
1K–4K bucket where reuse is highest.

| Context   | N   | Base acc | KV acc | Δ acc  | Base TTFT | KV TTFT  | Speedup | Reuse |
|-----------|-----|----------|--------|--------|-----------|----------|---------|-------|
| 0–512     | 200 | 100.0%   | 100.0% | +0.0%  | 387 ms    | 243 ms   | 2.3×    | 21%   |
| 512–1K    | 200 | 99.5%    | 98.0%  | −1.5%  | 4521 ms   | 2309 ms  | 3.5×    | 33%   |
| 1K–2K     | 200 | 99.0%    | 96.0%  | −3.0%  | 9134 ms   | 5255 ms  | 4.1×    | 40%   |
| 2K–4K     | 200 | 98.5%    | 92.0%  | −6.5%  | 26 075 ms | 10 909 ms| 11.2×   | 44%   |
| 4K+       | 20  | 100.0%   | 90.0%  | −10.0% | 73 791 ms | 39 428 ms| 29.0×   | 42%   |
| **All**   | 820 | 99.3%    | 96.3%  | −2.9%  | 11 584 ms | 5527 ms  | 5.8×    | 35%   |

Takeaway: you buy 2–29× TTFT with 0–10 points of task accuracy, and the
trade sharpens with context length. If you are latency-bound on long
prompts, this is usually a good deal; if you are accuracy-bound on long
reasoning, it is not.

---

## When KVBoost helps (and when it doesn't)

| Workload                                                  | Expect             |
|-----------------------------------------------------------|--------------------|
| Multi-turn conversation, 3B+ model, repeated system prompt| **~2× TTFT**       |
| Code/doc Q&A with repeated code context, 1K–4K tokens     | **3–11× TTFT**     |
| Very long shared context (4K+)                            | **up to ~29× TTFT**|
| RAG with mostly-unique documents (~500 tok, low reuse)    | roughly break-even |
| Short prompts (<500 tok) or <1B models                    | **slower** than baseline |
| One-shot prompts with no shared prefix                    | slower (pure overhead) |

Rules of thumb:

1. **You need reuse.** If successive prompts share nothing, KVBoost adds
   hashing and bookkeeping for no gain. `% zero reuse: 50%` in the bug
   run above is why the mean total latency was a wash.
2. **Prefill has to be expensive enough to amortize the overhead.** On a
   0.5B model or a 200-token prompt, prefill is already cheap. KVBoost
   cannot beat "just run it."
3. **Accuracy-critical long-context work needs testing.** Run
   `verify_correctness()` on your own prompts before shipping; don't
   assume the bug-localization numbers transfer.

---

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

---

## Benchmarks

All runners live under [`benchmarks_and_experiments/`](benchmarks_and_experiments/).
The main ones:

| Script                                    | What it measures                              |
|-------------------------------------------|-----------------------------------------------|
| `long_bench_arena.py`                     | Paired accuracy + latency on code bug-loc MC  |
| `01_scale_models.py`                      | TTFT across 1.1B / 3B / 7B on shared workloads|
| `02_latency_breakdown.py`                 | Where time goes (hash, lookup, recompute, fwd)|
| `03_hyperparameter_sweep.py`              | Chunk size, recompute window, strategy sweep  |
| `04_output_quality.py`                    | Logit cosine, KL, output-identity stats       |
| `05_realistic_workloads.py`               | Multi-turn, RAG, system-prompt scenarios      |
| `06_memory_analysis.py`                   | RAM / disk footprint vs. cache size           |
| `10_statistical_rigor.py`                 | McNemar, bootstrap CIs on paired runs         |
| `run_ablation.sh`                         | Adaptive / overlap / sink / recompute ablation|

Reproduce the headline numbers:

```bash
cd benchmarks_and_experiments
python long_bench_arena.py \
    --model Qwen/Qwen2.5-3B \
    --n-samples 1000 \
    --recompute-strategy selective \
    --chunk-boundary-window 16 \
    --overlap-k 16 \
    --sink-tokens 32 \
    --output results/ablation_all_selective.json
```

Raw JSON outputs are in
[`benchmarks_and_experiments/results/`](benchmarks_and_experiments/results/).

---

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

## Limitations and known sharp edges

- **Accuracy tax scales with reuse.** At >70% reuse, expect 5–10 points
  of accuracy loss on hard long-context tasks. Validate on your own
  data.
- **Total-latency can regress on short-output workloads.** TTFT wins
  don't show up in wall-clock if you generate 5 tokens. The 3B bug-loc
  MC run is the clearest example.
- **No free lunch on small models.** Below ~1B params, or below ~500
  tokens of shared context, the hashing and stitching cost dominates.
- **Greedy-equivalent ≠ task-equivalent.** Logit cosine 1.0 still
  coexists with a 2.2pp accuracy gap — a perturbation that doesn't flip
  the first argmax can still flip token 40.
- **`kv_cache_bits=4` is unvalidated for your workload by default.**
  Run `verify_correctness()` first.

---

## License

[MIT](LICENSE)
