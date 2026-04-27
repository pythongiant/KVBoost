Changelog
=========

0.4.0 (2026-04-27)
-------------------

Benchmark and correctness release. Comprehensive 3-way comparison against
vLLM (prefix cache) and HuggingFace baseline, plus two correctness fixes
that affected tensor memory and logit accuracy on newer transformers versions.

**New: 3-way benchmark suite** (``benchmarks_and_experiments/important/``)

- ``run_experiment.py`` runs each backend as an isolated subprocess (clean
  CUDA context), loads atomic checkpoints, and prints comparison tables for
  accuracy, latency, and GPU memory.
- ``run_benchmarks.py`` runs individual backends with full CLI control over
  vLLM settings (``--vllm-gpu-memory-utilization``, ``--vllm-no-enforce-eager``,
  ``--vllm-max-num-seqs``).
- ``report.py`` generates a self-contained HTML report with inline SVG charts
  from any experiment JSON.
- ``plot_benchmarks.py`` generates six matplotlib figures: COLD/WARM TTFT bar
  chart, TTFT CDF, TTFT by context-length bucket, KV reuse histogram, speedup
  summary, and accuracy-by-reuse.

**New: Benchmark figures** (``docs/figures/``)

- Six figures derived from the 500-sample LongBench experiment embedded in
  the README and available for reuse.

**Fixed: Logit tensor bloat** (``engine.py``)

- A ``_forward_kwargs`` helper now probes ``model.forward`` once and picks
  ``logits_to_keep`` (transformers ≥ 4.45) or ``num_logits_to_keep`` (older),
  caching the choice on the engine. Returns ``{}`` if neither is present —
  fail-safe for any version. Previously, if any layer in the call chain
  dropped the kwarg, the default ``0`` caused the model to return a full
  ``[batch, seq, vocab]`` tensor instead of the last-token-only slice,
  silently inflating memory and latency.

**Fixed: CacheBlend first-pass memory** (``cacheblend.py``)

- CacheBlend's deviation-measurement forward pass now only reads
  ``out.past_key_values``; it no longer retains the full logit tensor from
  that pass. Reduces peak memory during seam repair.

**Fixed: vLLM max_model_len headroom** (all benchmark files)

- Changed ``max_model_len = max_context_tokens + 128`` to ``+ 512``.
  The formatted prompt (diff + question + 4 choices + template) adds
  ~400–500 tokens on top of the raw context token count; ``+128`` caused
  ``VLLMValidationError`` on longer samples mid-run.

**Results (Qwen/Qwen2.5-3B, 500 samples, CUDA):**

- WARM TTFT: **63 ms** — 10.1× faster than HF baseline
- COLD TTFT: **222 ms** — 17% faster than vLLM (chunk-level partial hits)
- Overall speedup: **4.49×** vs baseline
- Accuracy at 72.9% avg KV reuse: **99.2%** — identical to cold, no degradation


0.3.0 (2026-04-17)
-------------------

Long-context continuity release. Three new knobs on the engine trade a
little warm-up time for better accuracy where fixed-size chunking
previously cut mid-sentence or orphaned downstream chunks from the
document's attention-sink tokens.

**New: Adaptive chunk boundaries** (``chunk_registry.py``, ``engine.py``)

- ``chunk_boundary_window=W`` nudges each fixed split to the nearest
  sentence/clause-ending token within ±W. Falls back to the fixed
  position if no boundary is in range.
- Boundary tokens are derived from the tokenizer (``. ; ? ! \n \n\n``)
  — no configuration required.
- Window is clamped to ``chunk_size // 2`` so chunks can't fall below
  ``min_chunk_tokens``.
- Zero cost at generation time; splits are computed once at warm/store.

**New: Overlap encoding** (``engine.py``, ``models.py``)

- ``overlap_k=K`` prepends the previous chunk's last K tokens when
  encoding a new chunk, then strips them before storing the KV.
- Bakes cross-chunk attention into the cache entry itself. The stored
  KV still corresponds to exactly the chunk's own tokens.
- Tracked per-chunk via ``CachedChunk.overlap_prefix_len``.

**New: Attention sinks** (``engine.py``, ``models.py``)

- ``sink_tokens=S`` prepends the document's first S tokens as a global
  prefix while encoding every later chunk, then strips them before
  storage.
- Gives interior chunks the position-0 "sink" context they'd otherwise
  be missing.
- Tracked per-chunk via ``CachedChunk.sink_prefix_len``. Chunk 0 is
  exempt (the sink is already in its own tokens).

**Fixed: Cross-pair cache bleed in benchmarks**

- The benchmark loop was reusing a warm cache across unrelated document
  pairs, inflating KVBoost's measured speedup. The cache is now cleared
  between pair groups.

**Changed: Default device selection** (``compat.py``)

- ``default_device()`` centralises the CUDA → MPS → CPU fallback.


0.2.0 (2026-04-07)
-------------------

Major feature release.

**New: CacheBlend recompute** (``cacheblend.py``)

- Deviation-guided recompute based on CacheBlend (USENIX ATC '25).
- Measures per-token KV cosine deviation, recomputes only the ~15% that
  changed.
- Automatically forced for approximate (content-hash) cache matches.

**New: Two-tier cache keys** (``models.py``, ``cache_manager.py``)

- Prefix-chained hashes (vLLM-style): ``SHA256(parent_hash + tokens)``.
- Content-hash fallback for non-prefix reuse with correctness flags.
- Fixes RoPE position collision and cross-chunk attention contamination.

**New: KV cache quantization** (``kv_quantize.py``)

- KIVI-style asymmetric int8/int4: keys per-channel, values per-token.
- ``kv_cache_bits=8`` for 2× RAM savings, ``kv_cache_bits=4`` for 4×.
- Quantize-on-store, dequantize-on-load (transparent to callers).

**New: Batch inference** (``batch.py``)

- ``generate_batch()`` for prompts sharing a common prefix.
- ``generate_many()`` with auto-grouping by prefix hash.
- Zero-copy KV broadcast via ``expand()``.

**New: Flat block-pool disk tier** (``disk_tier.py``)

- Replaces per-file ``torch.save/load`` with a single pre-allocated binary.
- Atomic index persistence via ``os.replace()``.
- Async-friendly reads with ``non_blocking=True``.

**New: Model compatibility checks** (``compat.py``)

- 11 supported RoPE architectures, 6 blocked (ALiBi, absolute embeddings).
- ``strict=True/False`` on ``from_pretrained()``.
- ``verify_correctness()`` self-test for untested models.

**Improved: Cache eviction**

- Frequency-based eviction replaces pure LRU.
- System prompt chunks protected, one-off documents evicted first.

**Documentation:**

- ReadTheDocs site at ``kvboost.readthedocs.io``.
- Full API reference, user guides, architecture docs.


0.1.0 (2026-03-30)
-------------------

Initial release.

- Chunk-level KV cache reuse with selective boundary recompute.
- Three generation modes: baseline, prefix cache, chunk KV reuse.
- Two-tier storage: hot RAM (LRU) + cold disk (torch.save).
- Content-addressed SHA256 chunk keys.
- 5 runnable examples (chatbot, RAG, few-shot, multi-turn, code).
