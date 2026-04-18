Changelog
=========

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

**New: LongBench Arena harness** (``benchmarks_and_experiments/long_bench_arena.py``)

- Single benchmark script replaces the per-experiment files that were
  removed from version control.
- Streams LongBench v1 and v2 samples, bucketizes by context length
  (``0-512``, ``512-1K``, ``1K-2K``, ``2K-4K``, ``4K+``), and reports
  per-bucket accuracy with a paired-bootstrap p-value, TTFT, total
  latency, and KV reuse.
- Checkpointing so long runs can resume without redoing baseline logits
  (``--skip-baseline-logits``).

**New: Ablation harness** (``benchmarks_and_experiments/run_ablation.sh``)

- Exercises each continuity feature independently and combined against
  the selective-recompute-only baseline, plus a CacheBlend + all-features
  configuration.

**Fixed: Cross-pair cache bleed in long-context benchmarks**

- The long-context benchmark loop was reusing a warm cache across
  unrelated document pairs, inflating KVBoost's measured speedup. The
  cache is now cleared between pair groups (``cache_manager.py``,
  ``engine.py``, ``cacheblend.py``).

**Changed: Default device selection** (``compat.py``)

- ``default_device()`` centralises the CUDA → MPS → CPU fallback. All
  subsystems (``engine``, ``batch``, ``cache_manager``, ``cacheblend``,
  ``selective_recompute``) route through it.

**Removed: Stale benchmark artefacts**

- Per-experiment scripts and their JSON result files were deleted from
  the repo and added to ``.gitignore``. The single LongBench arena
  harness replaces them.
- Docs no longer quote per-turn/per-query speedup tables that referred
  to removed result files. See :doc:`benchmarks/overview` for how to
  reproduce locally.


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
- Fixes RoPE position collision and cross-chunk attention contamination
  bugs.

**New: KV cache quantization** (``kv_quantize.py``)

- KIVI-style asymmetric int8/int4: keys per-channel, values per-token.
- ``kv_cache_bits=8`` for 2x RAM savings, ``kv_cache_bits=4`` for 4x.
- Quantize-on-store, dequantize-on-load (transparent to callers).

**New: Batch inference** (``batch.py``)

- ``generate_batch()`` for prompts sharing a common prefix.
- ``generate_many()`` with auto-grouping by prefix hash.
- Zero-copy KV broadcast via ``expand()``.

**New: Flat block-pool disk tier** (``disk_tier.py``)

- Replaces per-file ``torch.save/load`` with a single pre-allocated
  binary.
- Atomic index persistence via ``os.replace()``.
- Async-friendly reads with ``non_blocking=True``.

**New: Model compatibility checks** (``compat.py``)

- 11 supported RoPE architectures, 6 blocked (ALiBi, absolute
  embeddings).
- ``strict=True/False`` on ``from_pretrained()``.
- ``verify_correctness()`` self-test for untested models.

**Improved: Cache eviction**

- Frequency-based eviction replaces pure LRU.
- System prompt chunks protected, one-off documents evicted first.

**Improved: warm() diagnostics**

- Returns ``WarmResult`` with alignment warnings.
- Logs when prompt length is not a multiple of chunk_size.

**New: Accuracy benchmarks** (``benchmark_accuracy_vs_vllm.py``)

- Five HuggingFace benchmarks: HellaSwag, ARC-Challenge, MMLU, GSM8K,
  TruthfulQA.
- Validates near-zero accuracy degradation vs HF baseline.
- Optional head-to-head comparison with vLLM-MLX.

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
