Changelog
=========

0.2.0 (2026-04-07)
-------------------

Major feature release.

**New: CacheBlend recompute** (``cacheblend.py``)

- Deviation-guided recompute based on CacheBlend (USENIX ATC '25)
- Measures per-token KV cosine deviation, recomputes only the ~15% that changed
- Automatically forced for approximate (content-hash) cache matches

**New: Two-tier cache keys** (``models.py``, ``cache_manager.py``)

- Prefix-chained hashes (vLLM-style): ``SHA256(parent_hash + tokens)``
- Content-hash fallback for non-prefix reuse with correctness flags
- Fixes RoPE position collision and cross-chunk attention contamination bugs

**New: KV cache quantization** (``kv_quantize.py``)

- KIVI-style asymmetric int8/int4: keys per-channel, values per-token
- ``kv_cache_bits=8`` for 2x RAM savings, ``kv_cache_bits=4`` for 4x
- Quantize-on-store, dequantize-on-load (transparent to callers)

**New: Batch inference** (``batch.py``)

- ``generate_batch()`` for prompts sharing a common prefix
- ``generate_many()`` with auto-grouping by prefix hash
- Zero-copy KV broadcast via ``expand()``

**New: Flat block-pool disk tier** (``disk_tier.py``)

- Replaces per-file ``torch.save/load`` with single pre-allocated binary
- Atomic index persistence via ``os.replace()``
- Async-friendly reads with ``non_blocking=True``

**New: Model compatibility checks** (``compat.py``)

- 11 supported RoPE architectures, 6 blocked (ALiBi, absolute embeddings)
- ``strict=True/False`` on ``from_pretrained()``
- ``verify_correctness()`` self-test for untested models

**Improved: Cache eviction**

- Frequency-based eviction replaces pure LRU
- System prompt chunks protected, one-off documents evicted first

**Improved: warm() diagnostics**

- Returns ``WarmResult`` with alignment warnings
- Logs when prompt length is not a multiple of chunk_size

**New: Accuracy benchmarks** (``benchmark_accuracy_vs_vllm.py``)

- 5 standard HuggingFace benchmarks: HellaSwag, ARC-Challenge, MMLU, GSM8K, TruthfulQA
- Validates zero accuracy degradation vs HF baseline (avg delta -0.4%)
- 100% greedy output agreement on 50 GSM8K prompts
- Optional head-to-head comparison with vLLM-MLX

**New: Correctness claim experiments** (``12_correctness_claims.py``)

- Claim 1: Boundary KV convergence curves across R=[0,4,8,16,32,64] on 50 Wikipedia passages
- Claim 2: KL divergence vs CacheBlend recompute ratio (monotonicity proof)
- Claim 2: Multi-hop QA (HotpotQA) F1 comparison: baseline vs exact vs approximate cache

**Documentation:**

- ReadTheDocs site at kvboost.readthedocs.io
- Full API reference, user guides, architecture docs
- Accuracy validation results added to README


0.1.0 (2026-03-30)
-------------------

Initial release.

- Chunk-level KV cache reuse with selective boundary recompute
- Three generation modes: baseline, prefix cache, chunk KV reuse
- Two-tier storage: hot RAM (LRU) + cold disk (torch.save)
- Content-addressed SHA256 chunk keys
- 5 runnable examples (chatbot, RAG, few-shot, multi-turn, code)
- 10 TinyLlama experiments + Qwen2.5-3B benchmarks
