Architecture Overview
=====================

KVBoost is composed of six core modules that form a pipeline:

.. code-block:: text

   prompt_text
       |
       v
    tokenize
       |
       v
    ChunkRegistry.split()            -- split into fixed/semantic chunks
       |
       v
    KVCacheManager.find_matches()    -- two-tier hash lookup per chunk
       |
       v
    PromptAssembler.assemble()       -- stitch cached KV + live tokens
       |
       v
    Recompute (selective | cacheblend)  -- fix stale KV from stitching
       |
       v
    model.forward(live_tokens, past_key_values=cached_kv)
       |
       v
    GenerationResult

Module Responsibilities
-----------------------

**ChunkRegistry** (``chunk_registry.py``)
   Splits token sequences into cacheable chunks. Supports fixed-size,
   semantic (paragraph-aligned), and document-level strategies.

**KVCacheManager** (``cache_manager.py``)
   Two-tier storage (hot RAM + cold disk) with two-tier keying
   (prefix-chained + content-only). Frequency-based eviction protects
   frequently-used system prompt chunks.

**PromptAssembler** (``prompt_assembler.py``)
   Given a token sequence and the cache, assembles the cached KV prefix
   and identifies which tokens need fresh computation.

**SelectiveRecompute** (``selective_recompute.py``)
   Fixes chunk boundary seams by recomputing the last R tokens at each
   junction with full cross-chunk attention.

**CacheBlendRecompute** (``cacheblend.py``)
   Deviation-guided recompute: measures per-token KV deviation and
   recomputes only the ~15% that actually changed.

**InferenceEngine** (``engine.py``)
   Ties everything together. Exposes ``generate()``, ``generate_batch()``,
   ``generate_many()``, ``warm()``, and ``verify_correctness()``.

Supporting Modules
------------------

**kv_quantize.py**
   KIVI-style asymmetric int8/int4 quantization for compressed KV storage.

**disk_tier.py**
   Flat block-pool disk cache with atomic index persistence.

**batch.py**
   Batched inference primitives: prefix detection, zero-copy KV broadcast,
   padded decode loops.

**compat.py**
   Model architecture validation. Blocks ALiBi/absolute-position models,
   warns on untested architectures.

**models.py**
   Core data structures: ``CachedChunk``, ``AssembledPrompt``, ``WarmResult``,
   and the two-tier hashing functions.
