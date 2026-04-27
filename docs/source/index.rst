KVBoost
=======

**Chunk-level KV cache reuse for faster HuggingFace inference.**

Drop-in speedup for any HuggingFace causal LM on repeated long context.
Reuse K/V tensors across requests that share long prefixes, with seam-repair
strategies (selective recompute and CacheBlend) that preserve full output quality.

.. code-block:: python

   from kvboost import KVBoost

   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")
   engine.warm("You are a helpful assistant...")
   result = engine.generate("You are a helpful assistant...\n\nHello!")
   print(f"{result.output_text}  ({result.ttft_ms:.0f} ms, {result.kv_reuse_ratio:.0%} reuse)")

**Benchmarked on Qwen/Qwen2.5-3B — 500 LongBench samples, vs vLLM prefix cache and HF baseline:**

- WARM TTFT: **63 ms** — 10.1× faster than HF baseline
- COLD TTFT: **222 ms** — 17% faster than vLLM (chunk-level partial hits on first access)
- Overall TTFT speedup: **4.49×** vs baseline
- Accuracy at 72.9% avg KV reuse: **99.2%** — identical to cold path, no quality degradation

See :doc:`benchmarks/overview` for full tables and methodology.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/recompute
   guide/continuity
   guide/quantization
   guide/batch
   guide/disk
   guide/compatibility

.. toctree::
   :maxdepth: 2
   :caption: Benchmarks

   benchmarks/overview
   benchmarks/vs-vllm

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/overview
   architecture/cache-keys

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/engine
   api/models
   api/cache_manager
   api/internals

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
