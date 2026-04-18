KVBoost
=======

**Chunk-level KV cache reuse for faster HuggingFace inference.**

Drop-in speedup for 3B+ models on repeated long context, with cross-chunk
continuity knobs (adaptive boundaries, overlap encoding, attention sinks)
for accuracy on long documents.

.. code-block:: python

   from kvboost import KVBoost

   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")
   engine.warm("You are a helpful assistant...")
   result = engine.generate("You are a helpful assistant...\n\nHello!")
   print(result.output_text)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   quickstart
   installation

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
