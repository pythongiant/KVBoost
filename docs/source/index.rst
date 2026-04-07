KVBoost
=======

**Chunk-level KV cache reuse for faster HuggingFace inference.**

5-48x TTFT reduction on 3B+ models with repeated long context.

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
