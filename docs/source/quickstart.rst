Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   from kvboost import KVBoost

   # 1. Load any HuggingFace causal LM (must use RoPE)
   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")

   # 2. Cache your system prompt / document / few-shot examples
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

Batch Inference
---------------

Process multiple prompts sharing a common prefix in one pass:

.. code-block:: python

   results = engine.generate_batch([
       "You are a helpful assistant...\n\nUser: What is 2+2?",
       "You are a helpful assistant...\n\nUser: What is 3+3?",
       "You are a helpful assistant...\n\nUser: What is 4+4?",
   ])
   for r in results:
       print(r.output_text)

Or auto-group mixed prompts:

.. code-block:: python

   results = engine.generate_many([
       "System A...\n\nQuery 1",
       "System A...\n\nQuery 2",   # batched with above
       "System B...\n\nQuery 3",   # processed separately
   ])

Memory-Efficient Mode
---------------------

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       kv_cache_bits=8,              # int8 quantized KV (2x RAM savings)
       disk_cache_dir="/tmp/kv",     # evicted chunks go to disk
       recompute_strategy="cacheblend",  # smarter recompute
   )

When Does It Help?
------------------

.. list-table::
   :header-rows: 1

   * - Condition
     - Expected TTFT Speedup
   * - Multi-turn, 8+ turns, 3B+ model
     - **10-48x**
   * - Code/doc reuse, 800+ tokens
     - **15-21x**
   * - RAG, ~500 tokens
     - 1-2x
   * - System prompt, ~250 tokens
     - 0.3-0.5x (overhead)
   * - Any workload, 0.5B model
     - <1x (overhead)

Rule of thumb: benefits appear on **3B+ models** with **500+ token shared context**.
