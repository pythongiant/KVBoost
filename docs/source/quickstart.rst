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
   print(f"TTFT: {result.ttft_ms:.1f} ms | Cache reuse: {result.kv_reuse_ratio:.0%}")

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

Or auto-group mixed prompts by detected prefix:

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
       kv_cache_bits=8,                  # int8 quantized KV (2x RAM savings)
       disk_cache_dir="/tmp/kv",         # evicted chunks spill to disk
       recompute_strategy="cacheblend",  # deviation-guided seam repair
   )

Recommended Long-Context Configuration
---------------------------------------

For long documents (1K–6K tokens), use the full set of continuity features
validated in the 0.4.0 benchmarks. This is the configuration that produced
99.2% accuracy at 72.9% avg KV reuse:

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       recompute_strategy="cacheblend",
       max_cache_bytes=1.5e9,        # 1.5 GB hot tier
       chunk_boundary_window=16,     # snap splits to sentence boundaries
       overlap_k=16,                 # encode each chunk with 16-token overlap
       sink_tokens=32,               # carry the first 32 tokens as a global prefix
       recency_window_chunks=8,
   )

See :doc:`guide/continuity` for what each knob does.

When Does It Help?
------------------

.. list-table::
   :header-rows: 1

   * - Condition
     - Benefit
   * - Multi-turn chat, 3B+ model, shared system prompt
     - High: prefix-hash hits grow with history length.
   * - Long-context retrieval / multi-hop QA (1K–6K tokens)
     - High: validated in 0.4.0 benchmarks (10.1× warm TTFT speedup).
   * - Document / code reuse, 800+ tokens
     - High: non-prefix interior reuse via content-hash fallback.
   * - RAG with short context (~500 tokens)
     - Marginal: caching overhead is a larger share of total time.
   * - System prompt only, ~250 tokens
     - Net negative: overhead exceeds prefill savings.
   * - Any workload, <1B model
     - Net negative: prefill is already cheap.

Rule of thumb: benefits appear on **3B+ models** with **500+ token
shared context**. See :doc:`benchmarks/overview` for full results with
real numbers.
