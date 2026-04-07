Batch Inference
===============

KVBoost supports batched generation for multiple prompts sharing a common
prefix. The shared prefix KV is loaded from cache once, then broadcast
(zero-copy via ``expand()``) across the batch.

generate_batch()
----------------

Use when you know the prompts share a prefix:

.. code-block:: python

   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")
   engine.warm("You are a helpful assistant...")

   results = engine.generate_batch([
       "You are a helpful assistant...\n\nUser: What is 2+2?",
       "You are a helpful assistant...\n\nUser: Explain gravity",
       "You are a helpful assistant...\n\nUser: Write a haiku",
   ], max_new_tokens=64)

   for r in results:
       print(f"{r.output_text[:60]}... ({r.ttft_ms:.0f}ms)")

This runs:

1. One cache lookup for the shared prefix
2. One batched prefill over all suffix tokens
3. One batched decode loop

generate_many()
---------------

Use when prompts may or may not share prefixes -- auto-groups by prefix hash:

.. code-block:: python

   results = engine.generate_many([
       "System A...\n\nQuery 1",
       "System A...\n\nQuery 2",   # batched with Query 1
       "System B...\n\nQuery 3",   # processed separately
   ])

Prompts are clustered by the hash of their first few chunks. Each cluster
is batched together; single-prompt clusters fall back to ``generate()``.

How Broadcast Works
-------------------

The shared prefix KV has shape ``[1, heads, seq, dim]``. For batch size B,
KVBoost calls ``expand(B, -1, -1, -1)`` instead of ``repeat(B, 1, 1, 1)``:

- ``expand()`` creates a view with stride 0 -- **zero memory cost**, all
  batch elements read from the same physical tensor.
- ``repeat()`` would copy the tensor B times -- wasting B * chunk_MB of RAM.

Throughput Impact
-----------------

For the common case (same system prompt, N different queries):

- **Sequential:** N forward passes, each at ~10% GPU utilization
- **Batched:** 1 forward pass at ~80% GPU utilization, ~3x effective throughput

The batched prefill of N queries is typically ~2-3x slower than a single
query (not Nx), because the GPU is better utilized.
