Cross-Chunk Continuity
======================

The naive chunk-level cache chops a prompt into fixed 128-token windows
and encodes each window in isolation. That works, but every seam is a
place where the next chunk's KV was produced without being able to
attend to the previous chunk. Three options in 0.3.0 let you pay a
little more on the warm-up to make those seams better.

All three are off by default. Turn them on when the ablation in
:doc:`../benchmarks/overview` shows the long-context buckets losing
accuracy.

Adaptive chunk boundaries
-------------------------

Fixed-size windows cut mid-sentence by construction. If the seam lands
between "self-" and "attention", the KV for "attention" was produced
without the model seeing the prefix that defines it. Adaptive boundaries
search a ±``chunk_boundary_window`` window around each fixed split point
for the nearest sentence- or clause-ending token and split there
instead.

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       chunk_boundary_window=16,   # ± tokens to search for a boundary
   )

- **Boundary tokens** are computed from the tokenizer for
  ``. ; ? ! \n \n\n`` — no configuration needed.
- **Window is clamped** to at most half ``chunk_size`` so chunks never
  shrink below ``min_chunk_tokens``.
- **Fallback:** if no boundary lives in the window, the original fixed
  split point is used.
- **Cost:** zero at warm-up, zero at generate. It only changes where
  the splits land.

Overlap encoding
----------------

At each seam the first few tokens of the next chunk were encoded without
the last few tokens of the previous chunk in context. ``overlap_k``
prepends the previous chunk's final ``k`` tokens while encoding the new
chunk, then strips them back out before storing the KV:

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       overlap_k=16,               # last-K tokens of previous chunk
   )

The stored KV still contains exactly the chunk's own tokens — the
overlap only shapes the attention that produced them. Think of it as
baking cross-chunk attention into the cache entry itself.

- **Cost at warm:** ``O(overlap_k)`` extra tokens per chunk in the
  encode pass.
- **Cost at generate:** zero.
- **Pairs well with adaptive boundaries** — adaptive puts the seam at a
  natural place, overlap makes sure the tokens right after it still
  see the clause that precedes them.

Attention sinks
---------------

Later chunks are encoded "mid-document," but at inference time the model
sees position 0 and expects attention sinks there. ``sink_tokens``
prepends the first ``S`` tokens of the document as a global prefix while
encoding every later chunk, again stripping them before storage:

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       sink_tokens=32,             # first-S tokens as a global prefix
   )

- **Chunk 0 is exempt** — its tokens already contain the sink.
- **Cost at warm:** ``O(sink_tokens)`` extra tokens per chunk after
  chunk 0.
- **Cost at generate:** zero.

Combining them
--------------

All three are independent and compose. The recommended long-context
configuration tested in ``run_ablation.sh`` is:

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       chunk_boundary_window=16,
       overlap_k=16,
       sink_tokens=32,
       recompute_strategy="cacheblend",
   )

When to leave them off
----------------------

- **Short prompts** (``< 512`` tokens): one chunk, no seams, no benefit.
- **Exact-prefix workloads** (multi-turn chat sharing the same system
  prompt): prefix-hash lookups already dominate. Continuity features
  add warm-up cost for no accuracy gain.
- **Cold cache / single-shot:** warm-up cost is never amortized.

The features earn their keep on long-context retrieval, multi-hop QA,
and content-hash (approximate) reuse — exactly the cases where
boundary-only recompute leaves the most on the table.
