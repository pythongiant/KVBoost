Recompute Strategies
====================

When chunks are stitched together, their KV tensors are "stale" -- each chunk
was encoded independently without cross-chunk attention. KVBoost offers three
strategies to fix this.

Selective (default)
-------------------

Recomputes the last ``R`` tokens at each chunk boundary seam.

.. code-block:: python

   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B", recompute_strategy="selective")

- **Cost:** ``O(R * num_seams)``
- **Quality:** Fixes boundary tokens only; interior tokens that depend on
  cross-chunk context are not corrected.
- **When to use:** Safe default. Works well when chunks were encoded in order.

CacheBlend
----------

Runs a forward pass over the assembled KV, measures per-token cosine deviation
between cached and full-context KV vectors, and recomputes only the ~15% of
tokens that actually deviate.

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       recompute_strategy="cacheblend",
       recompute_ratio=0.15,  # fraction of tokens to recompute
   )

- **Cost:** One cheap forward pass + recompute of ~15% of tokens.
- **Quality:** Fixes tokens by deviation, not position. Catches interior
  tokens that selective misses.
- **When to use:** Long prompts where selective recompute is too expensive,
  or when chunks come from different contexts (approximate matches).
- **Reference:** Based on CacheBlend (USENIX ATC '25).

None
----

Skips recompute entirely.

.. code-block:: python

   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B", recompute_strategy="none")

- **Cost:** Zero.
- **Quality:** No correction. Safe only when chunks were originally encoded
  in sequence (e.g., the same conversation cached across turns).
- **When to use:** Maximum speed, when you know the chunks share original context.

Approximate Match Behavior
--------------------------

When a chunk is matched by content hash (approximate) rather than prefix hash
(exact), KVBoost automatically forces CacheBlend regardless of your configured
strategy. This is because approximate matches have wrong position encodings
and/or wrong preceding context -- boundary-only repair is insufficient.
