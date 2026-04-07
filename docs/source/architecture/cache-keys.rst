Cache Key Design
================

KVBoost uses a two-tier keying system inspired by vLLM v1's block hashes.

The Problem with Content-Only Keys
-----------------------------------

A naive ``SHA256(token_bytes)`` key means identical text at different positions
shares the same cache entry. This causes two bugs:

1. **RoPE position collision:** KV tensors cached at position 0 get loaded
   into a prompt where they should be at position 512. The assembled sequence
   has positions ``[0..127, 512..639]`` instead of ``[0..639]``.

2. **Cross-chunk attention contamination:** The same text in different
   conversations attended to different preceding context. The KV vectors
   encode "what this token is given everything before it" -- reusing them
   across contexts introduces semantic error.

Two-Tier Keys
-------------

.. list-table::
   :header-rows: 1

   * - Key Type
     - Formula
     - What It Encodes
   * - ``prefix_hash``
     - ``SHA256(parent_hash || token_bytes)``
     - Full prefix chain -- same tokens at different positions get different keys
   * - ``content_hash``
     - ``SHA256(token_bytes)``
     - Content only -- used for approximate reuse with mandatory full recompute

Lookup Order
------------

1. Try ``prefix_hash`` (exact match) -- positionally correct, use directly
2. Fall back to ``content_hash`` (approximate match) -- flag for CacheBlend
   full recompute, not just boundary repair

This preserves KVBoost's differentiator (non-prefix chunk reuse) while making
it correct. Approximate matches still hit the cache, but the system knows the
KV tensors need full correction.

Hash Chaining
-------------

.. code-block:: python

   from kvboost.models import chained_hash, content_hash_from_tokens

   tokens = [1, 2, 3, 4]

   # Content hash -- same for same tokens regardless of position
   c = content_hash_from_tokens(tokens)

   # Chained hash -- different for different preceding context
   h1 = chained_hash(tokens, parent_hash=None)    # first chunk
   h2 = chained_hash(tokens, parent_hash="abc")   # after chunk "abc"

   assert h1 != h2  # same tokens, different keys
