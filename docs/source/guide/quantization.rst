KV Cache Quantization
=====================

KVBoost can compress cached KV tensors from float16 to int8 or int4 using
KIVI-style asymmetric quantization (ICML 2024).

The key insight from KIVI: key cache has channel-specific outliers and should
be quantized **per-channel**, while value cache has token-specific outliers
and should be quantized **per-token**. Treating them identically causes
accuracy degradation.

Usage
-----

.. code-block:: python

   from kvboost import KVBoost

   # int8 -- 2x RAM savings, near-lossless (recommended)
   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B", kv_cache_bits=8)

   # int4 -- 4x RAM savings, aggressive
   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B", kv_cache_bits=4)
   assert engine.verify_correctness()  # validate before trusting

   # float16 -- no quantization (default)
   engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B", kv_cache_bits=16)

Memory Impact
-------------

For Qwen2.5-3B at chunk_size=128:

.. list-table::
   :header-rows: 1

   * - Precision
     - Per-chunk
     - 128 chunks (hot tier)
     - Quality
   * - float16
     - 9.4 MB
     - 1.2 GB
     - Baseline
   * - int8
     - 4.7 MB
     - 0.6 GB
     - Near-lossless (max error ~0.016)
   * - int4
     - 2.4 MB
     - 0.3 GB
     - Aggressive (validate first)

How It Works
------------

Quantization happens transparently:

1. **On store:** ``cache_manager.store()`` quantizes KV tensors and stores
   the compressed version + scale factors.
2. **On load:** ``cache_manager.get()`` dequantizes back to float16 before
   returning to the caller.

The caller never sees quantized tensors -- the compression is internal to
the cache tier.

Standalone API
--------------

For advanced use, the quantization functions are available directly:

.. code-block:: python

   from kvboost import quantize_kv, dequantize_kv

   # Compress
   qkv = quantize_kv(past_key_values, bits=8)
   print(f"Compressed: {qkv.memory_bytes() / 1e6:.1f} MB")

   # Decompress
   past_key_values = dequantize_kv(qkv)
