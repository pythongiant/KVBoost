Disk Cache Tier
===============

When the hot RAM tier is full, evicted chunks are demoted to a disk-backed
cold tier. On cache hit, chunks are promoted back to RAM.

Usage
-----

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       disk_cache_dir="/tmp/kv_cache",
   )

Combine with quantization for maximum efficiency:

.. code-block:: python

   engine = KVBoost.from_pretrained(
       "Qwen/Qwen2.5-3B",
       kv_cache_bits=8,
       disk_cache_dir="/tmp/kv_cache",
   )

How It Works
------------

The disk tier uses a **flat block pool** rather than per-file storage:

.. code-block:: text

   /tmp/kv_cache/
     kv_cache.bin    <- single pre-allocated file, fixed-size slots
     kv_index.json   <- hash -> slot mapping (loaded into RAM)
     kv_meta.json    <- chunk metadata (token_ids, positions, etc.)

- **No pickle:** Raw tensor bytes written directly to slots. No
  serialization overhead.
- **Atomic index:** Index updates use write-to-temp + ``os.replace()``
  for crash safety.
- **Async-friendly:** On CUDA, reads use ``non_blocking=True`` for
  pipelined disk-to-GPU transfer.

Flow
----

1. **Eviction:** When the hot tier is full and a new chunk arrives, the
   lowest-frequency chunk is evicted. If a disk tier is configured, the
   evicted chunk's KV tensors are flattened and written to a slot.

2. **Promotion:** On cache miss in the hot tier, the disk tier is checked.
   If found, the chunk is read from its slot, reconstructed, and promoted
   back to the hot tier.

3. **Stats:** ``engine.cache_stats()`` includes disk tier info when enabled:

   .. code-block:: python

      stats = engine.cache_stats()
      # {'hot_chunks': 42, 'disk_chunks': 18, 'disk_free_slots': 238, ...}
