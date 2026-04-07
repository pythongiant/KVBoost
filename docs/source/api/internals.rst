Internal Modules
================

These modules are used internally by KVBoost but are available for advanced use.

KV Quantization
---------------

.. module:: kvboost.kv_quantize

.. autofunction:: kvboost.kv_quantize.quantize_kv

.. autofunction:: kvboost.kv_quantize.dequantize_kv

.. autoclass:: kvboost.kv_quantize.QuantizedKV
   :members:
   :undoc-members:

Disk Tier
---------

.. module:: kvboost.disk_tier

.. autoclass:: kvboost.disk_tier.DiskTier
   :members: write, read, contains, remove, stats
   :undoc-members:

Batch Utilities
---------------

.. module:: kvboost.batch

.. autofunction:: kvboost.batch.find_common_chunk_prefix

.. autofunction:: kvboost.batch.broadcast_kv

.. autofunction:: kvboost.batch.group_by_prefix

.. autofunction:: kvboost.batch.pad_and_mask

CacheBlend Recompute
--------------------

.. module:: kvboost.cacheblend

.. autoclass:: kvboost.cacheblend.CacheBlendRecompute
   :members: apply
   :undoc-members:

Selective Recompute
-------------------

.. module:: kvboost.selective_recompute

.. autoclass:: kvboost.selective_recompute.SelectiveRecompute
   :members: apply
   :undoc-members:

Chunk Registry
--------------

.. module:: kvboost.chunk_registry

.. autoclass:: kvboost.chunk_registry.ChunkRegistry
   :members: split, chunk_ids_for
   :undoc-members:

.. autoclass:: kvboost.chunk_registry.ChunkStrategy
   :members:
   :undoc-members:

Model Compatibility
-------------------

.. module:: kvboost.compat

.. autofunction:: kvboost.compat.check_model_compatibility

.. autodata:: kvboost.compat.SUPPORTED_ARCHITECTURES

.. autodata:: kvboost.compat.UNSUPPORTED_ARCHITECTURES
