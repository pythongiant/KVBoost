Engine API
==========

.. module:: kvboost

KVBoost
-------

``KVBoost`` is the alias for :class:`kvboost.engine.InferenceEngine`.

.. autoclass:: kvboost.engine.InferenceEngine
   :members: from_pretrained, generate, generate_batch, generate_many, warm, cache_stats, verify_correctness
   :undoc-members:

GenerationMode
--------------

.. autoclass:: kvboost.engine.GenerationMode
   :members:
   :undoc-members:

RecomputeStrategy
-----------------

.. autoclass:: kvboost.engine.RecomputeStrategy
   :members:
   :undoc-members:

GenerationResult
----------------

.. autoclass:: kvboost.engine.GenerationResult
   :members:
   :undoc-members:
