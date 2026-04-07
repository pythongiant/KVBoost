KVBoost vs vLLM Prefix Caching
==============================

Head-to-head against `vLLM-MLX <https://github.com/waybarrios/vllm-mlx>`_
prefix caching on Apple Silicon.

vLLM caches system prompt KV and reuses on exact prefix match. KVBoost reuses
any matching chunk, including non-prefix interior content.

.. note::

   KVBoost uses Qwen2.5-3B float16 (MPS). vLLM-MLX uses Qwen2.5-3B 4-bit
   (MLX Metal). This is the realistic deployment comparison where each
   system uses its optimal format.

Axis 1: Non-Prefix Interior Reuse
----------------------------------

The core differentiator. Same document placed at the start, in the middle,
or as unique content:

- **Exact prefix:** Both systems can cache. KVBoost is 3-30x faster.
- **Interior:** vLLM gets zero cache hits. KVBoost achieves 82-83% reuse.
- **No reuse:** Even without caching, KVBoost's HF baseline (33ms) beats
  vLLM-MLX (1.3s) due to MPS vs MLX Metal overhead.

Axis 2: Cold-Start Overhead
----------------------------

Empty cache, no reuse possible. KVBoost at 32ms vs vLLM at 777ms -- even
cold, KVBoost is faster because of the underlying engine difference.

Axis 3: Break-Even Prompt Length
--------------------------------

.. list-table::
   :header-rows: 1

   * - Length
     - KVBoost (warm)
     - vLLM (warm)
   * - ~250 words
     - **37ms** (89%)
     - 849ms
   * - ~500 words
     - **48ms** (88%)
     - 1,960ms
   * - ~1000 words
     - **242ms** (98%)
     - 61,131ms
   * - ~2000 words
     - **1,452ms** (98%)
     - 66,714ms

Running
-------

.. code-block:: bash

   pip install vllm-mlx
   python benchmarks_and_experiments/benchmark_vs_vllm.py
   python benchmarks_and_experiments/benchmark_vs_vllm.py --axis non_prefix
