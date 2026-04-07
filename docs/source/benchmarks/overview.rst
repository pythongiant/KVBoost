Benchmark Results
=================

All benchmarks on Qwen/Qwen2.5-3B (float16) on MacBook Air M-series, 16GB RAM,
MPS backend. Chunk size 128, greedy decoding.

Multi-Turn Conversation
-----------------------

Baseline TTFT scales linearly with history. KVBoost stays flat at ~62ms.

.. list-table::
   :header-rows: 1

   * - Turn
     - Tokens
     - Baseline
     - KVBoost
     - Reuse
     - Speedup
   * - 1
     - 232
     - 35ms
     - 31ms
     - 0%
     - 1.1x
   * - 4
     - 621
     - 374ms
     - 62ms
     - 62%
     - **6.0x**
   * - 6
     - 946
     - 1,228ms
     - 63ms
     - 68%
     - **19.6x**
   * - 8
     - 1,353
     - 2,970ms
     - 62ms
     - 76%
     - **47.9x**

Code Context Reuse (~800 tokens)
--------------------------------

.. list-table::
   :header-rows: 1

   * - Query
     - Baseline
     - KVBoost
     - Reuse
     - Speedup
   * - Q1 (cold)
     - 1,670ms
     - 2,292ms
     - 0%
     - 0.7x
   * - Q2 (warm)
     - 1,577ms
     - **75ms**
     - 92%
     - **21.1x**
   * - Q3 (warm)
     - 2,133ms
     - **128ms**
     - 92%
     - **16.6x**

Running Benchmarks
------------------

.. code-block:: bash

   # All examples
   python examples/run.py

   # Single example
   python examples/run.py --example multiturn

   # Full experiment suite (TinyLlama, ~55 min)
   cd benchmarks_and_experiments && python run_all.py

   # Distribution correctness test
   python benchmarks_and_experiments/11_distribution_correctness.py
