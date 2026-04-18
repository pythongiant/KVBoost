Benchmarks
==========

KVBoost ships a single long-context benchmark harness:
``benchmarks_and_experiments/long_bench_arena.py``. It streams documents
from `LongBench <https://github.com/THUDM/LongBench>`_ (v1 and v2),
bucketizes by context length, and runs each sample twice — once with
the standard HF baseline and once with KVBoost — reporting:

- **Accuracy** (task-specific, per bucket) with a paired-bootstrap p-value
  against the baseline.
- **TTFT and end-to-end latency** per bucket.
- **KV reuse ratio** per bucket.

Raw JSON outputs are no longer tracked in the repo (see ``.gitignore``).
Rerun the harness locally — numbers depend on hardware, model, and
sampling configuration.

How to Run
----------

.. code-block:: bash

   # Full LongBench v1 + v2 sweep (baseline vs KVBoost)
   python benchmarks_and_experiments/long_bench_arena.py \
       --model Qwen/Qwen2.5-3B \
       --n-samples 200 \
       --output results/long_bench.json

   # Skip re-running baseline logits (faster iteration)
   python benchmarks_and_experiments/long_bench_arena.py \
       --skip-baseline-logits \
       --output results/kv_only.json

   # Accuracy vs HF + vLLM-MLX on HellaSwag / ARC / MMLU / GSM8K / TruthfulQA
   python benchmarks_and_experiments/benchmark_accuracy_vs_vllm.py

Ablation Harness
----------------

``run_ablation.sh`` exercises each continuity feature in isolation and
combined. Use it to reproduce the numbers that accompany a new
configuration, or to sanity-check a change to
:doc:`../guide/continuity`.

.. code-block:: bash

   cd benchmarks_and_experiments
   ./run_ablation.sh --n-samples 100

Configurations covered:

- ``baseline-recompute`` — selective recompute only (the 0.2.0 default).
- ``adaptive-only`` — adaptive chunk boundaries (``chunk_boundary_window=16``).
- ``overlap-only`` — overlap encoding (``overlap_k=16``).
- ``sink-only`` — attention sinks (``sink_tokens=32``).
- ``all-features`` — selective recompute + all three continuity features.
- ``cacheblend-all`` — CacheBlend recompute + all three continuity features.

What to Look For
----------------

LongBench results are reported per context-length bucket (``0-512``,
``512-1K``, ``1K-2K``, ``2K-4K``, ``4K+``). Three things to check on any
new run:

1. **Accuracy delta per bucket.** The paired bootstrap p-value flags
   buckets where KVBoost diverges from the baseline beyond what chance
   explains. The short-context bucket should always be PASS — if it
   isn't, there's a correctness bug, not a caching trade-off.
2. **TTFT speedup grows with context length.** The ``2K-4K`` and
   ``4K+`` buckets are where caching actually earns its keep. Flat
   speedup across buckets usually means the cache isn't being hit.
3. **KV reuse ratio.** If reuse is below ~20% on long contexts, the
   prompts don't actually share prefixes — the benchmark is measuring
   cold-start overhead, not reuse.

See Also
--------

- :doc:`../guide/continuity` — tuning adaptive boundaries, overlap
  encoding, and attention sinks to trade accuracy for speed.
- :doc:`../guide/recompute` — selective vs CacheBlend recompute.
- :doc:`vs-vllm` — accuracy vs vLLM-MLX prefix caching.
