Benchmarks
==========

0.4.0 includes a comprehensive 3-way benchmark suite comparing KVBoost,
vLLM (prefix cache), and the HuggingFace baseline on **500 bug-localization
samples from** `JetBrains-Research/lca-bug-localization <https://huggingface.co/datasets/JetBrains-Research/lca-bug-localization>`_
(Qwen/Qwen2.5-3B, max 6 000 context tokens).

Each backend ran in an **isolated subprocess** for a clean GPU state.
Accuracy is measured as exact-match on 4-choice multiple-choice questions.

KVBoost config: ``cacheblend`` strategy, 1.5 GB cache, recency window 8,
``chunk_boundary_window=16``, ``overlap_k=16``, ``sink_tokens=32``.

Latency — Time to First Token
------------------------------

.. list-table::
   :header-rows: 1

   * - Backend
     - TTFT mean
     - TTFT p95
     - COLD mean
     - WARM mean
     - Throughput
     - vs Baseline
   * - **KVBoost**
     - **142 ms**
     - 506 ms
     - 222 ms
     - **63 ms**
     - 11.7 tok/s
     - **4.49×**
   * - vLLM (prefix cache)
     - 166 ms
     - 653 ms
     - 269 ms
     - **62 ms**
     - 13.2 tok/s
     - 3.86×
   * - Baseline (HF)
     - 639 ms
     - 1 705 ms
     - 639 ms
     - 640 ms
     - 4.7 tok/s
     - 1.00×

COLD = first query in a pair (no cached KVs).
WARM = second query after the diff prefix is cached from the first.

KVBoost WARM TTFT is **3.5× faster than its own COLD** and **10.1× faster than Baseline**.

Both caching backends reach nearly identical WARM latency (~62–63 ms). KVBoost's
lower overall mean (142 ms vs 166 ms) comes from its COLD path (222 ms vs 269 ms):
chunk-level partial cache hits let first-time queries reuse chunks cached by earlier
requests with overlapping text — an advantage exact-prefix caches structurally can't reach.

Accuracy
--------

.. list-table::
   :header-rows: 1

   * - Backend
     - Overall
     - COLD
     - WARM
     - Avg KV reuse (warm)
   * - **KVBoost**
     - **99.2%**
     - 99.2%
     - 99.2%
     - **72.9%**
   * - vLLM (prefix cache)
     - 99.1%
     - 99.4%
     - 98.8%
     - —
   * - Baseline (HF)
     - 99.1%
     - 99.2%
     - 99.0%
     - —

Cold accuracy spread is **0.2 pp** across all three backends, confirming identical inputs.
KVBoost WARM accuracy matches COLD exactly (99.2%) despite 72.9% avg KV reuse —
the CacheBlend seam repair produces no measurable quality degradation.

KV Reuse Distribution (KVBoost, warm queries only)
---------------------------------------------------

.. list-table::
   :header-rows: 1

   * - Reuse bucket
     - Share of warm queries
   * - 80–100%
     - 49%
   * - 60–80%
     - 25%
   * - 40–60%
     - 16%
   * - 20–40%
     - 10%
   * - 0–20%
     - 0%

49% of warm queries reuse more than 80% of their prefix from cache. Average: **72.9%**.

GPU Memory
----------

.. list-table::
   :header-rows: 1

   * - Backend
     - Peak mean
     - Peak p95
     - COLD mean
     - WARM mean
   * - **KVBoost**
     - 6 126 MB
     - 6 495 MB
     - 6 140 MB
     - 6 111 MB
   * - Baseline (HF)
     - 6 141 MB
     - 6 517 MB
     - 6 140 MB
     - 6 141 MB

KVBoost warm queries use ~29 MB less peak memory than cold queries, as cached chunks
skip the full prefill activation spike. vLLM manages CUDA memory internally via its
own allocator; ``torch.cuda.max_memory_allocated()`` returns 0 for vLLM allocations
and is excluded from this table.

How to Reproduce
----------------

The benchmark suite lives in ``benchmarks_and_experiments/important/``.

Run all three backends end-to-end in isolated subprocesses:

.. code-block:: bash

   cd benchmarks_and_experiments/important
   python run_experiment.py \
       --model Qwen/Qwen2.5-3B \
       --n-samples 500 \
       --max-context-tokens 6000 \
       --max-cache-bytes 1.5e9 \
       --recency-window-chunks 8 \
       --recompute-strategy cacheblend \
       --chunk-boundary-window 16 \
       --overlap-k 16 \
       --sink-tokens 32

Or run the backends individually and load existing checkpoints:

.. code-block:: bash

   # Run individual backends
   python run_benchmarks.py --max-cache-bytes 1.5e9 \
       --recompute-strategy cacheblend \
       --chunk-boundary-window 16 --overlap-k 16 --sink-tokens 32 \
       --n-samples 500 --max-context-tokens 6000

   # Reload checkpoints and print tables without re-running
   python run_experiment.py --skip-run

Generate the benchmark figures:

.. code-block:: bash

   python plot_benchmarks.py

This produces six matplotlib figures in ``docs/figures/``:
COLD/WARM TTFT bars, TTFT CDF, TTFT by context-length bucket,
KV reuse histogram, speedup summary, and accuracy-by-reuse.

Generate an HTML report from the experiment JSON:

.. code-block:: bash

   python report.py                          # auto-picks latest JSON
   python report.py results/experiment_*.json

What to Look For
----------------

Three sanity checks on any new run:

1. **Cold accuracy spread ≤ 0.5 pp across backends.** If it's larger,
   the backends are not processing identical inputs — check the prompt
   construction in ``run_benchmarks.py``.
2. **TTFT speedup grows with context length.** The ``2K–4K`` and ``4K+``
   buckets are where chunk reuse earns its keep. Flat speedup across
   buckets means the cache isn't being hit.
3. **KV reuse ratio ≥ 50% on warm queries.** If reuse is below ~20%,
   prompts don't share enough text for caching to matter.

See Also
--------

- :doc:`../guide/continuity` — tuning adaptive boundaries, overlap
  encoding, and attention sinks.
- :doc:`../guide/recompute` — selective vs CacheBlend recompute strategies.
- :doc:`vs-vllm` — full 3-way comparison methodology and caveats.
