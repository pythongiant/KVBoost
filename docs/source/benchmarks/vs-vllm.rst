KVBoost vs vLLM — Full Comparison
==================================

0.4.0 adds a rigorous 3-way benchmark against vLLM (prefix cache) and the
HuggingFace baseline. This page explains the methodology, what makes this
comparison meaningful, and where each backend has structural advantages.

Methodology
-----------

**Dataset:** 500 bug-localization samples from
`LongBench <https://github.com/THUDM/LongBench>`_ (multi-file code diff +
question + 4 choices). Max context: 6 000 tokens. Each sample appears twice —
once as a COLD query (fresh cache) and once as a WARM query (diff prefix
cached from the COLD pass).

**Isolation:** Each backend runs in a separate subprocess so CUDA contexts and
allocator state are completely independent. Results are not affected by
cross-backend memory fragmentation.

**Accuracy metric:** Exact-match on first-token logits for 4-choice questions.
The model outputs A/B/C/D; the benchmark compares that against the ground-truth
file path that was modified to fix the reported bug.

**KVBoost config:** ``cacheblend`` strategy, 1.5 GB RAM cache, recency
window 8, ``chunk_boundary_window=16``, ``overlap_k=16``, ``sink_tokens=32``.

**vLLM config:** prefix caching enabled, ``gpu_memory_utilization=0.95``,
``enforce_eager=True``, ``max_num_seqs=1``.

Results Summary
---------------

.. list-table::
   :header-rows: 1

   * - Metric
     - KVBoost
     - vLLM (prefix cache)
     - Baseline (HF)
   * - TTFT overall mean
     - **142 ms**
     - 166 ms
     - 639 ms
   * - TTFT COLD mean
     - **222 ms**
     - 269 ms
     - 639 ms
   * - TTFT WARM mean
     - 63 ms
     - **62 ms**
     - 640 ms
   * - TTFT p95
     - **506 ms**
     - 653 ms
     - 1 705 ms
   * - Overall speedup vs baseline
     - **4.49×**
     - 3.86×
     - 1.00×
   * - WARM speedup vs baseline
     - **10.1×**
     - **10.3×**
     - 1.00×
   * - Accuracy overall
     - **99.2%**
     - 99.1%
     - 99.1%
   * - Accuracy WARM
     - **99.2%**
     - 98.8%
     - 99.0%
   * - Avg KV reuse (warm)
     - **72.9%**
     - —
     - —

Where KVBoost Wins
------------------

**COLD queries.** vLLM's prefix cache only hits when the *full* prefix has been
seen before. KVBoost's chunk-level hashing hits on any subsequence that
overlaps with a cached chunk from any prior request. In this benchmark,
KVBoost COLD mean (222 ms) is **17% faster** than vLLM COLD (269 ms) — that
delta comes entirely from partial chunk hits on first access.

**Overall mean.** Because COLD queries are 17% faster, KVBoost's overall mean
(142 ms) beats vLLM (166 ms) even though WARM latency is identical (~62–63 ms).

**WARM accuracy.** KVBoost WARM accuracy (99.2%) matches COLD exactly.
vLLM WARM accuracy (98.8%) is 0.6 pp below its own COLD (99.4%). The
CacheBlend seam repair closes the quality gap that vLLM's exact-copy KV
reuse leaves open at chunk boundaries.

Where vLLM Wins
---------------

**Throughput.** vLLM's optimized batch scheduler achieves 13.2 tok/s vs
KVBoost's 11.7 tok/s. For throughput-bound serving, vLLM's engine is
purpose-built for that regime.

**WARM p95 latency.** vLLM WARM p95 (93 ms) is slightly tighter than
KVBoost WARM p95 (101 ms), because vLLM has no seam-repair overhead on
exact-prefix hits.

**Operational simplicity.** vLLM is a production serving system with
request batching, scheduling, and observability built in. KVBoost is a
library — you manage the serving layer.

When to Choose KVBoost
-----------------------

- You are deploying against a **HuggingFace model** and don't want to port
  to vLLM's engine format.
- Your workload has **non-prefix text reuse** — interior document chunks
  that appear across requests but not always at position 0.
- You need **full output quality** at high reuse; CacheBlend's deviation-guided
  repair produces no measurable accuracy delta even at 80–100% reuse.
- You want a **library**, not a serving system — fine-grained control over
  generation parameters, custom tokenizers, and direct ``past_key_values``
  access.

When to Choose vLLM
--------------------

- You need maximum **throughput** (multi-tenant, batched requests).
- Your workload is **exact-prefix dominated** (same system prompt, many
  different suffixes) and accuracy at high reuse is not a concern.
- You are already invested in vLLM's ecosystem (OpenAI-compatible API,
  deployment tooling, monitoring).

Caveats
-------

These results are on a single GPU (CUDA), single model (Qwen/Qwen2.5-3B),
and a single task type (bug localization). Numbers on other hardware, models,
or tasks will differ. The comparison is most meaningful for the
**COLD vs WARM delta** and **accuracy at high reuse** — those reflect
structural properties of each caching strategy, not benchmark-specific tuning.

Reproducing the Comparison
---------------------------

.. code-block:: bash

   cd benchmarks_and_experiments/important
   python run_experiment.py \
       --model Qwen/Qwen2.5-3B \
       --n-samples 500 \
       --max-context-tokens 6000 \
       --max-cache-bytes 1.5e9 \
       --recompute-strategy cacheblend \
       --chunk-boundary-window 16 \
       --overlap-k 16 \
       --sink-tokens 32

See :doc:`overview` for the full benchmark methodology and figure-generation
instructions.
