Accuracy vs vLLM-MLX
====================

KVBoost's correctness bar is "does the cached path answer the same
questions the baseline does." ``benchmark_accuracy_vs_vllm.py`` runs
that check against five HuggingFace benchmarks and, optionally, against
`vLLM-MLX <https://github.com/waybarrios/vllm-mlx>`_ prefix caching on
Apple Silicon.

What it measures
----------------

- **HellaSwag, ARC-Challenge, MMLU, GSM8K, TruthfulQA.** Standard
  HuggingFace evaluation sets, run via ``datasets``.
- **Baseline vs chunk-reuse agreement.** For every prompt, the harness
  compares KVBoost's first-token logits and final answer against the
  unmodified HF path. The expected result is a near-zero accuracy delta
  across tasks.
- **Optional vLLM head-to-head.** With ``--compare-vllm``, the same
  prompts are routed through vLLM-MLX and scored side-by-side.

How to run
----------

.. code-block:: bash

   # HF baseline vs KVBoost only
   python benchmarks_and_experiments/benchmark_accuracy_vs_vllm.py \
       --model Qwen/Qwen2.5-3B

   # Add vLLM-MLX as a third column (Apple Silicon)
   pip install vllm-mlx
   python benchmarks_and_experiments/benchmark_accuracy_vs_vllm.py \
       --model Qwen/Qwen2.5-3B \
       --compare-vllm

Caveats for cross-engine comparisons
------------------------------------

KVBoost runs HuggingFace models in float16 on MPS/CUDA/CPU. vLLM-MLX
runs MLX-quantized weights on Metal. The two engines are not
apples-to-apples on latency — the backend difference dominates at short
context lengths. The useful comparison is on **non-prefix interior
reuse** and on **accuracy deltas**, where KVBoost's content-hash
fallback gives it semantic advantages that exact-prefix caches
structurally can't reach.

For latency trends as a function of context length, use the LongBench
harness in :doc:`overview` instead — it runs both engines on the same
hardware and reports per-bucket TTFT.
