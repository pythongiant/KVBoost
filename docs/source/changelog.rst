Changelog
=========

0.1.0 (2026-04-07)
-------------------

Initial release.

**Core Features:**

- Chunk-level KV cache reuse for any RoPE-based HuggingFace causal LM
- Three generation modes: baseline, prefix cache, chunk KV reuse
- Two recompute strategies: selective boundary and CacheBlend deviation-guided
- Prefix-chained cache keys (vLLM-style) for positional correctness
- Content-hash fallback for approximate non-prefix reuse

**Storage:**

- Two-tier cache: hot RAM + cold disk (flat mmap block pool)
- KIVI-style int8/int4 KV quantization (2-4x compression)
- Frequency-based eviction (protects system prompt chunks)

**Batch Inference:**

- ``generate_batch()`` for prompts sharing a common prefix
- ``generate_many()`` with auto-grouping by prefix hash
- Zero-copy KV broadcast via ``expand()``

**Quality & Safety:**

- Model compatibility validation (11 supported + 6 blocked architectures)
- ``verify_correctness()`` self-test for untested models
- ``WarmResult`` diagnostics with alignment warnings

**Benchmarks:**

- 5-48x TTFT reduction on 3B+ models with 500+ token shared context
- 3-41x faster than vLLM-MLX prefix caching
- 10 TinyLlama experiments + Qwen2.5-3B validation suite
