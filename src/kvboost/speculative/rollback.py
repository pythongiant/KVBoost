# src/kvboost/speculative/rollback.py

"""KV cache truncation for speculative rollback.

After a verifier round writes K+1 positions of KV but the sampler accepts
only ``n < K`` of them, the unwanted tail must be discarded so the next
round starts from the correct context length. This module is the single
truncation primitive used by both ``DraftModel`` and ``TargetVerifier``.

Handles both tuple-of-tuples (legacy transformers / KVBoost's internal
format) and HF ``DynamicCache`` (modern transformers). The tuple path
reuses :meth:`KVCacheManager.slice_kv`.
"""

from __future__ import annotations

from typing import Any

from ..cache_manager import KVCacheManager


def truncate_past_kv(past_kv: Any, keep_n: int) -> Any:
    """Truncate ``past_kv`` to the first ``keep_n`` positions along seq dim.

    Returns a value of the same format as the input:
    - tuple-of-tuples in → tuple-of-tuples out (new tensors, views over the
      original storage where possible)
    - ``DynamicCache`` in → ``DynamicCache`` out, mutated in place AND
      returned (so callers can either chain or rely on identity)

    ``keep_n == current_length`` is a no-op. ``keep_n > current_length``
    raises ValueError — caller has lost track of the cache state.
    """
    if past_kv is None:
        if keep_n != 0:
            raise ValueError(
                f"cannot truncate to keep_n={keep_n}, past_kv is None"
            )
        return None

    if keep_n < 0:
        raise ValueError(f"keep_n must be >= 0, got {keep_n}")

    # Modern DynamicCache path.
    if hasattr(past_kv, "get_seq_length"):
        return _truncate_dynamic_cache(past_kv, keep_n)

    # Legacy tuple-of-tuples path.
    return _truncate_tuple_kv(past_kv, keep_n)


def _truncate_tuple_kv(past_kv: tuple, keep_n: int) -> tuple:
    current = KVCacheManager.kv_seq_len(past_kv)
    if keep_n > current:
        raise ValueError(
            f"keep_n={keep_n} exceeds current seq_len={current}"
        )
    if keep_n == current:
        return past_kv
    return KVCacheManager.slice_kv(past_kv, 0, keep_n)


def _truncate_dynamic_cache(cache: Any, keep_n: int) -> Any:
    """Truncate a transformers ``DynamicCache`` in place.

    Modern transformers (>=4.40) expose ``DynamicCache.crop(max_length)``;
    older versions need manual slicing of ``key_cache`` / ``value_cache``
    lists. Both paths are handled.
    """
    current = cache.get_seq_length()
    if keep_n > current:
        raise ValueError(
            f"keep_n={keep_n} exceeds current seq_len={current}"
        )
    if keep_n == current:
        return cache

    crop = getattr(cache, "crop", None)
    if callable(crop):
        crop(keep_n)
        return cache

    # Manual slice fallback. Both .key_cache and .value_cache are lists of
    # per-layer tensors of shape (batch, heads, seq, head_dim).
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        for i in range(len(cache.key_cache)):
            cache.key_cache[i] = cache.key_cache[i][:, :, :keep_n, :]
            cache.value_cache[i] = cache.value_cache[i][:, :, :keep_n, :]
        # _seen_tokens was the pre-4.45 bookkeeping field; keep it consistent
        # if present.
        if hasattr(cache, "_seen_tokens"):
            cache._seen_tokens = keep_n
        return cache

    raise TypeError(
        f"DynamicCache-like object lacks crop() and key_cache/value_cache; "
        f"got type {type(cache).__name__}"
    )
