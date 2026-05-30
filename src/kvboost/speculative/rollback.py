# src/kvboost/speculative/rollback.py

"""KV cache truncation and gather primitives for speculative rollback.

Two operations:

  - ``truncate_past_kv(past_kv, keep_n)`` — drop everything past
    ``keep_n``. Used by flat speculative when a draft tail is rejected.

  - ``gather_kv_columns(past_kv, base_columns, tail_indices)`` — keep
    the first ``base_columns`` columns of past_kv plus the specific
    speculative columns in ``tail_indices`` (in order). Used by tree
    speculative's ``commit_path`` to collapse the multi-branch tree
    KV into the single accepted path's contiguous KV.

Both handle tuple-of-tuples (legacy / KVBoost internal) and HF
``DynamicCache`` (modern transformers).
"""

from __future__ import annotations

from typing import Any, List

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


# ── Gather (for tree commit_path) ─────────────────────────────────────────────


def gather_kv_columns(
    past_kv: Any,
    *,
    base_columns: int,
    tail_indices: List[int],
) -> Any:
    """Keep ``past_kv[:, :, :base_columns]`` plus the columns named by
    ``tail_indices`` (offsets RELATIVE to ``base_columns``).

    Used by tree speculative when the target forward produced N
    speculative columns at positions ``[base_columns, base_columns + N)``
    and we need to keep only the ``len(tail_indices)`` columns matching
    the accepted path's tree nodes — in order.

    Result length == ``base_columns + len(tail_indices)``. ``tail_indices``
    may be empty (commit nothing from the speculative tail).

    Returns a value of the same format as the input:
      - tuple-of-tuples → fresh tuple of tensors built via concat.
      - ``DynamicCache`` → mutated in place and returned. Both
        ``key_cache`` and ``value_cache`` are rewritten per layer.

    ``base_columns < 0`` or any tail index out of range raises
    ``ValueError``. ``base_columns == past_kv.seq_len`` with empty
    ``tail_indices`` is equivalent to ``truncate_past_kv(past_kv,
    base_columns)``.
    """
    if past_kv is None:
        if base_columns != 0 or tail_indices:
            raise ValueError(
                f"cannot gather from past_kv=None with base_columns="
                f"{base_columns}, tail_indices={tail_indices}"
            )
        return None
    if base_columns < 0:
        raise ValueError(f"base_columns must be >= 0, got {base_columns}")

    if hasattr(past_kv, "get_seq_length"):
        return _gather_dynamic_cache(past_kv, base_columns, tail_indices)
    return _gather_tuple_kv(past_kv, base_columns, tail_indices)


def _gather_tuple_kv(
    past_kv: tuple, base_columns: int, tail_indices: List[int],
) -> tuple:
    import torch
    current = KVCacheManager.kv_seq_len(past_kv)
    if base_columns > current:
        raise ValueError(
            f"base_columns={base_columns} exceeds seq_len={current}"
        )
    total_tail = current - base_columns
    for idx in tail_indices:
        if idx < 0 or idx >= total_tail:
            raise ValueError(
                f"tail index {idx} out of range [0, {total_tail})"
            )

    # Fast path: no tail to gather → contiguous slice.
    if not tail_indices:
        return KVCacheManager.slice_kv(past_kv, 0, base_columns)

    tail_idx_tensor = torch.tensor(tail_indices, dtype=torch.long)

    out = []
    for k_layer, v_layer in past_kv:
        # k_layer / v_layer shape: (batch, heads, seq, head_dim).
        base_k = k_layer[:, :, :base_columns, :]
        base_v = v_layer[:, :, :base_columns, :]
        tail_k = k_layer.index_select(
            dim=2, index=tail_idx_tensor.to(k_layer.device).add(base_columns),
        )
        tail_v = v_layer.index_select(
            dim=2, index=tail_idx_tensor.to(v_layer.device).add(base_columns),
        )
        out.append((
            torch.cat([base_k, tail_k], dim=2),
            torch.cat([base_v, tail_v], dim=2),
        ))
    return tuple(out)


def _gather_dynamic_cache(
    cache: Any, base_columns: int, tail_indices: List[int],
) -> Any:
    """Same gather semantics on ``DynamicCache``. Mutates in place."""
    import torch
    current = cache.get_seq_length()
    if base_columns > current:
        raise ValueError(
            f"base_columns={base_columns} exceeds seq_len={current}"
        )
    total_tail = current - base_columns
    for idx in tail_indices:
        if idx < 0 or idx >= total_tail:
            raise ValueError(
                f"tail index {idx} out of range [0, {total_tail})"
            )

    if not hasattr(cache, "key_cache") or not hasattr(cache, "value_cache"):
        raise TypeError(
            f"DynamicCache-like object lacks key_cache/value_cache; "
            f"got type {type(cache).__name__}"
        )

    new_len = base_columns + len(tail_indices)
    tail_idx_tensor = torch.tensor(tail_indices, dtype=torch.long)

    for i in range(len(cache.key_cache)):
        k = cache.key_cache[i]
        v = cache.value_cache[i]
        base_k = k[:, :, :base_columns, :]
        base_v = v[:, :, :base_columns, :]
        if tail_indices:
            tail_k = k.index_select(
                dim=2, index=tail_idx_tensor.to(k.device).add(base_columns),
            )
            tail_v = v.index_select(
                dim=2, index=tail_idx_tensor.to(v.device).add(base_columns),
            )
            cache.key_cache[i] = torch.cat([base_k, tail_k], dim=2)
            cache.value_cache[i] = torch.cat([base_v, tail_v], dim=2)
        else:
            cache.key_cache[i] = base_k
            cache.value_cache[i] = base_v
    if hasattr(cache, "_seen_tokens"):
        cache._seen_tokens = new_len
    return cache
