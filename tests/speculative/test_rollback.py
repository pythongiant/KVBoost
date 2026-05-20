"""Unit tests for kvboost.speculative.rollback.

Covers both KV formats: legacy tuple-of-tuples and HF DynamicCache.
"""

from __future__ import annotations

import pytest
import torch

from kvboost.cache_manager import KVCacheManager
from kvboost.speculative.rollback import truncate_past_kv


def _make_tuple_kv(seq_len: int, num_layers: int = 4, heads: int = 2, dim: int = 8):
    return tuple(
        (
            torch.randn(1, heads, seq_len, dim),
            torch.randn(1, heads, seq_len, dim),
        )
        for _ in range(num_layers)
    )


# ── tuple-of-tuples ──────────────────────────────────────────────────────────


def test_truncate_tuple_kv_basic():
    kv = _make_tuple_kv(seq_len=10)
    assert KVCacheManager.kv_seq_len(kv) == 10
    out = truncate_past_kv(kv, keep_n=7)
    assert KVCacheManager.kv_seq_len(out) == 7
    # Per-layer shapes preserved on non-seq dims.
    for layer in out:
        assert layer[0].shape == (1, 2, 7, 8)
        assert layer[1].shape == (1, 2, 7, 8)


def test_truncate_tuple_kv_preserves_values():
    """Truncated KV should be a prefix of the original — exact element
    equality on the kept positions."""
    kv = _make_tuple_kv(seq_len=10)
    out = truncate_past_kv(kv, keep_n=4)
    for orig, trunc in zip(kv, out):
        assert torch.allclose(orig[0][:, :, :4, :], trunc[0])
        assert torch.allclose(orig[1][:, :, :4, :], trunc[1])


def test_truncate_tuple_kv_noop_at_full_length():
    kv = _make_tuple_kv(seq_len=5)
    out = truncate_past_kv(kv, keep_n=5)
    # No-op may return the same object; verify by content.
    for orig, trunc in zip(kv, out):
        assert torch.equal(orig[0], trunc[0])


def test_truncate_tuple_kv_overflow_raises():
    kv = _make_tuple_kv(seq_len=5)
    with pytest.raises(ValueError, match="exceeds current seq_len"):
        truncate_past_kv(kv, keep_n=10)


def test_truncate_negative_keep_n_raises():
    kv = _make_tuple_kv(seq_len=5)
    with pytest.raises(ValueError, match="keep_n must be >= 0"):
        truncate_past_kv(kv, keep_n=-1)


def test_truncate_none_with_zero_keep_returns_none():
    assert truncate_past_kv(None, keep_n=0) is None


def test_truncate_none_with_nonzero_keep_raises():
    with pytest.raises(ValueError, match="past_kv is None"):
        truncate_past_kv(None, keep_n=3)


# ── DynamicCache ─────────────────────────────────────────────────────────────


def _try_import_dynamic_cache():
    try:
        from transformers import DynamicCache  # noqa: WPS433
        return DynamicCache
    except Exception:
        return None


def test_truncate_dynamic_cache_basic():
    DynamicCache = _try_import_dynamic_cache()
    if DynamicCache is None:
        pytest.skip("transformers DynamicCache not importable")

    kv = _make_tuple_kv(seq_len=10)
    cache = DynamicCache()
    for layer_k, layer_v in kv:
        cache.update(layer_k, layer_v, len(cache))

    assert cache.get_seq_length() == 10
    out = truncate_past_kv(cache, keep_n=4)
    # Same object returned, mutated in place.
    assert out is cache
    assert cache.get_seq_length() == 4


def _extract_layer_kv(cache, layer_idx: int):
    """Pull (keys, values) tensors from a DynamicCache across transformers
    versions. transformers 5.x exposes per-layer ``.layers[i].keys`` /
    ``.values``; older versions use ``.key_cache[i]`` / ``.value_cache[i]``
    or expose ``.to_legacy_cache()``."""
    if hasattr(cache, "layers") and len(cache.layers) > layer_idx:
        layer = cache.layers[layer_idx]
        if hasattr(layer, "keys") and hasattr(layer, "values"):
            return layer.keys, layer.values
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    if hasattr(cache, "to_legacy_cache"):
        legacy = cache.to_legacy_cache()
        return legacy[layer_idx][0], legacy[layer_idx][1]
    pytest.skip(
        f"unsupported DynamicCache layout: {type(cache).__name__} "
        f"has none of (.layers[].keys, .key_cache, .to_legacy_cache)"
    )


def test_truncate_dynamic_cache_preserves_values():
    DynamicCache = _try_import_dynamic_cache()
    if DynamicCache is None:
        pytest.skip("transformers DynamicCache not importable")

    kv = _make_tuple_kv(seq_len=8)
    cache = DynamicCache()
    for layer_k, layer_v in kv:
        cache.update(layer_k, layer_v, len(cache))

    truncate_past_kv(cache, keep_n=3)
    for i, (orig_k, orig_v) in enumerate(kv):
        keys, values = _extract_layer_kv(cache, i)
        assert torch.allclose(keys, orig_k[:, :, :3, :])
        assert torch.allclose(values, orig_v[:, :, :3, :])


def test_truncate_dynamic_cache_overflow_raises():
    DynamicCache = _try_import_dynamic_cache()
    if DynamicCache is None:
        pytest.skip("transformers DynamicCache not importable")

    kv = _make_tuple_kv(seq_len=5)
    cache = DynamicCache()
    for layer_k, layer_v in kv:
        cache.update(layer_k, layer_v, len(cache))

    with pytest.raises(ValueError, match="exceeds current seq_len"):
        truncate_past_kv(cache, keep_n=10)
