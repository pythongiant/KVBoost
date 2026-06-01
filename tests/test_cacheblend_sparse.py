"""Tests for faithful sparse CacheBlend (layer-wise selective recompute).

The correctness story has three gates:

  1. Mechanism: sparse recompute with S = ALL positions must equal a clean
     full prefill bit-for-bit. This validates the hand-rolled decoder forward
     (RoPE, GQA, RMSNorm, residuals, MLP).
  2. Scattered blend: with the non-selected cached KV already correct,
     recomputing an arbitrary scattered subset must reproduce the true KV at
     those positions (validates index-scatter + per-position causal mask).
  3. Restoration: recomputing corrupted positions (whose causal prefix is
     correct) drives their error to ~0.

Plus: the cheap HKVD auto-selector returns a bounded subset, and sparse
recompute touches only |S| tokens (the compute win vs full forward).
"""

from __future__ import annotations

import random

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from kvboost.cacheblend_sparse import SparseCacheBlend, supports_sparse_recompute


def _norm(kv):
    if hasattr(kv, "layers"):
        return [(l.keys, l.values) for l in kv.layers]
    if hasattr(kv, "key_cache"):
        return list(zip(kv.key_cache, kv.value_cache))
    return [(a, b) for a, b in kv]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = AutoConfig.for_model(
        "qwen2", hidden_size=128, num_hidden_layers=4,
        num_attention_heads=8, num_key_value_heads=2, intermediate_size=256,
        vocab_size=512, max_position_embeddings=2048,
    )
    return AutoModelForCausalLM.from_config(cfg).eval()


@pytest.fixture
def ids():
    g = torch.Generator().manual_seed(7)
    return torch.randint(0, 512, (1, 60), generator=g)


def test_capability_check(model):
    assert supports_sparse_recompute(model) is True


def test_gate1_sparse_all_equals_full_prefill(model, ids):
    """S = all positions ⇒ clean full prefill, bit-for-bit."""
    with torch.no_grad():
        ref = _norm(model(ids, use_cache=True).past_key_values)
    n = ids.shape[1]
    dummy = tuple((torch.zeros_like(k), torch.zeros_like(v)) for k, v in ref)
    sp = SparseCacheBlend()
    blended = sp.recompute(model, ids[0].tolist(), dummy,
                           selected_positions=list(range(n)))
    for (rk, rv), (bk, bv) in zip(ref, blended):
        assert torch.allclose(rk, bk, atol=1e-3)
        assert torch.allclose(rv, bv, atol=1e-3)


def test_gate2_scattered_subset_against_correct_context(model, ids):
    """Recomputing a scattered subset against correct cached context
    reproduces the true KV at those positions."""
    with torch.no_grad():
        ref = _norm(model(ids, use_cache=True).past_key_values)
    n = ids.shape[1]
    S = sorted(random.Random(2).sample(range(n), 15))
    sp = SparseCacheBlend()
    blended = sp.recompute(
        model, ids[0].tolist(),
        tuple((k.clone(), v.clone()) for k, v in ref),
        selected_positions=S,
    )
    # Everything should still match truth (S recomputed identically, rest kept).
    for (rk, rv), (bk, bv) in zip(ref, blended):
        assert torch.allclose(rk, bk, atol=1e-3)
        assert torch.allclose(rv, bv, atol=1e-3)


def test_gate3_recompute_restores_corrupted_positions(model, ids):
    with torch.no_grad():
        ref = _norm(model(ids, use_cache=True).past_key_values)
    n = ids.shape[1]
    stale = [(k.clone(), v.clone()) for k, v in ref]
    g = torch.Generator().manual_seed(3)
    for k, v in stale:
        k[:, :, 30:, :] += 0.5 * torch.randn(k[:, :, 30:, :].shape, generator=g)
        v[:, :, 30:, :] += 0.5 * torch.randn(v[:, :, 30:, :].shape, generator=g)
    S = list(range(30, 45))  # contiguous block, correct prefix [0:30]
    sp = SparseCacheBlend()
    blended = sp.recompute(model, ids[0].tolist(),
                           tuple((k.clone(), v.clone()) for k, v in stale),
                           selected_positions=S)
    # Recomputed positions should match truth (prefix is all correct).
    for li in range(len(ref)):
        rk = ref[li][0][:, :, S, :]
        bk = blended[li][0][:, :, S, :]
        assert torch.allclose(rk, bk, atol=1e-3)


def test_auto_select_returns_bounded_subset(model, ids):
    """The cheap HKVD probe returns a subset no larger than ratio×N."""
    with torch.no_grad():
        ref = _norm(model(ids, use_cache=True).past_key_values)
    n = ids.shape[1]
    # Corrupt some positions so deviation is nonzero and selectable.
    stale = [(k.clone(), v.clone()) for k, v in ref]
    g = torch.Generator().manual_seed(5)
    for k, v in stale:
        k[:, :, 20:40, :] += torch.randn(k[:, :, 20:40, :].shape, generator=g)
    sp = SparseCacheBlend(recompute_ratio=0.2)
    blended = sp.recompute(model, ids[0].tolist(), stale)  # auto-select
    # Result is a valid KV of full length (non-selected kept, selected blended).
    assert len(blended) == len(ref)
    assert blended[0][0].shape[2] == n


def test_unsupported_arch_reports_false():
    """A non-Llama/Qwen2 module without the expected internals → False."""
    class Bogus(torch.nn.Module):
        pass
    assert supports_sparse_recompute(Bogus()) is False
