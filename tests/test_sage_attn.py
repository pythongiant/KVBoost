"""Tests for the Triton SageAttention (INT8) + FP16 flash backends.

Two layers:

* CPU-runnable safety contract (always runs, incl. CI / the dev Mac): when the
  fast path is ineligible (no Triton, non-CUDA tensors, decode shape, padding
  mask) the backend MUST delegate to PyTorch SDPA and return byte-for-byte
  identical output. This is what guarantees correctness when the kernel can't
  or shouldn't run.

* Numerical accuracy (CUDA + Triton only): the kernel output — both the FP16
  ``triton_flash`` path and the INT8 ``sage`` path — is compared against an
  SDPA reference (with GQA expansion) across head dims, GQA ratios, sequence
  lengths, dtypes, and causal/non-causal. INT8 uses a looser tolerance since
  it's lossy by design. These run on the GPU box (e.g. the RTX 3060).
"""
import math

import pytest
import torch
import torch.nn.functional as F

from kvboost.kernels.sage_attn import (
    _launch,
    sage_attention_forward,
    triton_available,
    triton_flash_attention_forward,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward

HAS_CUDA = torch.cuda.is_available()
try:
    import triton  # noqa: F401
    HAS_TRITON = True
except Exception:
    HAS_TRITON = False

needs_gpu = pytest.mark.skipif(
    not (HAS_CUDA and HAS_TRITON), reason="needs CUDA + Triton"
)


class _DummyAttn(torch.nn.Module):
    """Minimal stand-in for an HF attention module."""
    def __init__(self, n_kv_groups=1):
        super().__init__()
        self.num_key_value_groups = n_kv_groups
        self.is_causal = True


def _qkv(B, Hq, q_len, kv_len, Hkv, D, dtype=torch.float32, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, Hq, q_len, D, generator=g).to(device=device, dtype=dtype)
    k = torch.randn(B, Hkv, kv_len, D, generator=g).to(device=device, dtype=dtype)
    v = torch.randn(B, Hkv, kv_len, D, generator=g).to(device=device, dtype=dtype)
    return q, k, v


def _sdpa_gqa_ref(q, k, v, causal, scale):
    """Direct SDPA reference with GQA expansion, in fp32 for a clean baseline."""
    groups = q.shape[1] // k.shape[1]
    kk = k.repeat_interleave(groups, dim=1).float()
    vv = v.repeat_interleave(groups, dim=1).float()
    return F.scaled_dot_product_attention(
        q.float(), kk, vv, is_causal=causal, scale=scale
    )  # [B, Hq, S, D]


# ── CPU safety contract (runs everywhere) ────────────────────────────────────

@pytest.mark.parametrize("fwd", [sage_attention_forward, triton_flash_attention_forward])
def test_cpu_prefill_delegates_to_sdpa(fwd):
    # CPU tensors are never eligible → must match SDPA exactly.
    mod = _DummyAttn()
    q, k, v = _qkv(1, 4, 8, 8, 4, 16)
    out, _ = fwd(mod, q, k, v, None, scaling=0.25, is_causal=True)
    ref, _ = sdpa_attention_forward(mod, q, k, v, None, scaling=0.25, is_causal=True)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("fwd", [sage_attention_forward, triton_flash_attention_forward])
def test_padding_mask_delegates_to_sdpa(fwd):
    # A non-None attention_mask (padded batch) must fall back to SDPA.
    mod = _DummyAttn()
    q, k, v = _qkv(1, 4, 8, 8, 4, 16)
    mask = torch.zeros(1, 1, 8, 8)
    out, _ = fwd(mod, q, k, v, mask, scaling=0.25)
    ref, _ = sdpa_attention_forward(mod, q, k, v, mask, scaling=0.25)
    assert torch.equal(out, ref)


def test_resolve_attn_impl_sage_passthrough_or_fallback():
    from kvboost.kernels import resolve_attn_impl
    assert resolve_attn_impl("sdpa") == "sdpa"
    for name in ("sage", "triton_flash"):
        resolved = resolve_attn_impl(name)
        assert resolved == (name if triton_available() else "sdpa")


# ── Numerical accuracy (CUDA + Triton) ───────────────────────────────────────

@needs_gpu
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (8, 2), (16, 4)])
@pytest.mark.parametrize("S", [16, 200, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_triton_flash_matches_sdpa(D, Hq, Hkv, S, dtype, causal):
    scale = 1.0 / math.sqrt(D)
    q, k, v = _qkv(1, Hq, S, S, Hkv, D, dtype=dtype, device="cuda")
    out = _launch(q, k, v, scale, causal, use_int8=False)          # [1,Hq,S,D]
    ref = _sdpa_gqa_ref(q, k, v, causal, scale)
    diff = (out.float() - ref).abs().max().item()
    assert diff < 2e-2, f"FP16 triton flash max|Δ|={diff:.3g}"


@needs_gpu
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("Hq,Hkv", [(8, 8), (8, 2)])
@pytest.mark.parametrize("S", [128, 333])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("causal", [True, False])
def test_sage_int8_matches_sdpa(D, Hq, Hkv, S, dtype, causal):
    scale = 1.0 / math.sqrt(D)
    q, k, v = _qkv(1, Hq, S, S, Hkv, D, dtype=dtype, device="cuda")
    out = _launch(q, k, v, scale, causal, use_int8=True)           # [1,Hq,S,D]
    ref = _sdpa_gqa_ref(q, k, v, causal, scale)
    of, rf = out.float(), ref
    max_diff = (of - rf).abs().max().item()
    cos = F.cosine_similarity(of.flatten(), rf.flatten(), dim=0).item()
    # INT8 attention is lossy; require close-but-not-exact + high cosine sim.
    assert cos > 0.99, f"INT8 sage cosine={cos:.4f} (max|Δ|={max_diff:.3g})"
    assert max_diff < 6e-2, f"INT8 sage max|Δ|={max_diff:.3g}"


@needs_gpu
def test_sage_forward_end_to_end_matches_sdpa():
    # Exercise the registered HF forward fn (quant + smoothing + transpose +
    # self-check) on a CUDA prefill, vs HF's SDPA forward.
    mod = _DummyAttn(n_kv_groups=4)
    scale = 1.0 / math.sqrt(64)
    q, k, v = _qkv(1, 8, 256, 256, 2, 64, dtype=torch.float16, device="cuda")
    out, _ = sage_attention_forward(mod, q, k, v, None, scaling=scale, is_causal=True)
    ref, _ = sdpa_attention_forward(mod, q, k, v, None, scaling=scale, is_causal=True)
    cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0).item()
    assert cos > 0.99, f"sage forward cosine={cos:.4f}"
