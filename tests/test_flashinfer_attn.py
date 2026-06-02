"""CPU-runnable tests for the FlashInfer attention integration.

We can't exercise the FlashInfer CUDA kernel here (no GPU / package), but we
CAN verify the safety contract that protects correctness when it's absent or
ineligible: the function must delegate to PyTorch SDPA and produce byte-for-
byte identical output. The actual kernel + numerical self-check are validated
on the GPU box.
"""
import torch

from kvboost.kernels.flashinfer_attn import (
    flashinfer_attention_forward,
    flashinfer_available,
    resolve_attn_impl,
)
from transformers.integrations.sdpa_attention import sdpa_attention_forward


class _DummyAttn(torch.nn.Module):
    """Minimal stand-in for an HF attention module (what the attn fn reads)."""
    def __init__(self, n_kv_groups=1):
        super().__init__()
        self.num_key_value_groups = n_kv_groups
        self.is_causal = True


def _qkv(B, Hq, q_len, kv_len, Hkv, D):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, Hq, q_len, D, generator=g)
    k = torch.randn(B, Hkv, kv_len, D, generator=g)
    v = torch.randn(B, Hkv, kv_len, D, generator=g)
    return q, k, v


def test_flashinfer_absent_delegates_to_sdpa_prefill():
    # Prefill shape (q_len>1) must always delegate to SDPA regardless.
    mod = _DummyAttn()
    q, k, v = _qkv(1, 4, 8, 8, 4, 16)
    out, _ = flashinfer_attention_forward(mod, q, k, v, None, scaling=0.25)
    ref, _ = sdpa_attention_forward(mod, q, k, v, None, scaling=0.25)
    assert torch.equal(out, ref)


def test_flashinfer_cpu_decode_falls_back_to_sdpa():
    # Decode shape but CPU (not CUDA) → must delegate, identical output.
    mod = _DummyAttn()
    q, k, v = _qkv(1, 4, 1, 8, 4, 16)
    out, _ = flashinfer_attention_forward(mod, q, k, v, None, scaling=0.25)
    ref, _ = sdpa_attention_forward(mod, q, k, v, None, scaling=0.25)
    assert torch.equal(out, ref)


def test_flashinfer_gqa_fallback_matches_sdpa():
    # GQA (Hkv < Hq) on the fallback path must still match SDPA.
    mod = _DummyAttn(n_kv_groups=2)
    q, k, v = _qkv(1, 4, 1, 8, 2, 16)
    out, _ = flashinfer_attention_forward(mod, q, k, v, None, scaling=0.25)
    ref, _ = sdpa_attention_forward(mod, q, k, v, None, scaling=0.25)
    assert torch.equal(out, ref)


def test_resolve_attn_impl_passthrough_and_fallback():
    assert resolve_attn_impl("sdpa") == "sdpa"
    assert resolve_attn_impl("flash_attention_2") == "flash_attention_2"
    # flashinfer resolves to itself if available, else sdpa — never crashes.
    resolved = resolve_attn_impl("flashinfer")
    if flashinfer_available():
        assert resolved == "flashinfer"
    else:
        assert resolved == "sdpa"
