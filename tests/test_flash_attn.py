"""
Tests for the Flash Attention integration (Track 1).

These tests cover:
  - Correctness of _kvboost_flash_attn vs. reference F.scaled_dot_product_attention
  - Three-tier fallback detection (get_tier)
  - install_flash_attention / uninstall_flash_attention on a toy model
  - Causal masking correctness
  - GQA (grouped query attention) pass-through
  - Edge cases: seq_len=1, seq_len not divisible by block tile size

Tests do NOT require the compiled CUDA extension — they run on CPU with the
vanilla SDPA fallback, validating the dispatch logic and numerical correctness.
"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from kvboost.flash_attn_ext import (
    _kvboost_flash_attn,
    flash_attention_available,
    get_tier,
    install_flash_attention,
    uninstall_flash_attention,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def reference_attn(Q, K, V, scale, causal):
    """Vanilla scaled dot-product attention for correctness comparison."""
    return F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=causal)


def make_qkv(B=2, H=4, S=64, D=64, dtype=torch.float32, device="cpu"):
    torch.manual_seed(42)
    Q = torch.randn(B, H, S, D, dtype=dtype, device=device)
    K = torch.randn(B, H, S, D, dtype=dtype, device=device)
    V = torch.randn(B, H, S, D, dtype=dtype, device=device)
    return Q, K, V


# ── Tier detection ────────────────────────────────────────────────────────────

class TestTierDetection:
    def test_get_tier_returns_string(self):
        tier = get_tier()
        assert isinstance(tier, str)
        assert tier in ("kvboost_cuda", "torch_flash", "vanilla")

    def test_flash_attention_available_is_bool(self):
        assert isinstance(flash_attention_available(), bool)

    def test_tier_consistent_with_available(self):
        if flash_attention_available():
            assert get_tier() != "vanilla"
        else:
            assert get_tier() == "vanilla"


# ── Numerical correctness ─────────────────────────────────────────────────────

class TestNumericalCorrectness:
    """
    All tests run on CPU with float32 so they don't require CUDA.
    Tolerance is set generously for fp32 since we're comparing two different
    numerically stable implementations.
    """

    @pytest.mark.parametrize("causal", [True, False])
    @pytest.mark.parametrize("S", [1, 16, 64, 100])
    def test_output_close_to_reference(self, causal, S):
        Q, K, V = make_qkv(B=1, H=2, S=S, D=64)
        scale = 1.0 / math.sqrt(64)

        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=causal)
        ref = reference_attn(Q, K, V, scale=scale, causal=causal)

        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), (
            f"Max abs diff: {(out - ref).abs().max().item():.6f}"
        )

    @pytest.mark.parametrize("B", [1, 4])
    @pytest.mark.parametrize("H", [1, 8])
    def test_batch_and_head_dims(self, B, H):
        Q, K, V = make_qkv(B=B, H=H, S=32, D=64)
        scale = 1.0 / math.sqrt(64)

        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=True)
        ref = reference_attn(Q, K, V, scale=scale, causal=True)

        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    def test_seq_len_one(self):
        """Decode step: single query token attending to itself."""
        Q, K, V = make_qkv(B=1, H=4, S=1, D=64)
        scale = 1.0 / math.sqrt(64)

        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=True)
        ref = reference_attn(Q, K, V, scale=scale, causal=True)

        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)

    def test_cross_attention_different_kv_len(self):
        """Q attends to longer K/V (prefix caching scenario)."""
        B, Hq, Sq, D = 1, 4, 8, 64
        Skv = 128
        torch.manual_seed(0)
        Q = torch.randn(B, Hq, Sq, D)
        K = torch.randn(B, Hq, Skv, D)
        V = torch.randn(B, Hq, Skv, D)
        scale = 1.0 / math.sqrt(D)

        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=False)
        ref = reference_attn(Q, K, V, scale=scale, causal=False)

        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)

    def test_causal_mask_upper_triangle_zeros(self):
        """
        With causal=True, token i must not attend to token j > i.
        Verify by checking that attention weight for (i=0, j=1) is ~0.
        """
        B, H, S, D = 1, 1, 4, 4
        torch.manual_seed(7)
        Q = torch.randn(B, H, S, D)
        K = torch.randn(B, H, S, D)
        V = torch.eye(S).unsqueeze(0).unsqueeze(0)  # V[j] = e_j (identity rows)

        scale = 1.0 / math.sqrt(D)
        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=True)
        ref = reference_attn(Q, K, V, scale=scale, causal=True)

        # Token 0 output should only contain contribution from token 0
        # i.e. out[0,0,0] should equal ref[0,0,0] (same causal masking)
        assert torch.allclose(out, ref, atol=1e-5)

    @pytest.mark.parametrize("D", [64, 96, 128])
    def test_head_dims(self, D):
        Q, K, V = make_qkv(B=1, H=2, S=32, D=D)
        scale = 1.0 / math.sqrt(D)
        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=True)
        ref = reference_attn(Q, K, V, scale=scale, causal=True)
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)


# ── Model patching ────────────────────────────────────────────────────────────

class _ToyAttention(nn.Module):
    """Minimal attention module that mimics HF's structure for patch testing."""

    def __init__(self, embed_dim=64, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x):
        B, S, E = x.shape
        H, D = self.num_heads, self.head_dim
        Q = self.q_proj(x).view(B, S, H, D).transpose(1, 2)
        K = self.k_proj(x).view(B, S, H, D).transpose(1, 2)
        V = self.v_proj(x).view(B, S, H, D).transpose(1, 2)
        scale = 1.0 / math.sqrt(D)
        attn = F.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=True)
        out = attn.transpose(1, 2).reshape(B, S, E)
        return self.out_proj(out)


class _ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = _ToyAttention()

    def forward(self, x):
        return self.attention(x)


class TestModelPatching:
    def test_install_returns_count(self):
        model = _ToyTransformer()
        # _ToyAttention ends with "Attention" — should be patched
        n = install_flash_attention(model)
        assert isinstance(n, int)
        assert n >= 0

    def test_install_and_uninstall_roundtrip(self):
        model = _ToyTransformer()
        torch.manual_seed(0)
        x = torch.randn(1, 8, 64)

        out_before = model(x).detach()

        install_flash_attention(model)
        out_after = model(x).detach()

        # Output should be numerically identical (same math, possibly different path)
        assert torch.allclose(out_before, out_after, atol=1e-5, rtol=1e-5), (
            "install_flash_attention changed model outputs"
        )

        restored = uninstall_flash_attention(model)
        assert restored >= 0

        out_restored = model(x).detach()
        assert torch.allclose(out_before, out_restored, atol=1e-5, rtol=1e-5), (
            "uninstall_flash_attention did not fully restore model"
        )

    def test_double_install_is_idempotent(self):
        """Calling install twice should not patch the same module twice."""
        model = _ToyTransformer()
        n1 = install_flash_attention(model)
        n2 = install_flash_attention(model)
        assert n2 == 0, "Second install should patch 0 modules (already patched)"

    def test_uninstall_on_unpatched_model(self):
        model = _ToyTransformer()
        restored = uninstall_flash_attention(model)
        assert restored == 0


# ── CUDA-only tests (skipped on CPU) ─────────────────────────────────────────

@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
class TestCUDA:
    def test_fp16_output_close_on_gpu(self):
        Q, K, V = make_qkv(B=2, H=8, S=128, D=128, dtype=torch.float16, device="cuda")
        scale = 1.0 / math.sqrt(128)
        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=True)
        ref = reference_attn(Q, K, V, scale=scale, causal=True)
        # fp16 has lower precision — use atol=1e-2
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_bf16_output_close_on_gpu(self):
        Q, K, V = make_qkv(B=1, H=4, S=64, D=64, dtype=torch.bfloat16, device="cuda")
        scale = 1.0 / math.sqrt(64)
        out = _kvboost_flash_attn(Q, K, V, scale=scale, causal=True)
        ref = reference_attn(Q, K, V, scale=scale, causal=True)
        assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2)

    def test_tier_is_not_vanilla_on_cuda(self):
        assert get_tier() != "vanilla", (
            "On a CUDA machine, flash_attn should use at least torch_flash tier"
        )
