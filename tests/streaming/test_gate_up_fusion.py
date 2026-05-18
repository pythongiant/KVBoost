"""Tests for the SwiGLU gate+up fusion (concat helper + module forward).

CUDA paths are skipped on machines without a GPU; the concat/spec
helpers are pure-torch and exercise on CPU.
"""

from __future__ import annotations

import torch

from kvboost.streaming.awq_loader import (
    LayerSpec,
    TensorSpec,
    fuse_gate_up_layer_spec,
    fuse_gate_up_tensors,
)
from kvboost.streaming.qkv_proj import StreamingQLinear, StreamingQLinearGateUp


# ── helpers ─────────────────────────────────────────────────────────────────


def _awq_layer(in_features: int, out_features: int, group_size: int):
    """Random valid AWQ tensors at the given shape — for arithmetic
    smoke tests, NOT bit-exact parity (we'd need real autoawq packing).
    """
    pack = 8
    torch.manual_seed(0)
    qweight = torch.randint(
        low=0, high=2**31 - 1,
        size=(in_features, out_features // pack), dtype=torch.int32,
    )
    scales = torch.randn(in_features // group_size, out_features, dtype=torch.float16) * 0.01
    qzeros = torch.randint(
        low=0, high=2**31 - 1,
        size=(in_features // group_size, out_features // pack), dtype=torch.int32,
    )
    return qweight, scales, qzeros


# ── fuse_gate_up_tensors ────────────────────────────────────────────────────


def test_fuse_gate_up_tensors_concats_along_last_dim():
    gate_qw, gate_sc, gate_qz = _awq_layer(128, 256, 32)
    up_qw, up_sc, up_qz = _awq_layer(128, 192, 32)

    tensors = {
        "mlp.gate_proj.qweight": gate_qw,
        "mlp.gate_proj.scales": gate_sc,
        "mlp.gate_proj.qzeros": gate_qz,
        "mlp.up_proj.qweight": up_qw,
        "mlp.up_proj.scales": up_sc,
        "mlp.up_proj.qzeros": up_qz,
        "self_attn.q_proj.qweight": torch.zeros(3),  # passthrough
    }
    fused = fuse_gate_up_tensors(tensors)

    # Concat dimensions
    assert fused["mlp.gate_up_proj.qweight"].shape == (128, (256 + 192) // 8)
    assert fused["mlp.gate_up_proj.scales"].shape == (128 // 32, 256 + 192)
    assert fused["mlp.gate_up_proj.qzeros"].shape == (128 // 32, (256 + 192) // 8)

    # The first half of fused.scales must equal gate.scales (cat order).
    assert torch.equal(fused["mlp.gate_up_proj.scales"][:, :256], gate_sc)
    assert torch.equal(fused["mlp.gate_up_proj.scales"][:, 256:], up_sc)

    # Originals gone, unrelated passthrough preserved.
    assert "mlp.gate_proj.qweight" not in fused
    assert "mlp.up_proj.qweight" not in fused
    assert "self_attn.q_proj.qweight" in fused


def test_fuse_gate_up_tensors_passthrough_when_no_swiglu():
    """A layer with no gate/up (pure attention) is unchanged."""
    tensors = {
        "self_attn.q_proj.qweight": torch.zeros(3),
        "self_attn.k_proj.qweight": torch.zeros(3),
    }
    out = fuse_gate_up_tensors(tensors)
    assert out == tensors


# ── fuse_gate_up_layer_spec ─────────────────────────────────────────────────


def _spec(name: str, shape: tuple[int, ...], dtype=torch.int32) -> TensorSpec:
    from pathlib import Path
    nbytes = 1
    for d in shape:
        nbytes *= d
    nbytes *= torch.tensor([], dtype=dtype).element_size()
    return TensorSpec(
        name=name,
        path=Path("/dev/null"),
        shape=shape,
        dtype=dtype,
        layer_idx=0,
        is_quantized=True,
        is_resident=False,
        nbytes=nbytes,
    )


def test_fuse_gate_up_layer_spec_merges_sizes_and_bytes():
    layer = LayerSpec(
        layer_idx=0,
        tensors={
            "mlp.gate_proj.qweight": _spec("mlp.gate_proj.qweight", (128, 32)),
            "mlp.gate_proj.scales": _spec("mlp.gate_proj.scales", (4, 256), torch.float16),
            "mlp.gate_proj.qzeros": _spec("mlp.gate_proj.qzeros", (4, 32)),
            "mlp.up_proj.qweight": _spec("mlp.up_proj.qweight", (128, 24)),
            "mlp.up_proj.scales": _spec("mlp.up_proj.scales", (4, 192), torch.float16),
            "mlp.up_proj.qzeros": _spec("mlp.up_proj.qzeros", (4, 24)),
            "self_attn.q_proj.qweight": _spec("self_attn.q_proj.qweight", (128, 64)),
        },
        resident=False,
    )
    fused = fuse_gate_up_layer_spec(layer)

    qw = fused.tensors["mlp.gate_up_proj.qweight"]
    assert qw.shape == (128, 32 + 24)
    assert qw.nbytes == (128 * 32 + 128 * 24) * 4  # int32

    sc = fused.tensors["mlp.gate_up_proj.scales"]
    assert sc.shape == (4, 256 + 192)
    assert sc.dtype == torch.float16

    # Pure-attention spec untouched
    assert "self_attn.q_proj.qweight" in fused.tensors
    # Originals removed
    assert "mlp.gate_proj.qweight" not in fused.tensors
    assert "mlp.up_proj.qweight" not in fused.tensors
    # Top-level layer attrs preserved
    assert fused.layer_idx == 0
    assert fused.resident is False


def test_fuse_gate_up_layer_spec_no_swiglu_passthrough():
    layer = LayerSpec(
        layer_idx=0,
        tensors={"self_attn.q_proj.qweight": _spec("self_attn.q_proj.qweight", (4, 8))},
        resident=False,
    )
    assert fuse_gate_up_layer_spec(layer) is layer


# ── StreamingQLinearGateUp ──────────────────────────────────────────────────


def test_streaming_qlinear_gate_up_forward_splits_correctly():
    """forward_silu_mul(x) must equal silu(gate(x)) * up(x) where
    (gate, up) come from splitting the fused matmul's output.
    Uses prefer='torch' so the result is deterministic on CPU.
    """
    in_features, gate_out, up_out, group_size = 128, 256, 256, 32

    gate_qw, gate_sc, gate_qz = _awq_layer(in_features, gate_out, group_size)
    up_qw, up_sc, up_qz = _awq_layer(in_features, up_out, group_size)

    # Reference: two separate StreamingQLinears
    gate = StreamingQLinear(
        in_features=in_features, out_features=gate_out,
        group_size=group_size, prefer="torch",
    )
    gate.rebind(qweight=gate_qw, scales=gate_sc, qzeros=gate_qz)
    up = StreamingQLinear(
        in_features=in_features, out_features=up_out,
        group_size=group_size, prefer="torch",
    )
    up.rebind(qweight=up_qw, scales=up_sc, qzeros=up_qz)

    # Fused: one StreamingQLinearGateUp with concat'd tensors
    fused = StreamingQLinearGateUp(
        in_features=in_features, gate_out=gate_out, up_out=up_out,
        group_size=group_size, prefer="torch",
    )
    fused.rebind(
        qweight=torch.cat([gate_qw, up_qw], dim=-1),
        scales=torch.cat([gate_sc, up_sc], dim=-1),
        qzeros=torch.cat([gate_qz, up_qz], dim=-1),
    )

    x = torch.randn(2, 5, in_features, dtype=torch.float16)
    expected = torch.nn.functional.silu(gate(x)) * up(x)
    actual = fused.forward_silu_mul(x)

    assert actual.shape == expected.shape
    # fp16 accumulation noise — loose atol on these scales.
    assert torch.allclose(actual, expected, atol=1e-2, rtol=1e-2)


def test_streaming_qlinear_gate_up_forward_returns_concat():
    """forward (unsplit) should be a concat — useful for parity work."""
    in_features, gate_out, up_out, group_size = 64, 128, 128, 32
    gate_qw, gate_sc, gate_qz = _awq_layer(in_features, gate_out, group_size)
    up_qw, up_sc, up_qz = _awq_layer(in_features, up_out, group_size)

    fused = StreamingQLinearGateUp(
        in_features=in_features, gate_out=gate_out, up_out=up_out,
        group_size=group_size, prefer="torch",
    )
    fused.rebind(
        qweight=torch.cat([gate_qw, up_qw], dim=-1),
        scales=torch.cat([gate_sc, up_sc], dim=-1),
        qzeros=torch.cat([gate_qz, up_qz], dim=-1),
    )

    x = torch.randn(3, in_features, dtype=torch.float16)
    out = fused(x)
    assert out.shape == (3, gate_out + up_out)
