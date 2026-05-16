"""Tests for the streamed-layer linear replacement + hook rebind plumbing.

These run offline by faking the autoawq-style quant linear: any nn.Module
with ``qweight`` / ``scales`` / ``qzeros`` attributes counts.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from kvboost.streaming.model_shell import (
    _detect_in_out,
    _is_quant_linear,
    _iter_quant_linears,
    _make_pre_hook,
    _make_post_hook,
    _replace_streamed_linears,
    _set_submodule,
)
from kvboost.streaming.qkv_proj import StreamingQLinear


class _FakeAwqLinear(nn.Module):
    """Stand-in for autoawq's WQLinear_GEMM — just owns the tensor attrs."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        pack = 8
        self.qweight = torch.zeros(in_features, out_features // pack, dtype=torch.int32)
        self.scales = torch.zeros(in_features // 32, out_features, dtype=torch.float16)
        self.qzeros = torch.zeros(in_features // 32, out_features // pack, dtype=torch.int32)


class _FakeAttn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _FakeAwqLinear(128, 128)
        self.k_proj = _FakeAwqLinear(128, 64)
        self.v_proj = _FakeAwqLinear(128, 64)
        self.o_proj = _FakeAwqLinear(128, 128)


class _FakeMlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = _FakeAwqLinear(128, 256)
        self.up_proj = _FakeAwqLinear(128, 256)
        self.down_proj = _FakeAwqLinear(256, 128)


class _FakeDecoderLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(128)
        self.post_attention_layernorm = nn.LayerNorm(128)
        self.self_attn = _FakeAttn()
        self.mlp = _FakeMlp()


class _FakeInnerModel(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 128)
        self.layers = nn.ModuleList(_FakeDecoderLayer() for _ in range(num_layers))
        self.norm = nn.LayerNorm(128)


class _FakeHfModel(nn.Module):
    def __init__(self, num_layers: int) -> None:
        super().__init__()
        from types import SimpleNamespace
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.model = _FakeInnerModel(num_layers)
        self.lm_head = nn.Linear(128, 32, bias=False)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_is_quant_linear_duck_types():
    assert _is_quant_linear(_FakeAwqLinear(64, 64))
    assert not _is_quant_linear(nn.Linear(64, 64))


def test_iter_quant_linears_finds_all_seven():
    layer = _FakeDecoderLayer()
    found = _iter_quant_linears(layer)
    paths = sorted(p for p, _ in found)
    assert paths == sorted([
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ])


def test_detect_in_out():
    assert _detect_in_out(_FakeAwqLinear(128, 256)) == (128, 256)
    # If in_features attr is missing, derive from qweight shape.
    fake = _FakeAwqLinear(128, 256)
    delattr(fake, "in_features")
    delattr(fake, "out_features")
    assert _detect_in_out(fake) == (128, 256)


def test_set_submodule_replaces_at_path():
    layer = _FakeDecoderLayer()
    sq = StreamingQLinear(128, 64, group_size=32, prefer="torch")
    _set_submodule(layer, "self_attn.k_proj", sq)
    assert layer.self_attn.k_proj is sq


def test_replace_streamed_linears_swaps_all_projections():
    model = _FakeHfModel(num_layers=4)
    replacements = _replace_streamed_linears(
        model,
        layer_indices={1, 2},
        group_size=32,
        prefer="torch",
    )
    # Layers 1, 2 swapped; 0, 3 untouched.
    assert set(replacements.keys()) == {1, 2}
    for layer_idx in (1, 2):
        assert isinstance(model.model.layers[layer_idx].self_attn.q_proj, StreamingQLinear)
        assert len(replacements[layer_idx]) == 7
    for layer_idx in (0, 3):
        assert isinstance(model.model.layers[layer_idx].self_attn.q_proj, _FakeAwqLinear)


def test_pre_hook_rebinds_streaming_qlinears():
    """Pre-hook must populate qweight/scales/qzeros on each StreamingQLinear
    using slot views keyed by *layer-relative* paths (the layout strips the
    ``model.layers.{i}.`` prefix so layer schemas match across the streamed
    sequence — see ``_build_scheduler`` in model_shell.py).
    """
    qlin = StreamingQLinear(128, 64, group_size=32, prefer="torch")
    qlinears = {"self_attn.k_proj": qlin}

    class _FakeScheduler:
        def __init__(self) -> None:
            self.before_calls = 0
            self.after_calls = 0
            pack = 8
            self._views = {
                "self_attn.k_proj.qweight": torch.zeros(128, 64 // pack, dtype=torch.int32),
                "self_attn.k_proj.scales": torch.zeros(128 // 32, 64, dtype=torch.float16),
                "self_attn.k_proj.qzeros": torch.zeros(128 // 32, 64 // pack, dtype=torch.int32),
            }

        def before_layer(self, _idx: int):
            self.before_calls += 1
            return self._views

        def after_layer(self, _idx: int) -> None:
            self.after_calls += 1

    sched = _FakeScheduler()
    pre = _make_pre_hook(sched, layer_idx=7, qlinears=qlinears)
    post = _make_post_hook(sched, layer_idx=7)

    assert not qlin.is_bound
    pre(_module=nn.Identity(), _inputs=())
    assert qlin.is_bound
    assert sched.before_calls == 1

    post(_module=nn.Identity(), _inputs=(), _output=None)
    assert sched.after_calls == 1


def test_pre_hook_resident_layer_is_noop():
    """When the scheduler reports the layer as resident (returns None),
    the pre-hook must not attempt to rebind.
    """
    qlin = StreamingQLinear(128, 64, group_size=32, prefer="torch")
    qlinears = {"self_attn.k_proj": qlin}

    class _ResidentScheduler:
        def before_layer(self, _idx: int):
            return None

        def after_layer(self, _idx: int) -> None:
            pass

    pre = _make_pre_hook(_ResidentScheduler(), layer_idx=0, qlinears=qlinears)
    pre(_module=nn.Identity(), _inputs=())
    assert not qlin.is_bound


def test_strip_quantization_config_survives_to_dict():
    """Regression: setting ``cfg.quantization_config = None`` leaves the
    key present in ``__dict__`` and :meth:`PretrainedConfig.to_dict` then
    blindly calls ``.to_dict()`` on the None value — crashing during
    ``AutoModelForCausalLM.from_config``. The strip helper must remove the
    attribute entirely so the stripped config round-trips cleanly.
    """
    pytest.importorskip("transformers")
    from transformers import AutoConfig

    from kvboost.streaming.model_shell import _strip_quantization_config

    # Build a minimal config with an AWQ quantization_config dict attached
    # so the strip has something real to remove. Dimensions must satisfy
    # ``hidden_size % num_attention_heads == 0`` for HF's validator.
    cfg = AutoConfig.for_model(
        "llama",
        num_hidden_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
    )
    cfg.quantization_config = {"quant_method": "awq", "bits": 4, "group_size": 8}

    # Sanity: pre-strip, the dict-form has the quant key.
    pre = cfg.to_dict()
    assert "quantization_config" in pre

    stripped = _strip_quantization_config(cfg)
    assert "quantization_config" not in stripped.__dict__

    # The critical assertion: to_dict must not crash on the stripped config.
    # This is what AutoModelForCausalLM.from_config calls indirectly via
    # GenerationConfig.from_model_config.
    post = stripped.to_dict()
    assert "quantization_config" not in post


def test_scheduler_primes_on_inner_forward_not_just_wrapper_forward(tmp_path):
    """Regression: ``model.generate`` calls ``hf_model.generate`` which
    calls ``hf_model(...)`` internally — bypassing the wrapper's
    ``forward``. The scheduler must still be primed because the priming
    hook is attached to ``hf_model`` itself, not to the wrapper.
    """
    pytest.importorskip("accelerate")

    class _RecordingScheduler:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.begin_calls = 0
            self.before_calls: list[int] = []
            self.after_calls: list[int] = []

        def begin_forward(self) -> None:
            self.begin_calls += 1

        def before_layer(self, idx: int):
            self.before_calls.append(idx)
            return None  # treat as resident — no rebind

        def after_layer(self, idx: int) -> None:
            self.after_calls.append(idx)

    from kvboost.streaming.config import StreamingConfig
    from kvboost.streaming.model_shell import StreamingCausalLM

    sched = _RecordingScheduler()
    hf_model = _FakeHfModel(num_layers=2)
    wrapper = StreamingCausalLM(
        hf_model=hf_model,
        streaming_config=StreamingConfig(residency_mode="partial_resident"),
        loader=None,
        scheduler=sched,  # type: ignore[arg-type]
        streamed_qlinears={},  # no per-layer hooks — only the model-level priming hook
    )

    # Simulate what hf_model.generate does internally: call hf_model
    # directly (bypassing wrapper.forward). The model-level pre-hook
    # must still fire because it's attached to hf_model, not to the
    # wrapper. The fake doesn't define forward(), so it raises
    # NotImplementedError *after* the pre-hook has already run.
    try:
        wrapper.hf_model(torch.tensor([[1, 2, 3]]))
    except NotImplementedError:
        pass

    assert sched.begin_calls >= 1, (
        "scheduler.begin_forward() was not invoked by the model-level "
        "pre-hook; generate() would skip priming"
    )


def test_pre_hook_missing_view_raises_clearly():
    qlin = StreamingQLinear(128, 64, group_size=32, prefer="torch")
    qlinears = {"self_attn.k_proj": qlin}

    class _IncompleteScheduler:
        def before_layer(self, _idx: int):
            return {}  # empty — every lookup misses

        def after_layer(self, _idx: int) -> None:
            pass

    pre = _make_pre_hook(_IncompleteScheduler(), layer_idx=3, qlinears=qlinears)
    with pytest.raises(RuntimeError, match="slot views missing"):
        pre(_module=nn.Identity(), _inputs=())
