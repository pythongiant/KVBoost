"""Offline tests for the MPS unified-memory load path.

These never touch ``StreamingCausalLM.from_pretrained`` end-to-end (that
requires downloading a real AWQ model), but exercise the two pieces the
MPS branch adds:

- :meth:`AWQLoader.bind_streaming_qlinears` — one-shot bind from disk
- :func:`_replace_linears_for_quant_paths` — replaces plain ``nn.Linear``
  at AWQ projection paths with :class:`StreamingQLinear`

The tests pass on CPU (and on MPS): the binding is device-agnostic, and
the pure-torch dequant kernel works on any backend.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from kvboost.streaming.awq_loader import AWQIndex, AWQLoader, detect_device
from kvboost.streaming.config import StreamingConfig
from kvboost.streaming.model_shell import (
    _replace_linears_for_quant_paths,
    _iter_decoder_layers,
)
from kvboost.streaming.qkv_proj import StreamingQLinear


# ── Synthetic checkpoint ────────────────────────────────────────────────────


def _build_fake_awq_repo(tmp_path: Path, num_layers: int = 4) -> Path:
    pack = 8
    in_features = 16
    out_features = 16
    group_size = 8

    tensors: dict[str, torch.Tensor] = {}
    tensors["model.embed_tokens.weight"] = torch.zeros(32, in_features, dtype=torch.float16)
    tensors["model.norm.weight"] = torch.ones(in_features, dtype=torch.float16)
    tensors["lm_head.weight"] = torch.zeros(32, in_features, dtype=torch.float16)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(in_features, dtype=torch.float16)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(in_features, dtype=torch.float16)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{prefix}.self_attn.{proj}.qweight"] = torch.zeros(
                in_features, out_features // pack, dtype=torch.int32,
            )
            tensors[f"{prefix}.self_attn.{proj}.scales"] = torch.zeros(
                in_features // group_size, out_features, dtype=torch.float16,
            )
            tensors[f"{prefix}.self_attn.{proj}.qzeros"] = torch.zeros(
                in_features // group_size, out_features // pack, dtype=torch.int32,
            )
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"{prefix}.mlp.{proj}.qweight"] = torch.zeros(
                in_features, out_features // pack, dtype=torch.int32,
            )
            tensors[f"{prefix}.mlp.{proj}.scales"] = torch.zeros(
                in_features // group_size, out_features, dtype=torch.float16,
            )
            tensors[f"{prefix}.mlp.{proj}.qzeros"] = torch.zeros(
                in_features // group_size, out_features // pack, dtype=torch.int32,
            )

    save_file(tensors, str(tmp_path / "model.safetensors"))
    with open(tmp_path / "config.json", "w") as f:
        json.dump(
            {
                "model_type": "llama",
                "num_hidden_layers": num_layers,
                "hidden_size": in_features,
                "tie_word_embeddings": False,
            },
            f,
        )
    with open(tmp_path / "quantize_config.json", "w") as f:
        json.dump({"bits": 4, "group_size": group_size, "version": "GEMM"}, f)
    return tmp_path


def _make_loader(tmp_path: Path, *, num_layers: int = 4) -> AWQLoader:
    repo = _build_fake_awq_repo(tmp_path, num_layers=num_layers)
    cfg = StreamingConfig(residency_mode="partial_resident")
    loader = AWQLoader.__new__(AWQLoader)
    loader.model_name_or_path = str(repo)
    loader.streaming_config = cfg
    loader.revision = None
    loader.cache_dir = None
    loader.device_spec = detect_device("cpu")
    loader.max_workers = 1
    loader.model_dir = repo
    loader.index = None
    loader._resident_tensors = {}
    loader._pinned_tensors = {}

    tensors = loader._build_tensor_index()
    layers = loader._build_layer_index(tensors)
    loader.index = AWQIndex(
        model_dir=repo,
        config={"num_hidden_layers": num_layers, "tie_word_embeddings": False},
        quant_config={"bits": 4, "group_size": 8},
        tensors=tensors,
        layers=layers,
        tied_embeddings=False,
    )
    loader._apply_residency_policy()
    return loader


# ── Skeleton fixtures (no transformers required) ────────────────────────────


class _FakeAttn(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)


class _FakeMlp(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden, hidden, bias=False)
        self.up_proj = nn.Linear(hidden, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, hidden, bias=False)


class _FakeDecoderLayer(nn.Module):
    def __init__(self, hidden: int = 16) -> None:
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden)
        self.post_attention_layernorm = nn.LayerNorm(hidden)
        self.self_attn = _FakeAttn(hidden)
        self.mlp = _FakeMlp(hidden)


class _FakeInner(nn.Module):
    def __init__(self, num_layers: int, hidden: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden)
        self.layers = nn.ModuleList(_FakeDecoderLayer(hidden) for _ in range(num_layers))
        self.norm = nn.LayerNorm(hidden)


class _FakeHfModel(nn.Module):
    def __init__(self, num_layers: int, hidden: int = 16) -> None:
        super().__init__()
        from types import SimpleNamespace
        self.config = SimpleNamespace(num_hidden_layers=num_layers)
        self.model = _FakeInner(num_layers, hidden)
        self.lm_head = nn.Linear(hidden, 32, bias=False)


# ── Tests ───────────────────────────────────────────────────────────────────


def test_replace_linears_for_quant_paths_swaps_all_seven_per_layer(tmp_path):
    loader = _make_loader(tmp_path, num_layers=3)
    model = _FakeHfModel(num_layers=3)

    replacements = _replace_linears_for_quant_paths(
        model, loader=loader, group_size=8, prefer="torch",
    )

    assert set(replacements.keys()) == {0, 1, 2}
    expected_paths = {
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
        "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
    }
    for layer_idx in (0, 1, 2):
        assert set(replacements[layer_idx].keys()) == expected_paths
        for sub_path in expected_paths:
            parent_chain = sub_path.split(".")
            sub: nn.Module = model.model.layers[layer_idx]
            for p in parent_chain:
                sub = getattr(sub, p)
            assert isinstance(sub, StreamingQLinear)
            assert sub.in_features == 16
            assert sub.out_features == 16


def test_bind_streaming_qlinears_loads_from_disk(tmp_path):
    loader = _make_loader(tmp_path, num_layers=2)
    model = _FakeHfModel(num_layers=2)

    replacements = _replace_linears_for_quant_paths(
        model, loader=loader, group_size=8, prefer="torch",
    )

    # None of the StreamingQLinears are bound yet.
    for layer_repl in replacements.values():
        for qlin in layer_repl.values():
            assert not qlin.is_bound

    loader.bind_streaming_qlinears(replacements, device=torch.device("cpu"))

    # Every replacement now bound.
    for layer_idx, layer_repl in replacements.items():
        for sub_path, qlin in layer_repl.items():
            assert qlin.is_bound, f"layer {layer_idx} {sub_path} not bound"


def test_bound_streaming_qlinear_runs_forward(tmp_path):
    loader = _make_loader(tmp_path, num_layers=1)
    model = _FakeHfModel(num_layers=1)

    replacements = _replace_linears_for_quant_paths(
        model, loader=loader, group_size=8, prefer="torch",
    )
    loader.bind_streaming_qlinears(replacements, device=torch.device("cpu"))

    q_proj = model.model.layers[0].self_attn.q_proj
    out = q_proj(torch.zeros(2, 16, dtype=torch.float16))
    assert out.shape == (2, 16)


def test_iter_decoder_layers_on_fake_skeleton():
    model = _FakeHfModel(num_layers=4)
    pairs = _iter_decoder_layers(model)
    assert [i for i, _ in pairs] == [0, 1, 2, 3]
