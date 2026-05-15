"""M1: AWQ loader smoke tests.

These tests construct an :class:`AWQLoader` against a synthetic safetensors
shard so we don't need network access. They verify that the tensor index,
layer index, and residency policy behave correctly across the residency
modes used by KVBoost.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from kvboost.streaming.awq_loader import AWQLoader, detect_device
from kvboost.streaming.config import StreamingConfig


def _build_fake_awq_repo(tmp_path: Path, num_layers: int = 6) -> Path:
    tensors: dict[str, torch.Tensor] = {}
    tensors["model.embed_tokens.weight"] = torch.zeros(32, 16, dtype=torch.float16)
    tensors["model.norm.weight"] = torch.ones(16, dtype=torch.float16)
    tensors["lm_head.weight"] = torch.zeros(32, 16, dtype=torch.float16)

    for i in range(num_layers):
        prefix = f"model.layers.{i}"
        tensors[f"{prefix}.input_layernorm.weight"] = torch.ones(16, dtype=torch.float16)
        tensors[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(16, dtype=torch.float16)
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            tensors[f"{prefix}.self_attn.{proj}.qweight"] = torch.zeros(16, 2, dtype=torch.int32)
            tensors[f"{prefix}.self_attn.{proj}.scales"] = torch.zeros(2, 16, dtype=torch.float16)
            tensors[f"{prefix}.self_attn.{proj}.qzeros"] = torch.zeros(2, 2, dtype=torch.int32)
        for proj in ("gate_proj", "up_proj", "down_proj"):
            tensors[f"{prefix}.mlp.{proj}.qweight"] = torch.zeros(16, 2, dtype=torch.int32)
            tensors[f"{prefix}.mlp.{proj}.scales"] = torch.zeros(2, 16, dtype=torch.float16)
            tensors[f"{prefix}.mlp.{proj}.qzeros"] = torch.zeros(2, 2, dtype=torch.int32)

    save_file(tensors, str(tmp_path / "model.safetensors"))

    with open(tmp_path / "config.json", "w") as f:
        json.dump(
            {
                "model_type": "llama",
                "num_hidden_layers": num_layers,
                "hidden_size": 16,
                "tie_word_embeddings": False,
            },
            f,
        )
    with open(tmp_path / "quantize_config.json", "w") as f:
        json.dump({"bits": 4, "group_size": 8, "version": "GEMM"}, f)

    return tmp_path


def _make_loader(tmp_path: Path, streaming_config: StreamingConfig) -> AWQLoader:
    repo = _build_fake_awq_repo(tmp_path)

    loader = AWQLoader.__new__(AWQLoader)
    loader.model_name_or_path = str(repo)
    loader.streaming_config = streaming_config
    loader.revision = None
    loader.cache_dir = None
    loader.device_spec = detect_device("cpu")
    loader.max_workers = 1
    loader.model_dir = repo
    loader.index = None
    loader._resident_tensors = {}
    loader._pinned_tensors = {}
    return loader


def test_loader_indexes_layers(tmp_path):
    cfg = StreamingConfig(residency_mode="full_resident")
    loader = _make_loader(tmp_path, cfg)

    # Manually drive the indexing pipeline since we skip snapshot_download.
    tensors = loader._build_tensor_index()
    layers = loader._build_layer_index(tensors)

    assert len(layers) == 6
    for layer_idx, spec in layers.items():
        assert {"input_layernorm.weight", "post_attention_layernorm.weight"} <= {
            k.split(".", 3)[-1] for k in spec.tensors.keys()
        }
        assert any("self_attn.q_proj.qweight" in k for k in spec.tensors.keys())


def test_residency_policy_partial(tmp_path):
    cfg = StreamingConfig(residency_mode="partial_resident", keep_first_k=2, keep_last_k=1)
    loader = _make_loader(tmp_path, cfg)

    tensors = loader._build_tensor_index()
    layers = loader._build_layer_index(tensors)
    from kvboost.streaming.awq_loader import AWQIndex

    loader.index = AWQIndex(
        model_dir=loader.model_dir,
        config={"num_hidden_layers": 6, "tie_word_embeddings": False},
        quant_config={"bits": 4, "group_size": 8},
        tensors=tensors,
        layers=layers,
        tied_embeddings=False,
    )
    loader._apply_residency_policy()

    # Layers 0, 1 (keep_first_k=2) and 5 (keep_last_k=1) must be resident;
    # 2, 3, 4 must stream. Layernorms are always resident regardless of layer,
    # so only check the projection weights.
    by_layer_proj: dict[int, list] = {i: [] for i in range(6)}
    for name, spec in loader.index.tensors.items():
        if spec.layer_idx is None:
            continue
        if "proj" not in name:
            continue
        by_layer_proj[spec.layer_idx].append(spec.is_resident)
    for idx in (0, 1, 5):
        assert all(by_layer_proj[idx]), f"layer {idx} projections should be resident"
    for idx in (2, 3, 4):
        assert not any(by_layer_proj[idx]), f"layer {idx} projections should stream"


def test_residency_policy_ffn_only_keeps_attention_resident(tmp_path):
    cfg = StreamingConfig(residency_mode="ffn_only_stream", keep_first_k=0, keep_last_k=0)
    loader = _make_loader(tmp_path, cfg)

    tensors = loader._build_tensor_index()
    layers = loader._build_layer_index(tensors)
    from kvboost.streaming.awq_loader import AWQIndex

    loader.index = AWQIndex(
        model_dir=loader.model_dir,
        config={"num_hidden_layers": 6, "tie_word_embeddings": False},
        quant_config={"bits": 4, "group_size": 8},
        tensors=tensors,
        layers=layers,
        tied_embeddings=False,
    )
    loader._apply_residency_policy()

    for name, spec in loader.index.tensors.items():
        if spec.layer_idx is None:
            continue
        if "self_attn" in name:
            assert spec.is_resident, f"{name} should be resident in ffn_only mode"
        elif "mlp" in name:
            assert not spec.is_resident, f"{name} should stream in ffn_only mode"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
def test_loader_device_spec_cuda():
    spec = detect_device("cuda")
    assert spec.kind == "cuda"
    assert spec.use_pinned_memory
    assert spec.supports_async_transfer


def test_loader_device_spec_cpu_fallback():
    spec = detect_device("cpu")
    assert spec.kind == "cpu"
    assert not spec.use_pinned_memory
