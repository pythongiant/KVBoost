"""Verify the server CLI wires --awq-streaming into the engine factory.

We don't actually load a model here — we patch
``InferenceEngine.from_pretrained`` and confirm the server's ``load_engine``
calls it with a ``StreamingConfig`` built from the CLI flags.
"""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest


def _fake_args(**overrides):
    """Build the minimal namespace ``load_engine`` reads."""
    defaults = dict(
        model="dummy/model",
        gguf_file=None,
        model_name=None,
        device=None,
        dtype="float16",
        backend="default",
        quantization="none",
        use_slow_tokenizer=False,
        max_memory=None,
        max_cache_bytes=2e9,
        chunk_size=128,
        recompute_strategy="cacheblend",
        kv_cache_bits=16,
        sink_tokens=0,
        overlap_k=0,
        prefill_chunk_size=0,
        block_size=16,
        num_blocks=4096,
        awq_streaming=False,
        streaming_mode="partial_resident",
        keep_first_k=4,
        keep_last_k=4,
        streaming_quant_kernel="auto",
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_awq_streaming_routes_through_engine_factory():
    from kvboost.server.__main__ import load_engine

    args = _fake_args(
        awq_streaming=True,
        streaming_mode="partial_resident",
        keep_first_k=2,
        keep_last_k=2,
        streaming_quant_kernel="torch",
    )

    with patch("kvboost.engine.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("kvboost.engine.InferenceEngine.from_pretrained") as mock_engine_factory:
        mock_tok.return_value.pad_token = "x"
        mock_tok.return_value.eos_token = "x"
        mock_engine_factory.return_value = object()

        load_engine(args)

        mock_engine_factory.assert_called_once()
        kwargs = mock_engine_factory.call_args.kwargs
        sc = kwargs.get("streaming_config")
        assert sc is not None, "streaming_config must be forwarded to the engine"
        assert sc.residency_mode == "partial_resident"
        assert sc.keep_first_k == 2
        assert sc.keep_last_k == 2
        assert sc.quant_kernel == "torch"

        # Standard engine knobs should also flow through.
        assert kwargs["chunk_size"] == args.chunk_size
        assert kwargs["recompute_strategy"] == args.recompute_strategy


def test_awq_streaming_rejects_gguf():
    from kvboost.server.__main__ import load_engine

    args = _fake_args(awq_streaming=True, gguf_file="weights.gguf")

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok:
        mock_tok.return_value.pad_token = "x"
        mock_tok.return_value.eos_token = "x"
        with pytest.raises(SystemExit, match="gguf-file"):
            load_engine(args)


def test_no_awq_streaming_uses_standard_path():
    """Sanity: when --awq-streaming is NOT set, the standard
    AutoModelForCausalLM.from_pretrained path runs (NOT the engine factory).
    """
    from kvboost.server.__main__ import load_engine

    args = _fake_args(awq_streaming=False)

    with patch("transformers.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_hf, \
         patch("kvboost.engine.InferenceEngine.from_pretrained") as mock_engine_factory, \
         patch("kvboost.engine.InferenceEngine.__init__", return_value=None) as mock_engine_init:
        mock_tok.return_value.pad_token = "x"
        mock_tok.return_value.eos_token = "x"
        mock_hf.return_value = object()

        load_engine(args)

        mock_engine_factory.assert_not_called()
        mock_engine_init.assert_called_once()
