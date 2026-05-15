"""Test that `InferenceEngine.from_pretrained` accepts a streaming_config kw.

We don't actually load a model here — we just confirm the dispatch chooses
the streaming code path. The streaming wrapper itself is exercised by the
end-to-end M5 test (marked slow).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_engine_dispatches_to_streaming_loader():
    from kvboost.streaming import StreamingConfig

    cfg = StreamingConfig(residency_mode="full_resident")

    with patch("kvboost.engine.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("kvboost.streaming.model_shell.StreamingCausalLM.from_pretrained") as mock_stream:
        mock_tok.return_value.pad_token = "x"
        mock_tok.return_value.eos_token = "x"
        mock_stream.return_value = _FakeModel()

        from kvboost.engine import InferenceEngine

        with patch("kvboost.engine.check_model_compatibility"):
            with patch.object(InferenceEngine, "__init__", return_value=None) as init:
                InferenceEngine.from_pretrained(
                    "dummy/model",
                    streaming_config=cfg,
                    awq_path="local-awq",
                )

        mock_stream.assert_called_once()
        kwargs = mock_stream.call_args.kwargs
        assert kwargs["streaming_config"] is cfg
        assert kwargs["awq_path"] == "local-awq"
        init.assert_called_once()


def test_engine_skips_streaming_when_config_is_none():
    with patch("kvboost.engine.AutoTokenizer.from_pretrained") as mock_tok, \
         patch("kvboost.engine.AutoModelForCausalLM.from_pretrained") as mock_hf, \
         patch("kvboost.streaming.model_shell.StreamingCausalLM.from_pretrained") as mock_stream:
        mock_tok.return_value.pad_token = "x"
        mock_tok.return_value.eos_token = "x"
        mock_hf.return_value = _FakeModel()

        from kvboost.engine import InferenceEngine

        with patch("kvboost.engine.check_model_compatibility"):
            with patch.object(InferenceEngine, "__init__", return_value=None):
                InferenceEngine.from_pretrained("dummy/model")

        mock_hf.assert_called_once()
        mock_stream.assert_not_called()


class _FakeModel:
    def eval(self):
        return self
