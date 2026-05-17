"""Tests for StreamingConfig validation + residency gating."""

import pytest

from kvboost.streaming.config import StreamingConfig


def test_defaults_validate():
    cfg = StreamingConfig()
    cfg.validate()
    assert cfg.use_partial_residency
    assert not cfg.use_full_streaming
    assert not cfg.use_ffn_only_streaming


def test_full_resident_disables_streaming():
    cfg = StreamingConfig(residency_mode="full_resident")
    assert not cfg.should_stream_model(100)


def test_small_model_skips_streaming():
    cfg = StreamingConfig(streaming_disable_below_layers=12)
    assert not cfg.should_stream_model(8)
    assert cfg.should_stream_model(12)
    assert cfg.should_stream_model(32)


def test_invalid_keep_k_raises():
    with pytest.raises(ValueError):
        StreamingConfig(keep_first_k=-1).validate()
    with pytest.raises(ValueError):
        StreamingConfig(keep_last_k=-1).validate()


def test_double_buffering_requires_two_slots():
    cfg = StreamingConfig(enable_double_buffering=True, n_staging_slots=1)
    with pytest.raises(ValueError):
        cfg.validate()


def test_n_staging_slots_zero_is_auto_sentinel():
    """0 = 'pick at load time based on free VRAM'. Validate() must accept
    it and the summary must call it out so traces are readable.
    """
    cfg = StreamingConfig(n_staging_slots=0)
    cfg.validate()  # must not raise
    assert "slots=auto" in cfg.summary()


def test_negative_n_staging_slots_rejected():
    with pytest.raises(ValueError):
        StreamingConfig(n_staging_slots=-1).validate()


def test_auto_slots_bounds_validated():
    with pytest.raises(ValueError):
        StreamingConfig(auto_slots_margin_gb=-0.5).validate()
    with pytest.raises(ValueError):
        StreamingConfig(auto_slots_max=1).validate()


def test_mode_helpers():
    assert StreamingConfig(residency_mode="full_stream").use_full_streaming
    assert StreamingConfig(residency_mode="ffn_only_stream").use_ffn_only_streaming
    assert not StreamingConfig(residency_mode="full_resident").use_partial_residency
