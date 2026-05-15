"""S4 gate: verify partial_resident actually saves VRAM vs full_resident.

This is a sanity check, not a hard absolute bound — the saving depends on
``keep_first_k`` / ``keep_last_k`` and on transformers' overhead. We just
require that the streaming load uses strictly less peak VRAM than the
fully-resident one.

Run with::

    pytest -m slow tests/streaming/test_streaming_vram_budget.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.slow


MODEL_ID = os.environ.get(
    "KVBOOST_STREAMING_VRAM_MODEL",
    "Qwen/Qwen2.5-3B-Instruct-AWQ",
)


def _peak_after_load(streaming_config) -> int:
    from kvboost.streaming import StreamingCausalLM

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = StreamingCausalLM.from_pretrained(
        MODEL_ID,
        streaming_config=streaming_config,
        dtype=torch.float16,
    )
    if streaming_config.residency_mode == "full_resident":
        model.hf_model.cuda()
    peak = torch.cuda.max_memory_allocated()
    del model
    torch.cuda.empty_cache()
    return peak


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_partial_resident_saves_vram():
    from kvboost.streaming import StreamingConfig

    full_peak = _peak_after_load(StreamingConfig(residency_mode="full_resident"))
    partial_peak = _peak_after_load(
        StreamingConfig(
            residency_mode="partial_resident",
            keep_first_k=4,
            keep_last_k=4,
        )
    )
    # Streaming should save *something*. Don't put a tight bound on the gap
    # because it depends on model size; assert at least 10% savings.
    assert partial_peak < full_peak, (
        f"streaming did not save VRAM: full={full_peak/1e9:.2f}GB, "
        f"partial={partial_peak/1e9:.2f}GB"
    )
    savings = (full_peak - partial_peak) / full_peak
    assert savings > 0.10, f"expected >10% VRAM savings, got {savings:.1%}"
