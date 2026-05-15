"""M5: end-to-end forward parity for a fully-resident streaming model.

This test downloads a small AWQ model from the Hub and checks that the
streaming-shell wrapper (in ``full_resident`` mode) produces logits matching
the underlying HF AWQ model. Marked ``slow`` — skipped by default.

To run:
    pytest -m slow tests/streaming/test_full_forward.py
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = pytest.mark.slow


MODEL_ID = os.environ.get(
    "KVBOOST_STREAMING_TEST_MODEL",
    "casperhansen/llama-3.2-1b-instruct-awq",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_full_resident_parity_with_hf():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvboost.streaming import StreamingCausalLM, StreamingConfig

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    inputs = tok("Hello world.", return_tensors="pt").to("cuda")

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float16).cuda()
    hf_model.eval()
    with torch.inference_mode():
        ref_logits = hf_model(**inputs).logits

    del hf_model
    torch.cuda.empty_cache()

    wrapped = StreamingCausalLM.from_pretrained(
        MODEL_ID,
        streaming_config=StreamingConfig(residency_mode="full_resident"),
        dtype=torch.float16,
    )
    wrapped.hf_model.cuda()
    with torch.inference_mode():
        out_logits = wrapped(**inputs).logits

    assert torch.allclose(out_logits, ref_logits, atol=1e-2, rtol=1e-2)
