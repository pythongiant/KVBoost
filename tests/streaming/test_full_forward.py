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

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16).cuda()
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

    # Loading the same AWQ model twice in one process can pick different
    # cuBLAS/autoawq kernel autotune paths, which introduces fp16-noise
    # differences in the last 2–3 mantissa bits. What matters for
    # generation correctness is greedy-decode equivalence: top-1 token
    # IDs must agree at every position.
    ref_top1 = ref_logits.argmax(dim=-1)
    out_top1 = out_logits.argmax(dim=-1)
    assert torch.equal(ref_top1, out_top1), (
        f"top-1 token mismatch: ref={ref_top1.tolist()} out={out_top1.tolist()}"
    )

    # As a softer numerical sanity check, also bound the largest fp16
    # deviation. 0.05 is comfortably above autotune jitter and well below
    # what would change argmax in practice.
    max_abs = (out_logits.float() - ref_logits.float()).abs().max().item()
    assert max_abs < 0.05, f"max abs logit diff {max_abs:.4f} > 0.05"
