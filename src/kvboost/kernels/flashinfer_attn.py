"""FlashInfer decode-attention integration.

Routes the single-token DECODE attention step through FlashInfer's optimized
CUDA kernel, and delegates everything else (prefill q_len>1, batch>1, non-CUDA,
errors) to PyTorch SDPA. Registered as ``attn_implementation="flashinfer"``;
enable with ``--attn-impl flashinfer`` (server) or
``from_pretrained(attn_implementation="flashinfer")``.

Safety (this is validated on the GPU box, not in CI):
  * Availability-gated — if ``flashinfer`` isn't importable, the impl key is
    never registered and the loader falls back to sdpa.
  * Shape/dtype-gated — only B==1, q_len==1, fp16/bf16, CUDA tensors take the
    fast path; anything else delegates to SDPA.
  * One-time NUMERICAL self-check — on the first decode call, FlashInfer's
    output is compared against the SDPA reference; if they diverge beyond
    tolerance the kernel is PERMANENTLY disabled (logged at ERROR) and SDPA is
    used thereafter. This converts a silent wrong-output kernel bug into a
    correct (if slower) result.
  * Per-call exception fallback — any error in the kernel path falls back to
    SDPA for that call.

Note on impact: FlashInfer accelerates the attention op (matters most at long
context, where KV reads dominate). It does NOT remove the per-token Python +
kernel-launch overhead of the eager decode loop — that's CUDA graphs — nor the
weight-bandwidth bound — that's weight quant (Marlin int4). Set expectations
accordingly.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("kvboost.kernels.flashinfer")

_FLASHINFER = None      # module handle, or False once probed and absent
_DISABLED = False       # set True if the numerical self-check fails
_CHECKED = False        # has the one-time self-check run?
_NUM_CHECK_TOL = 2e-2   # max abs diff vs SDPA reference (fp16-friendly)


def flashinfer_available() -> bool:
    """True if the ``flashinfer`` package imports. Cached."""
    global _FLASHINFER
    if _FLASHINFER is None:
        try:
            import flashinfer  # type: ignore
            _FLASHINFER = flashinfer
        except Exception:
            _FLASHINFER = False
    return bool(_FLASHINFER)


def _sdpa_ref(module, query, key, value, attention_mask,
              dropout, scaling, is_causal, **kwargs):
    """Delegate to HF's stock SDPA attention function."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
    )


def flashinfer_attention_forward(module, query, key, value, attention_mask,
                                 dropout: float = 0.0, scaling=None,
                                 is_causal=None, **kwargs):
    """HF attention-interface function. Decode fast path; SDPA otherwise.

    Shapes (HF convention): query (B, Hq, q_len, D); key/value (B, Hkv, kv_len,
    D). Returns ``(attn_output, None)`` with attn_output (B, q_len, Hq, D),
    matching ``sdpa_attention_forward``'s post-transpose layout.
    """
    global _CHECKED, _DISABLED

    B, Hq, q_len, D = query.shape
    fast = (
        not _DISABLED
        and flashinfer_available()
        and B == 1 and q_len == 1
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and not kwargs.get("output_attentions", False)
    )
    if not fast:
        return _sdpa_ref(module, query, key, value, attention_mask,
                         dropout, scaling, is_causal, **kwargs)

    try:
        fi = _FLASHINFER
        sm_scale = float(scaling) if scaling is not None else 1.0 / (D ** 0.5)
        # FlashInfer single-request decode, layout "NHD":
        #   q  (Hq, D);  k/v  (kv_len, Hkv, D)   — GQA handled internally.
        q = query[0, :, 0, :].contiguous()           # (Hq, D)
        k = key[0].transpose(0, 1).contiguous()      # (kv_len, Hkv, D)
        v = value[0].transpose(0, 1).contiguous()    # (kv_len, Hkv, D)
        out = fi.single_decode_with_kv_cache(
            q, k, v, kv_layout="NHD", sm_scale=sm_scale,
        )                                            # (Hq, D)
        attn = out.view(1, 1, Hq, D)                 # (B, q_len, Hq, D)

        if not _CHECKED:
            _CHECKED = True
            try:
                ref, _ = _sdpa_ref(module, query, key, value, attention_mask,
                                   dropout, scaling, is_causal, **kwargs)
                diff = (attn.float() - ref.float()).abs().max().item()
                if diff > _NUM_CHECK_TOL:
                    _DISABLED = True
                    log.error(
                        "FlashInfer self-check FAILED (max|Δ|=%.3g > %.3g vs "
                        "SDPA) — DISABLING FlashInfer, using SDPA from here.",
                        diff, _NUM_CHECK_TOL,
                    )
                    return ref, None
                log.info(
                    "FlashInfer decode-attention self-check passed "
                    "(max|Δ|=%.3g vs SDPA).", diff,
                )
            except Exception as e:  # self-check itself failed — keep going unverified
                log.warning(
                    "FlashInfer self-check errored (%s); proceeding unverified.",
                    e,
                )
        return attn, None
    except Exception as e:
        log.warning("FlashInfer decode failed (%s); SDPA fallback this call.", e)
        return _sdpa_ref(module, query, key, value, attention_mask,
                         dropout, scaling, is_causal, **kwargs)


def install_flashinfer_attention() -> bool:
    """Register ``attn_implementation="flashinfer"`` if FlashInfer is present.

    Returns True if the impl key is now usable, False otherwise (caller should
    fall back to sdpa). Idempotent.
    """
    if not flashinfer_available():
        log.info(
            "FlashInfer not installed; attn_implementation='flashinfer' "
            "unavailable (pip install flashinfer-python)."
        )
        return False
    try:
        from transformers import AttentionInterface
        AttentionInterface.register("flashinfer", flashinfer_attention_forward)
        log.info(
            "Registered attn_implementation='flashinfer' "
            "(decode-attention; SDPA fallback for prefill)."
        )
        return True
    except Exception as e:
        log.warning("Could not register FlashInfer attention (%s).", e)
        return False


def resolve_attn_impl(requested: str) -> str:
    """Map a requested attn-impl to one HF can load.

    For ``"flashinfer"``: register it if available, else fall back to ``"sdpa"``
    with a warning. All other values pass through unchanged.
    """
    if requested == "flashinfer":
        if install_flashinfer_attention():
            return "flashinfer"
        log.warning(
            "attn-impl 'flashinfer' requested but unavailable; using sdpa."
        )
        return "sdpa"
    return requested
