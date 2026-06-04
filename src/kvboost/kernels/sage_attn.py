"""SageAttention (INT8) + Triton FlashAttention-2 (FP16) backends.

Registers two HuggingFace attention implementations backed by the Triton
kernel in :mod:`kvboost.kernels.sage_kernels`:

* ``attn_implementation="sage"``        — SageAttention v1 (arXiv:2410.02367):
  INT8 Q·Kᵀ with per-channel K-smoothing, fp16 P·V. The genuine win SDPA/FA2
  can't give you on Ampere — INT8 tensor-core QKᵀ.
* ``attn_implementation="triton_flash"`` — plain FP16 FlashAttention-2 in
  Triton. Roughly matches torch SDPA's flash kernel on Ampere; primarily the
  correctness baseline / the "our own flash attention" deliverable.

Enable with ``--attn-impl sage`` (server) or
``from_pretrained(attn_implementation="sage")``.

Why Triton: it JIT-compiles through the CUDA *driver* at runtime — no nvcc, no
prebuilt-wheel matching, no multi-arch source build (the things that were
breaking the flash-attn install). It emits tensor-core code for sm_86 and is
how SageAttention's reference kernels are written.

Scope / fast-path gating (everything else delegates to SDPA, so a missing or
misbehaving kernel never corrupts output):
  * Availability-gated — if ``triton`` doesn't import, the impl keys are never
    registered and the loader falls back to sdpa.
  * Shape/dtype-gated — only fp16/bf16 CUDA tensors with head_dim ∈ {64, 128},
    q_len > 1 (PREFILL; single-token decode → SDPA, or FlashInfer if enabled),
    and ``attention_mask is None`` (causal-only; padded batches → SDPA) take the
    fast path.
  * One-time NUMERICAL self-check — on the first prefill the kernel output is
    compared against the SDPA reference; if they diverge beyond tolerance the
    backend is PERMANENTLY disabled (logged at ERROR) and SDPA is used
    thereafter. Turns a silently-wrong kernel into a correct (slower) result.
  * Per-call exception fallback — any error in the kernel path falls back to
    SDPA for that call.

Decode vs prefill: SageAttention accelerates the (compute-bound) prefill QKᵀ.
Single-token decode is bandwidth-bound on KV reads — use FlashInfer
(``--attn-impl flashinfer``) for that; the two are complementary, and this
backend explicitly delegates q_len==1 to SDPA.
"""
from __future__ import annotations

import logging

import torch

log = logging.getLogger("kvboost.kernels.sage")

_TRITON = None          # None=unprobed, module once present, False once absent
# Per-backend self-check state.
_STATE = {
    "sage":         {"checked": False, "disabled": False},
    "triton_flash": {"checked": False, "disabled": False},
}
# INT8 attention is lossy by design, so its tolerance vs the fp16 SDPA
# reference is looser than the (near-exact) fp16 Triton path.
_TOL = {"sage": 3e-2, "triton_flash": 2e-2}


def triton_available() -> bool:
    """True if ``triton`` imports. Cached."""
    global _TRITON
    if _TRITON is None:
        try:
            import triton  # type: ignore
            _TRITON = triton
        except Exception:
            _TRITON = False
    return bool(_TRITON)


# Alias used by the package smoke test / availability reporting.
def sage_available() -> bool:
    return triton_available()


# ── Quantisation (Python side; cheap, memory-bound — dwarfed by O(N²·D)) ──────

def _quant_per_token(x: torch.Tensor):
    """Symmetric per-token INT8 quant of ``x`` ([B, H, S, D]).

    Returns ``(q_int8 [B,H,S,D], scale_fp32 [B,H,S])`` with
    ``x ≈ q_int8 * scale[..., None]``. Per-token (one scale per (b,h,s)) is the
    most accurate granularity and keeps the in-kernel dequant a cheap
    row/column broadcast.
    """
    xf = x.float()
    amax = xf.abs().amax(dim=-1)                       # [B, H, S]
    scale = (amax / 127.0).clamp(min=1e-8)
    q = (xf / scale.unsqueeze(-1)).round().clamp_(-127, 127).to(torch.int8)
    return q.contiguous(), scale.to(torch.float32).contiguous()


def _smooth_and_quant_k(k: torch.Tensor):
    """SageAttention K-smoothing + per-token INT8 quant.

    Subtracts the per-channel token-mean from K before quantising. The mean
    adds a per-query constant to every logit in a row → cancels in the softmax,
    so no correction term is needed downstream.
    """
    kf = k.float()
    delta = kf.mean(dim=2, keepdim=True)               # [B, Hkv, 1, D]
    return _quant_per_token(kf - delta)


def _launch(query, key, value, sm_scale: float, causal: bool, use_int8: bool):
    """Run the Triton kernel. Returns O in ``[B, Hq, Sq, D]`` (query layout)."""
    from .sage_kernels import (
        _attn_fwd, BLOCK_M, BLOCK_N, NUM_WARPS, NUM_STAGES,
    )
    import triton

    B, Hq, Sq, D = query.shape
    Hkv, Skv = key.shape[1], key.shape[2]
    q = query.contiguous()
    v = value.contiguous()
    out = torch.empty_like(q)

    if use_int8:
        q_arg, q_scale = _quant_per_token(q)
        k_arg, k_scale = _smooth_and_quant_k(key)
        qs_strides = (q_scale.stride(0), q_scale.stride(1), q_scale.stride(2))
        ks_strides = (k_scale.stride(0), k_scale.stride(1), k_scale.stride(2))
    else:
        q_arg, k_arg = q, key.contiguous()
        # Unused placeholders (the kernel never loads them when INT8=False).
        q_scale = k_scale = torch.empty(1, device=q.device, dtype=torch.float32)
        qs_strides = ks_strides = (0, 0, 0)

    grid = (triton.cdiv(Sq, BLOCK_M), B * Hq)
    _attn_fwd[grid](
        q_arg, k_arg, v, q_scale, k_scale, out, sm_scale,
        q_arg.stride(0), q_arg.stride(1), q_arg.stride(2), q_arg.stride(3),
        k_arg.stride(0), k_arg.stride(1), k_arg.stride(2), k_arg.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        qs_strides[0], qs_strides[1], qs_strides[2],
        ks_strides[0], ks_strides[1], ks_strides[2],
        B, Hq, Hkv, Sq, Skv,
        HEAD_DIM=D, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        CAUSAL=causal, INT8=use_int8,
        num_warps=NUM_WARPS, num_stages=NUM_STAGES,
    )
    return out


# ── HF attention-interface integration ───────────────────────────────────────

def _sdpa_ref(module, query, key, value, attention_mask,
              dropout, scaling, is_causal, **kwargs):
    """Delegate to HF's stock SDPA attention function."""
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    return sdpa_attention_forward(
        module, query, key, value, attention_mask,
        dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs,
    )


def _forward(name: str, use_int8: bool, module, query, key, value,
             attention_mask, dropout: float = 0.0, scaling=None,
             is_causal=None, **kwargs):
    """Shared HF attention-interface body for the sage / triton_flash backends.

    Shapes (HF convention): query (B, Hq, q_len, D); key/value (B, Hkv, kv_len,
    D) with GQA un-repeated. Returns ``(attn_output, None)`` with attn_output
    (B, q_len, Hq, D) — matching ``sdpa_attention_forward``'s layout.
    """
    st = _STATE[name]
    B, Hq, Sq, D = query.shape
    causal = bool(is_causal) if is_causal is not None else (Sq > 1)

    fast = (
        not st["disabled"]
        and triton_available()
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and D in (64, 128)
        and Sq > 1                                  # prefill only; decode → SDPA
        and attention_mask is None                  # causal-only; padded → SDPA
        and not kwargs.get("output_attentions", False)
    )
    if not fast:
        return _sdpa_ref(module, query, key, value, attention_mask,
                         dropout, scaling, is_causal, **kwargs)

    try:
        sm_scale = float(scaling) if scaling is not None else 1.0 / (D ** 0.5)
        o = _launch(query, key, value, sm_scale, causal, use_int8)   # [B,Hq,Sq,D]
        attn = o.transpose(1, 2).contiguous()                        # [B,Sq,Hq,D]

        if not st["checked"]:
            st["checked"] = True
            try:
                ref, _ = _sdpa_ref(module, query, key, value, attention_mask,
                                   dropout, scaling, is_causal, **kwargs)
                diff = (attn.float() - ref.float()).abs().max().item()
                if diff > _TOL[name]:
                    st["disabled"] = True
                    log.error(
                        "%s self-check FAILED (max|Δ|=%.3g > %.3g vs SDPA) — "
                        "DISABLING %s, using SDPA from here.",
                        name, diff, _TOL[name], name,
                    )
                    return ref, None
                log.info("%s self-check passed (max|Δ|=%.3g vs SDPA).", name, diff)
            except Exception as e:  # self-check itself failed — keep going
                log.warning("%s self-check errored (%s); proceeding unverified.",
                            name, e)
        return attn, None
    except Exception as e:
        log.warning("%s kernel failed (%s); SDPA fallback this call.", name, e)
        return _sdpa_ref(module, query, key, value, attention_mask,
                         dropout, scaling, is_causal, **kwargs)


def sage_attention_forward(module, query, key, value, attention_mask,
                           dropout: float = 0.0, scaling=None,
                           is_causal=None, **kwargs):
    """INT8 SageAttention; SDPA fallback. HF attention-interface signature."""
    return _forward("sage", True, module, query, key, value, attention_mask,
                    dropout, scaling, is_causal, **kwargs)


def triton_flash_attention_forward(module, query, key, value, attention_mask,
                                   dropout: float = 0.0, scaling=None,
                                   is_causal=None, **kwargs):
    """FP16 Triton FlashAttention-2; SDPA fallback. HF interface signature."""
    return _forward("triton_flash", False, module, query, key, value,
                    attention_mask, dropout, scaling, is_causal, **kwargs)


def install_sage_attention() -> bool:
    """Register ``"sage"`` and ``"triton_flash"`` if Triton is present.

    Returns True if the impl keys are usable, False otherwise (caller should
    fall back to sdpa). Idempotent.
    """
    if not triton_available():
        log.info(
            "Triton not installed; attn_implementation='sage'/'triton_flash' "
            "unavailable (pip install triton)."
        )
        return False
    try:
        from transformers import AttentionInterface
        AttentionInterface.register("sage", sage_attention_forward)
        AttentionInterface.register("triton_flash", triton_flash_attention_forward)
        log.info(
            "Registered attn_implementation='sage' (INT8 SageAttention prefill) "
            "and 'triton_flash' (FP16 Triton flash); SDPA fallback for decode / "
            "unsupported shapes."
        )
        return True
    except Exception as e:
        log.warning("Could not register Triton attention backends (%s).", e)
        return False
