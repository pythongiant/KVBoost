"""Triton FlashAttention-2 forward kernel — FP16 and INT8 (SageAttention) modes.

This is the low-level kernel used by :mod:`kvboost.kernels.sage_attn`. It is
imported **lazily** (only from inside the launch wrapper) so that importing
``kvboost`` on a box without Triton/CUDA never touches ``@triton.jit`` code.

Two modes, one kernel, selected by the ``INT8`` compile-time constant:

* ``INT8 == False`` — plain FlashAttention-2. Q·Kᵀ runs in the input dtype
  (fp16/bf16) on the tensor cores → fp32 accumulator. This is the "build our
  own flash attention" path; it also serves as the numerical baseline.

* ``INT8 == True``  — SageAttention (Zhang et al., 2024, arXiv:2410.02367).
  Q and (smoothed) K arrive pre-quantized to INT8 with per-token scales, so
  Q·Kᵀ runs as an INT8 tensor-core matmul (``mma.s8.s8.s32``, available on
  sm_80/86/89/90) → int32, then dequantised by ``q_scale·k_scale·sm_scale``.
  P (softmax probabilities) and V stay in fp16/bf16 for the P·V matmul — only
  the QKᵀ matmul is quantised, exactly as in SageAttention v1.

The softmax (online, numerically stable) and the P·V matmul are byte-for-byte
identical between the two modes — only the QKᵀ section differs. That keeps the
INT8 path easy to validate against the FP16 path and against SDPA.

K-smoothing note: SageAttention subtracts the per-channel token-mean from K
before quantisation (K has large channel outliers that wreck INT8). The mean
subtraction adds a per-query *constant* to every logit in a row, which cancels
in the softmax — so no correction term is needed here; the smoothing happens in
the Python wrapper and this kernel just consumes the already-smoothed INT8 K.

Block sizes / launch params are env-overridable for tuning on the target GPU
without editing code (the defaults are conservative for sm_86's 100 KB smem):

    KVBOOST_SAGE_BLOCK_M, KVBOOST_SAGE_BLOCK_N, KVBOOST_SAGE_WARPS,
    KVBOOST_SAGE_STAGES
"""
from __future__ import annotations

import os

import triton
import triton.language as tl


def _envi(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Conservative defaults: 64×64 tiles, 4 warps, 2 pipeline stages keep shared
# memory well under sm_86's 100 KB opt-in cap for head_dim up to 128.
BLOCK_M = _envi("KVBOOST_SAGE_BLOCK_M", 64)
BLOCK_N = _envi("KVBOOST_SAGE_BLOCK_N", 64)
NUM_WARPS = _envi("KVBOOST_SAGE_WARPS", 4)
NUM_STAGES = _envi("KVBOOST_SAGE_STAGES", 2)


@triton.jit
def _attn_fwd(
    Q,            # [B, Hq, Sq, D]  fp16/bf16 (FP16 mode) or int8 (INT8 mode)
    K,            # [B, Hkv, Skv, D] same dtype family as Q
    V,            # [B, Hkv, Skv, D] fp16/bf16 always
    Q_scale,      # [B, Hq, Sq]   fp32 per-token scale (INT8 mode); placeholder otherwise
    K_scale,      # [B, Hkv, Skv] fp32 per-token scale (INT8 mode); placeholder otherwise
    Out,          # [B, Hq, Sq, D] fp16/bf16
    sm_scale,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_qsb, stride_qsh, stride_qsm,
    stride_ksb, stride_ksh, stride_ksn,
    B, Hq, Hkv, Sq, Skv,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    INT8: tl.constexpr,
):
    pid_m = tl.program_id(0)          # which BLOCK_M-sized query tile
    off_bh = tl.program_id(1)         # flat (batch * Hq + q-head)
    b = off_bh // Hq
    hq = off_bh % Hq
    groups = Hq // Hkv                # GQA: q-heads per kv-head
    hkv = hq // groups

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    m_mask = offs_m < Sq

    # ── Load this tile's Q rows ([BLOCK_M, HEAD_DIM]) ────────────────────────
    q_base = Q + b * stride_qb + hq * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0)

    if INT8:
        q_scale = tl.load(
            Q_scale + b * stride_qsb + hq * stride_qsh + offs_m * stride_qsm,
            mask=m_mask, other=0.0,
        )

    k_base = K + b * stride_kb + hkv * stride_kh
    v_base = V + b * stride_vb + hkv * stride_vh

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # Causal alignment: query global pos i attends keys j ≤ i + (Skv - Sq).
    causal_offset = Skv - Sq
    if CAUSAL:
        hi = tl.minimum(Skv, (pid_m + 1) * BLOCK_M + causal_offset)
    else:
        hi = Skv

    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        n_idx = start_n + offs_n
        n_mask = n_idx < Skv

        # K tile laid out as [HEAD_DIM, BLOCK_N] so tl.dot(q, k) = Q·Kᵀ.
        k_ptrs = k_base + n_idx[None, :] * stride_kn + offs_d[:, None] * stride_kd
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0)

        if INT8:
            qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
            k_scale = tl.load(
                K_scale + b * stride_ksb + hkv * stride_ksh + n_idx * stride_ksn,
                mask=n_mask, other=0.0,
            )
            qk = qk * q_scale[:, None] * k_scale[None, :] * sm_scale
        else:
            qk = tl.dot(q, k) * sm_scale

        qk = tl.where(n_mask[None, :], qk, float("-inf"))
        if CAUSAL:
            causal_ok = (offs_m[:, None] + causal_offset) >= n_idx[None, :]
            qk = tl.where(causal_ok, qk, float("-inf"))

        # ── Online softmax update ────────────────────────────────────────────
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        # P·V in fp16/bf16 → fp32 accumulator (identical in both modes).
        v_ptrs = v_base + n_idx[:, None] * stride_vn + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0)
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = (
        Out + b * stride_ob + hq * stride_oh
        + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_mask[:, None])
