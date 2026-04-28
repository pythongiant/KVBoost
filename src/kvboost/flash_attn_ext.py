"""
Flash Attention integration for KVBoost.

Three-tier attention fallback:
  1. kvboost._flash_attn_cuda  — custom CUDA kernel (this repo, Ampere+)
  2. torch SDPA flash          — cuDNN FlashAttention via enable_flash_sdp(True)
  3. vanilla SDPA              — default PyTorch, O(N²) memory

install_flash_attention(model) replaces the scaled_dot_product_attention call
inside every supported transformer attention module with the best available
implementation.  The rest of the forward pass (RoPE, projections, output,
past_key_values) is completely unchanged, so KVBoost's position_ids injection
and KV extraction continue to work transparently.

Usage (called automatically by InferenceEngine.__init__):
    from kvboost.flash_attn_ext import install_flash_attention
    install_flash_attention(model)
"""

from __future__ import annotations

import logging
import math
import types
from typing import Optional

import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)

# ── Tier detection ────────────────────────────────────────────────────────────

_TIER: str = "vanilla"
_flash_attn_cuda = None

try:
    from kvboost import _flash_attn_cuda  # type: ignore[attr-defined]
    _TIER = "kvboost_cuda"
    log.info("flash_attn: using kvboost custom CUDA kernel")
except ImportError:
    pass

if _TIER == "vanilla":
    # Check if torch SDPA has a flash implementation available
    try:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            _TIER = "torch_flash"
            log.info("flash_attn: using torch SDPA flash kernel")
    except Exception:
        log.info("flash_attn: falling back to vanilla SDPA (no flash available)")


def flash_attention_available() -> bool:
    """Return True if any accelerated flash attention tier is active."""
    return _TIER != "vanilla"


def get_tier() -> str:
    """Return the active attention tier name."""
    return _TIER


# ── Core attention function ───────────────────────────────────────────────────

def _kvboost_flash_attn(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    scale: float,
    causal: bool = True,
    attn_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Unified attention dispatch.  Q/K/V: [B, H, S, D], all on the same device.
    Returns O: [B, H, S, D].
    """
    if _TIER == "kvboost_cuda" and Q.is_cuda and attn_mask is None:
        # Custom kernel — only handles fp16/bf16; fall through for fp32
        if Q.dtype in (torch.float16, torch.bfloat16):
            O, _L = _flash_attn_cuda.flash_attn_fwd(Q, K, V, scale, causal)
            return O

    if _TIER in ("kvboost_cuda", "torch_flash") and Q.is_cuda:
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            return F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attn_mask,
                scale=scale,
                is_causal=(causal and attn_mask is None),
            )

    # Vanilla fallback (CPU or fp32 on GPU)
    return F.scaled_dot_product_attention(
        Q, K, V,
        attn_mask=attn_mask,
        scale=scale,
        is_causal=(causal and attn_mask is None),
    )


# ── HuggingFace module patching ───────────────────────────────────────────────

# Names of attention sub-modules in supported architectures
_ATTN_MODULE_SUFFIXES = (
    "Attention",          # LlamaAttention, Qwen2Attention, GemmaAttention …
    "SdpaAttention",      # LlamaSdpaAttention (HF ≥ 4.40)
    "FlashAttention2",    # HF built-in flash wrapper — we override anyway
)

# Attribute name used for SDPA inside HF attention modules
_SDPA_ATTR = "_kvboost_attn_fn"


def _make_patched_sdpa(original_sdpa_fn):
    """
    Return a replacement for F.scaled_dot_product_attention that routes through
    _kvboost_flash_attn.  We close over original_sdpa_fn so we can call it as
    the ultimate fallback if shapes are unsupported.
    """
    def patched_sdpa(
        query, key, value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        scale=None,
        **kwargs,
    ):
        if scale is None:
            scale = 1.0 / math.sqrt(query.size(-1))
        # Drop dropout during inference (KVBoost is inference-only)
        return _kvboost_flash_attn(query, key, value, scale, is_causal, attn_mask)

    return patched_sdpa


def _patch_module(module: torch.nn.Module) -> bool:
    """
    Monkey-patch a single attention module to use _kvboost_flash_attn.
    Returns True if the module was patched.

    Strategy: replace the module's reference to F.scaled_dot_product_attention
    by injecting a patched version into its __dict__ (shadowing the global).
    We store the original so we can restore it.
    """
    # Already patched
    if hasattr(module, _SDPA_ATTR):
        return False

    # Stash our custom fn as an attribute; the forward method picks it up via
    # the closure we inject below.
    module.__dict__[_SDPA_ATTR] = _kvboost_flash_attn
    module.__dict__["_original_forward"] = module.forward

    original_forward = module.forward

    def patched_forward(self, *args, **kwargs):
        # Temporarily replace F.scaled_dot_product_attention in the module's
        # global namespace with our version.
        mod_globals = original_forward.__globals__
        old_sdpa = mod_globals.get("scaled_dot_product_attention")
        mod_globals["scaled_dot_product_attention"] = _make_patched_sdpa(old_sdpa)
        try:
            return original_forward(*args, **kwargs)
        finally:
            if old_sdpa is None:
                mod_globals.pop("scaled_dot_product_attention", None)
            else:
                mod_globals["scaled_dot_product_attention"] = old_sdpa

    module.forward = types.MethodType(patched_forward, module)
    return True


def install_flash_attention(model: torch.nn.Module) -> int:
    """
    Walk all sub-modules of *model* and patch attention modules to use the
    best available flash attention implementation.

    Returns the number of modules patched.
    """
    if _TIER == "vanilla":
        log.debug("flash_attn: no accelerated tier available, skipping patch")
        return 0

    patched = 0
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if any(cls_name.endswith(suffix) for suffix in _ATTN_MODULE_SUFFIXES):
            if _patch_module(module):
                patched += 1
                log.debug("flash_attn: patched %s (%s)", name, cls_name)

    if patched:
        log.info("flash_attn: patched %d attention module(s) → tier=%s", patched, _TIER)
    return patched


def uninstall_flash_attention(model: torch.nn.Module) -> int:
    """Restore all patched attention modules to their original forward methods."""
    restored = 0
    for module in model.modules():
        if hasattr(module, "_original_forward"):
            module.forward = module.__dict__.pop("_original_forward")
            module.__dict__.pop(_SDPA_ATTR, None)
            module.__dict__.pop("_kvboost_attn_fn", None)
            restored += 1
    return restored
