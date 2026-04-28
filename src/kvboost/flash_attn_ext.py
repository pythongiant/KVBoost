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


# ── Paged attention layer interceptor ────────────────────────────────────────
#
# Same monkey-patch approach as install_flash_attention, but the replacement
# SDPA reads historical K/V directly from the BlockAllocator pool rather than
# from a reconstructed contiguous past_key_values tensor.
#
# How it works during a single-token decode step
# -----------------------------------------------
# 1. model.forward() is called with input_ids=[new_token], NO past_key_values.
# 2. Inside each attention layer the model computes Q, K_new, V_new for that
#    one token via its projection matrices.
# 3. Our patched SDPA intercepts the call and receives:
#      Q      : [1, Hq, 1,      D]   (new query)
#      K_new  : [1, Hkv, 1,     D]   (new key — just projected, not yet stored)
#      V_new  : [1, Hkv, 1,     D]   (new value)
# 4. We write K_new/V_new into the block pool at the current slot.
# 5. We gather the full K/V history [0..seq_len] from the pool.
# 6. We run paged_attention_fwd(Q, gathered_K, gathered_V) and return the
#    result — attention over the full context without ever building a
#    contiguous past_key_values tensor.
#
# The interceptor is installed just before model.forward() and uninstalled
# immediately after, so it does not affect warm() or prefill.

_PAGED_ATTR = "_kvboost_paged"

# Name used for the legacy SDPA global (transformers ≤ 4.x)
_LEGACY_SDPA_NAME = "scaled_dot_product_attention"


def _get_attn_registry(model: torch.nn.Module):
    """Return the ALL_ATTENTION_FUNCTIONS registry dict if the model uses one
    (transformers 5.x), otherwise None."""
    for _, module in model.named_modules():
        fwd = getattr(module, "forward", None)
        if fwd is None:
            continue
        g = getattr(fwd, "__globals__", {})
        reg = g.get("ALL_ATTENTION_FUNCTIONS")
        if reg is not None and hasattr(reg, "get_interface"):
            return reg
    return None


def _make_paged_attn_fn(layer_idx: int, state: dict, sdpa_style: bool):
    """
    Return a replacement for the model's attention dispatch function.

    sdpa_style=True  : signature matches transformers 5.x sdpa/eager registry fns
                        (module, query, key, value, attention_mask, dropout, scaling, ...)
                       K/V already contains the FULL history (past + new token).

    sdpa_style=False : signature matches torch.nn.functional.scaled_dot_product_attention
                        (query, key, value, attn_mask, dropout_p, is_causal, scale)
                       K/V contains only the NEW token(s).
    """
    from .cpu_paged.paged_attn_cpu import append_kv_to_blocks, paged_attention_fwd

    if sdpa_style:
        def paged_fn(
            module,
            query,          # [B, Hq, S_q, D]
            key,            # [B, Hkv, S_kv, D]  — may be only new tokens when no past cache
            value,          # [B, Hkv, S_kv, D]
            attention_mask=None,
            dropout=0.0,
            scaling=None,
            is_causal=None,
            **kwargs,
        ):
            allocator  = state["allocator"]
            write_slot = state["_write_slot"]

            if scaling is None:
                scaling = 1.0 / math.sqrt(query.size(-1))

            kv_total = key.size(2)
            if kv_total > write_slot:
                # past_key_values.update() concatenated old+new; extract only new
                k_new = key[:, :, write_slot:, :].squeeze(0).to(allocator.dtype)
                v_new = value[:, :, write_slot:, :].squeeze(0).to(allocator.dtype)
            else:
                # use_cache=False: key contains only the new token(s)
                k_new = key.squeeze(0).to(allocator.dtype)
                v_new = value.squeeze(0).to(allocator.dtype)

            num_new = k_new.size(1)
            new_table, _ = append_kv_to_blocks(
                allocator=allocator,
                layer=layer_idx,
                block_table=list(state["block_table"]),
                slot_offset=write_slot,
                k=k_new,
                v=v_new,
            )
            state["block_table"] = new_table
            full_seq_len = write_slot + num_new

            out = paged_attention_fwd(
                query=query.to(allocator.dtype),
                allocator=allocator,
                block_tables=[new_table],
                seq_lens=[full_seq_len],
                layer=layer_idx,
                causal=(query.size(2) > 1),
                scale=scaling,
            )  # [B, Hq, S, D]

            # Registry fns return (attn_output, attn_weights) where attn_output
            # has already been transposed: [B, S, Hq, D]
            return out.to(query.dtype).transpose(1, 2).contiguous(), None

    else:
        def paged_fn(
            query, key, value,
            attn_mask=None, dropout_p=0.0, is_causal=False, scale=None,
            **kwargs,
        ):
            allocator  = state["allocator"]
            write_slot = state["_write_slot"]

            if scale is None:
                scale = 1.0 / math.sqrt(query.size(-1))

            k_new = key.squeeze(0).to(allocator.dtype)
            v_new = value.squeeze(0).to(allocator.dtype)

            new_table, _ = append_kv_to_blocks(
                allocator=allocator,
                layer=layer_idx,
                block_table=list(state["block_table"]),
                slot_offset=write_slot,
                k=k_new,
                v=v_new,
            )
            state["block_table"] = new_table
            full_seq_len = write_slot + key.size(2)

            out = paged_attention_fwd(
                query=query.to(allocator.dtype),
                allocator=allocator,
                block_tables=[new_table],
                seq_lens=[full_seq_len],
                layer=layer_idx,
                causal=is_causal or (query.size(2) > 1),
                scale=scale,
            )
            return out.to(query.dtype)

    return paged_fn


def install_paged_attention(
    model: torch.nn.Module,
    state: dict,
) -> int:
    """
    Monkey-patch every attention module in *model* so that its attention call
    reads K/V from the BlockAllocator pool rather than from past_key_values.

    Works with:
    - transformers 5.x: temporarily replaces the active entry in the
      ALL_ATTENTION_FUNCTIONS registry (e.g. 'sdpa') with a paged version.
    - transformers ≤ 4.x: swaps scaled_dot_product_attention in the module's
      global namespace.

    Parameters
    ----------
    model : the HuggingFace model
    state : mutable dict with keys:
                allocator   : BlockAllocator
                block_table : List[int]   (updated in place during forward)
                seq_len     : int
                _write_slot : int         (the logical slot for the next write)

    Returns the number of modules patched.
    """
    registry = _get_attn_registry(model)
    patched = 0
    layer_idx = 0

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if not any(cls_name.endswith(s) for s in _ATTN_MODULE_SUFFIXES):
            continue
        if hasattr(module, _PAGED_ATTR):
            continue

        original_forward = module.forward
        module.__dict__[_PAGED_ATTR] = layer_idx
        module.__dict__["_paged_original_forward"] = original_forward

        def _make_forward(orig_fwd, lidx):
            if registry is not None:
                # transformers 5.x: swap inside the shared registry
                impl_key = getattr(model.config, "_attn_implementation", "eager")
                def paged_forward(self, *args, **kwargs):
                    old_fn = registry.get(impl_key)
                    registry[impl_key] = _make_paged_attn_fn(lidx, state, sdpa_style=True)
                    try:
                        return orig_fwd(*args, **kwargs)
                    finally:
                        if old_fn is None:
                            registry.pop(impl_key, None)
                        else:
                            registry[impl_key] = old_fn
            else:
                # transformers ≤ 4.x: swap in module globals
                def paged_forward(self, *args, **kwargs):
                    g = orig_fwd.__globals__
                    old_fn = g.get(_LEGACY_SDPA_NAME)
                    g[_LEGACY_SDPA_NAME] = _make_paged_attn_fn(lidx, state, sdpa_style=False)
                    try:
                        return orig_fwd(*args, **kwargs)
                    finally:
                        if old_fn is None:
                            g.pop(_LEGACY_SDPA_NAME, None)
                        else:
                            g[_LEGACY_SDPA_NAME] = old_fn
            return paged_forward

        module.forward = types.MethodType(_make_forward(original_forward, layer_idx), module)
        layer_idx += 1
        patched += 1
        log.debug("paged_attn: patched layer %d (%s %s)", layer_idx - 1, name, cls_name)

    log.debug("paged_attn: patched %d attention modules", patched)
    return patched


def uninstall_paged_attention(model: torch.nn.Module) -> int:
    """Remove paged attention patches, restoring original forwards."""
    restored = 0
    for module in model.modules():
        if hasattr(module, "_paged_original_forward"):
            module.forward = module.__dict__.pop("_paged_original_forward")
            module.__dict__.pop(_PAGED_ATTR, None)
            restored += 1
    return restored
