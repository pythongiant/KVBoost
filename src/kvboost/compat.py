"""
Model Compatibility
===================
KVBoost's position-ID stitching assumes RoPE positional encoding where
positions are passed explicitly via position_ids and the model correctly
handles non-contiguous sequences.

This module tracks which HuggingFace architectures are known-safe, which
are known-broken, and provides runtime validation for untested models.

Known-broken architectures:
  - ALiBi models (MPT, Falcon): positional bias added directly to attention
    scores based on distance — no position_ids injection possible
  - Learned absolute embeddings (GPT-2): position info baked into token
    representations at the input layer, irreversible by KV stitching
  - Sliding window attention (Mistral with sliding_window != None): KV from
    outside the window includes tokens that should have been masked
"""

from __future__ import annotations

import inspect
import logging
import warnings
from typing import Optional

import torch

log = logging.getLogger(__name__)


_logits_kwarg_cache: dict = {}


def logits_to_keep_kwargs(model) -> dict:
    """
    Return {"logits_to_keep": 1} (or {"num_logits_to_keep": 1}) if the given
    model's forward accepts either, else {}. Used by callers that only need
    the final token's logits (or don't need logits at all) to avoid a
    `[batch, seq_len, vocab]` allocation on the LM head — a 1.5GB+ tensor
    on long prefills with large-vocab models like Qwen2.5.

    The kwarg choice is cached per model type since inspecting signatures
    on every forward is wasteful.

    Note: relying on this kwarg alone is brittle across transformers versions
    and stale bytecode. For hard guarantees use `last_logit_only(model)`.
    """
    cls = type(model)
    if cls in _logits_kwarg_cache:
        key = _logits_kwarg_cache[cls]
    else:
        try:
            params = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        if "logits_to_keep" in params:
            key = "logits_to_keep"
        elif "num_logits_to_keep" in params:
            key = "num_logits_to_keep"
        else:
            key = ""
        _logits_kwarg_cache[cls] = key
    return {key: 1} if key else {}


class _LastTokenHead(torch.nn.Module):
    """
    Wraps an LM head so it projects only the final sequence position on
    the forward pass. Replaces `x → self.inner(x)` with
    `x → self.inner(x[:, -1:, :])`. Downstream callers that only read
    logits[:, -1, :] get identical results. Callers that try to read
    intermediate positions will see wrong values — only use this inside
    `last_logit_only()` context.
    """
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.inner = inner

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() >= 3 and hidden_states.shape[-2] > 1:
            hidden_states = hidden_states[..., -1:, :]
        return self.inner(hidden_states)


class last_logit_only:  # noqa: N801 — context-manager factory
    """
    Context manager that temporarily replaces the model's LM head with a
    last-position-only projection. Guarantees the `[batch, seq_len, vocab]`
    allocation is reduced to `[batch, 1, vocab]`, regardless of
    transformers version or whether `logits_to_keep` is honored.

        with last_logit_only(model):
            out = model(input_ids=...)
            # out.logits.shape == [batch, 1, vocab]

    Works for any HF CausalLM that exposes an `lm_head` attribute. If the
    model has no `lm_head` (e.g. models where the projection lives in a
    submodule), the context manager is a no-op and falls back to whatever
    the forward pass does naturally.
    """
    def __init__(self, model):
        self.model = model
        self._original = None

    def __enter__(self):
        head = getattr(self.model, "lm_head", None)
        if head is None or isinstance(head, _LastTokenHead):
            return self
        self._original = head
        self.model.lm_head = _LastTokenHead(head)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._original is not None:
            self.model.lm_head = self._original
            self._original = None
        return False


def default_device() -> str:
    """Return the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# Architectures verified to work correctly with KV cache stitching
SUPPORTED_ARCHITECTURES = {
    "LlamaForCausalLM",        # RoPE
    "Qwen2ForCausalLM",        # RoPE
    "Qwen2_5ForCausalLM",      # RoPE
    "GemmaForCausalLM",        # RoPE
    "Gemma2ForCausalLM",       # RoPE
    "MistralForCausalLM",      # RoPE (safe only with full attention)
    "PhiForCausalLM",          # RoPE
    "Phi3ForCausalLM",         # RoPE
    "StableLmForCausalLM",     # RoPE
    "InternLMForCausalLM",     # RoPE
    "InternLM2ForCausalLM",    # RoPE
}

# Architectures known to be incompatible, with reasons
UNSUPPORTED_ARCHITECTURES = {
    "GPT2LMHeadModel": (
        "GPT-2 uses learned absolute positional embeddings. Position info is "
        "baked into token representations at the embedding layer — KV cache "
        "stitching cannot correct for position mismatches."
    ),
    "GPTNeoForCausalLM": (
        "GPT-Neo uses learned absolute positional embeddings."
    ),
    "GPTNeoXForCausalLM": (
        "GPT-NeoX uses rotary embeddings but the HF implementation does not "
        "accept position_ids — KV stitching may produce incorrect positions."
    ),
    "MPTForCausalLM": (
        "MPT uses ALiBi positional encoding. Positional bias is added directly "
        "to attention scores based on token distance — there is no position_ids "
        "tensor to inject, so KV cache stitching cannot produce correct positions."
    ),
    "FalconForCausalLM": (
        "Falcon uses ALiBi positional encoding."
    ),
    "BloomForCausalLM": (
        "BLOOM uses ALiBi positional encoding."
    ),
}


def check_model_compatibility(model, strict: bool = True) -> None:
    """
    Validate that a model's architecture is compatible with KV cache stitching.

    Args:
        model: A HuggingFace CausalLM model instance.
        strict: If True (default), raise ValueError for unsupported models
                and warn for untested ones. If False, only warn.

    Raises:
        ValueError: If the model architecture is known to be incompatible.
    """
    arch = type(model).__name__

    # Check for known-broken architectures
    if arch in UNSUPPORTED_ARCHITECTURES:
        reason = UNSUPPORTED_ARCHITECTURES[arch]
        msg = f"KVBoost does not support {arch}: {reason}"
        if strict:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=3)
        return

    # Check for Mistral sliding window
    if arch == "MistralForCausalLM":
        sliding_window = getattr(model.config, "sliding_window", None)
        if sliding_window is not None:
            msg = (
                f"MistralForCausalLM with sliding_window={sliding_window} is not "
                f"supported. KV cache stitching breaks the sliding window mask "
                f"assumption — tokens outside the window that should be invisible "
                f"will be included in the stitched KV."
            )
            if strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=3)
            return

    # Warn for untested architectures
    if arch not in SUPPORTED_ARCHITECTURES:
        warnings.warn(
            f"KVBoost has not been tested with {arch}. Output correctness is "
            f"not guaranteed. Run engine.verify_correctness() to validate "
            f"before trusting cached outputs. Pass strict=False to "
            f"from_pretrained() to suppress this warning.",
            stacklevel=3,
        )
