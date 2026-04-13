"""
CacheBlend Recompute
====================
An alternative to SelectiveRecompute that fixes stale KV tensors
*statistically* rather than *spatially*.

Problem (same as SelectiveRecompute)
------------------------------------
When independently-cached chunks are stitched together, their KV tensors
are "stale" — each chunk was encoded without cross-chunk attention context.

Why SelectiveRecompute is suboptimal
------------------------------------
SelectiveRecompute blindly recomputes the last R tokens at every chunk seam.
This is both too broad (recomputes tokens that may not actually deviate) and
too narrow (misses tokens inside a chunk that depend on cross-chunk context).
In practice it costs ~79% of full prefill (Experiment 2: 553ms vs 703ms).

CacheBlend approach
-------------------
1. Concatenate all cached KV chunks into one assembled tensor
2. Run a single forward pass over the cached token IDs with the assembled
   KV as context — this produces "what KV *would* be with full attention"
3. Compare the updated KV against the cached KV per token per layer using
   cosine distance — tokens that deviate significantly are the ones that
   actually need fixing
4. Recompute only the top-k% most deviated tokens (default 15%)

The result: ~15% of tokens recomputed instead of O(R * num_seams), and
the tokens chosen are the ones that actually matter for output quality.

Reference: CacheBlend (USENIX ATC '25)

Trade-off knobs
---------------
recompute_ratio : fraction of tokens to recompute (default 0.15 = 15%)
                  Higher → better quality, more compute.
                  Lower  → faster, slight quality risk.
min_deviation   : minimum cosine distance to consider a token "deviated".
                  Tokens below this are never recomputed regardless of ratio.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from .models import PastKVType, AssembledPrompt
from .cache_manager import KVCacheManager

log = logging.getLogger(__name__)


class CacheBlendRecompute:
    def __init__(
        self,
        recompute_ratio: float = 0.15,
        min_deviation: float = 0.01,
        device: str = "cpu",
    ):
        self.recompute_ratio = recompute_ratio
        self.min_deviation = min_deviation
        self.device = device

    def apply(
        self,
        assembled: AssembledPrompt,
        model,
    ) -> AssembledPrompt:
        """
        Fix stale KV tensors by deviation-guided selective recomputation.
        Same interface as SelectiveRecompute.apply().
        """
        if assembled.cached_past_kv is None:
            return assembled
 
        if assembled.cached_length == 0:
            return assembled

        new_kv = self._deviation_recompute(
            assembled.full_token_ids,
            assembled.cached_past_kv,
            assembled.cached_length,
            model,
        )

        return AssembledPrompt(
            full_token_ids=assembled.full_token_ids,
            cached_past_kv=new_kv,
            cached_length=assembled.cached_length,
            live_token_ids=assembled.live_token_ids,
            live_position_ids=assembled.live_position_ids,
            chunk_boundaries=assembled.chunk_boundaries,
            cache_hit_ratio=assembled.cache_hit_ratio,
        )

    def _deviation_recompute(
        self,
        full_token_ids: List[int],
        cached_kv: PastKVType,
        cached_length: int,
        model,
    ) -> PastKVType:
        """
        Core CacheBlend algorithm:
        1. Forward pass with assembled KV to get updated KV
        2. Measure per-token deviation (cosine distance on K tensors)
        3. Select top-k% deviated positions
        4. Recompute only those positions
        """
        model_device = next(model.parameters()).device
        cached_tokens = full_token_ids[:cached_length]

        # ── Step 1: cheap forward pass to get full-context KV ──────
        input_ids = torch.tensor(
            [cached_tokens], dtype=torch.long, device=model_device
        )
        pos_ids = torch.arange(
            0, cached_length, dtype=torch.long, device=model_device
        ).unsqueeze(0)

        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                position_ids=pos_ids,
                use_cache=True,
            )

        updated_kv = self._extract_kv(out.past_key_values)

        # ── Step 2: compute per-token deviation across layers ──────
        num_layers = len(cached_kv)
        # Deviation per layer per token: cosine distance on K vectors
        # K shape: [1, num_heads, seq_len, head_dim]
        layer_deviations = []
        for layer_idx in range(num_layers):
            cached_k = cached_kv[layer_idx][0].to(model_device)
            updated_k = updated_kv[layer_idx][0].to(model_device)

            # Flatten heads into feature dim: [1, seq_len, heads * head_dim]
            seq_len = cached_k.shape[2]
            ck = cached_k.permute(0, 2, 1, 3).reshape(1, seq_len, -1)
            uk = updated_k.permute(0, 2, 1, 3).reshape(1, seq_len, -1)

            # Cosine distance per token position
            cos_sim = F.cosine_similarity(ck, uk, dim=-1)  # [1, seq_len]
            dev = 1.0 - cos_sim  # 0 = identical, 2 = opposite
            layer_deviations.append(dev)

        # Average deviation across all layers → [1, seq_len]
        mean_dev = torch.stack(layer_deviations, dim=0).mean(dim=0)
        mean_dev = mean_dev.squeeze(0)  # [seq_len]

        # ── Step 3: select positions that need recomputation ───────
        num_tokens = mean_dev.shape[0]
        num_to_recompute = max(1, int(num_tokens * self.recompute_ratio))

        # Apply minimum deviation threshold
        above_min = mean_dev > self.min_deviation
        if not above_min.any():
            log.debug("CacheBlend: no tokens above min_deviation=%.3f, skipping recompute", self.min_deviation)
            return cached_kv

        # Top-k most deviated among those above threshold
        masked_dev = mean_dev.clone()
        masked_dev[~above_min] = -1.0
        _, top_indices = masked_dev.topk(min(num_to_recompute, above_min.sum().item()))
        top_indices = top_indices.sort().values  # sort by position for locality

        recompute_pct = len(top_indices) / num_tokens * 100
        log.debug(
            "CacheBlend: recomputing %d/%d tokens (%.1f%%), max_dev=%.4f, mean_dev=%.4f",
            len(top_indices), num_tokens, recompute_pct,
            mean_dev.max().item(), mean_dev[above_min].mean().item(),
        )

        # ── Step 4: patch cached KV with updated values at selected positions
        patched_kv = []
        for layer_idx in range(num_layers):
            cached_k = cached_kv[layer_idx][0].clone()
            cached_v = cached_kv[layer_idx][1].clone()
            updated_k_layer = updated_kv[layer_idx][0].to(self.device)
            updated_v_layer = updated_kv[layer_idx][1].to(self.device)

            # Patch only the deviated positions
            idx = top_indices.to(self.device)
            cached_k[:, :, idx, :] = updated_k_layer[:, :, idx, :]
            cached_v[:, :, idx, :] = updated_v_layer[:, :, idx, :]

            patched_kv.append((cached_k, cached_v))

        return tuple(patched_kv)

    @staticmethod
    def _extract_kv(past_key_values) -> PastKVType:
        """Normalize HF past_key_values to tuple of (K, V) on CPU."""
        if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
            return tuple(
                (k.cpu(), v.cpu())
                for k, v in zip(past_key_values.key_cache, past_key_values.value_cache)
            )
        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            return tuple((layer[0].cpu(), layer[1].cpu()) for layer in legacy)
        return tuple((layer[0].cpu(), layer[1].cpu()) for layer in past_key_values)
