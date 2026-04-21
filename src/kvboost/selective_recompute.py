"""
SelectiveRecompute
==================
Problem
-------
When we concatenate KV caches from independently-cached chunks,
each chunk's stored KV was computed with attention only over its own
tokens (and whatever prefix preceded it at cache-time). After stitching
chunks together the seam positions have "stale" KV — they didn't attend
to cross-chunk context.

This matters primarily for the *keys and values* of the boundary tokens
(not the queries, which are recomputed fresh for every generation step).

Fix
---
For the last R tokens of each chunk boundary (except the final chunk),
re-run a forward pass using the full merged KV as prefix context and
replace those R positions in the stitched KV with freshly computed ones.

This is "selective" because we only recompute O(R * num_boundaries)
tokens rather than the full prompt.

Trade-off knobs
---------------
recompute_overlap   : R — how many tokens back from each seam to fix.
                      Larger → better quality, more compute.
                      Default 16 is a good balance for chunk_size >= 64.
skip_if_no_seams    : skip the module entirely for pure prefix hits
                      (no stitching was done).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch

from .models import PastKVType, AssembledPrompt
from .cache_manager import KVCacheManager
from .compat import default_device, logits_to_keep_kwargs

log = logging.getLogger(__name__)


class SelectiveRecompute:
    def __init__(
        self,
        recompute_overlap: int = 16,
        skip_if_no_seams: bool = True,
        device: Optional[str] = None,
    ):
        self.overlap = recompute_overlap
        self.skip_if_no_seams = skip_if_no_seams
        self.device = device if device is not None else default_device()

    def apply(
        self,
        assembled: AssembledPrompt,
        model,
    ) -> AssembledPrompt:
        """
        Optionally fix the KV seams in assembled.cached_past_kv.
        Returns a (possibly modified) AssembledPrompt.
        model must be a HuggingFace CausalLM.
        """
        boundaries = assembled.chunk_boundaries
        # Only seams between chunks need fixing (not the final boundary)
        seams = boundaries[:-1]  # each seam is the END of a non-final chunk

        if self.skip_if_no_seams and not seams:
            return assembled  # pure prefix — no stitching seams

        if assembled.cached_past_kv is None:
            return assembled

        new_kv = self._recompute_seams(
            assembled.full_token_ids,
            assembled.cached_past_kv,
            seams,
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

    # ------------------------------------------------------------------
    # Core recomputation
    # ------------------------------------------------------------------

    def _recompute_seams(
        self,
        full_token_ids: List[int],
        merged_kv: PastKVType,
        seams: List[Tuple[int, int]],
        model,
    ) -> PastKVType:
        """
        For each (start, end) seam in `seams`, re-encode the last
        `self.overlap` tokens of that seam position using the
        preceding merged_kv as context.

        Returns updated merged_kv with those positions patched.
        """
        current_kv = merged_kv

        for _start, end_pos in seams:
            recompute_start = max(0, end_pos - self.overlap)
            recompute_tokens = full_token_ids[recompute_start:end_pos]

            if not recompute_tokens:
                continue

            # Prefix = everything before recompute_start
            prefix_kv = KVCacheManager.slice_kv(current_kv, 0, recompute_start)
            prefix_len = recompute_start

            # Build inputs
            input_ids = torch.tensor(
                [recompute_tokens], dtype=torch.long, device=self.device
            )
            pos_ids = torch.arange(
                prefix_len, prefix_len + len(recompute_tokens),
                dtype=torch.long, device=self.device
            ).unsqueeze(0)

            # Move prefix_kv to model device
            model_device = next(model.parameters()).device
            from transformers import DynamicCache as _DynCache
            prefix_kv_dev = _DynCache()
            for layer_k, layer_v in prefix_kv:
                prefix_kv_dev.update(
                    layer_k.to(model_device), layer_v.to(model_device),
                    len(prefix_kv_dev),
                )
            input_ids = input_ids.to(model_device)
            pos_ids = pos_ids.to(model_device)

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    past_key_values=prefix_kv_dev,
                    position_ids=pos_ids,
                    use_cache=True,
                    **logits_to_keep_kwargs(model),
                )

            # Normalize DynamicCache → plain tuple[(key, val), ...]
            raw_kv = out.past_key_values
            if hasattr(raw_kv, "layers"):
                raw_kv = tuple((l.keys, l.values) for l in raw_kv.layers)
            elif hasattr(raw_kv, "key_cache") and hasattr(raw_kv, "value_cache"):
                raw_kv = tuple(zip(raw_kv.key_cache, raw_kv.value_cache))
            else:
                raw_kv = tuple((layer[0], layer[1]) for layer in raw_kv)

            fresh_kv = tuple(
                (layer[0].to(self.device), layer[1].to(self.device))
                for layer in raw_kv
            )

            # fresh_kv now covers [0, recompute_start + overlap)
            # Splice it back: take fresh_kv up to end_pos, then append
            # whatever was in current_kv after end_pos
            tail_kv = KVCacheManager.slice_kv(current_kv, end_pos, KVCacheManager.kv_seq_len(current_kv))

            if KVCacheManager.kv_seq_len(tail_kv) > 0:
                current_kv = KVCacheManager.merge_kv_list([fresh_kv, tail_kv])
            else:
                current_kv = fresh_kv

            log.debug(
                "Recomputed seam at pos %d–%d (%d tokens)",
                recompute_start, end_pos, len(recompute_tokens),
            )

        return current_kv
