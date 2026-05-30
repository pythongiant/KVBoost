"""Target verifier for tree speculative decoding.

One forward over the flattened tree with a custom 4-D attention mask.
Produces per-node logits ``(N, V)`` that the sampler walks during the
acceptance phase.

Reuses the existing flat ``TargetVerifier``'s model + device handling
by composition — no inheritance — so changes to ``TargetVerifier``
don't break tree mode silently.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple

import torch

from ..rollback import truncate_past_kv
from .structure import (
    DraftTree,
    build_tree_attention_mask,
    build_tree_position_ids,
    flatten_tree_input_ids,
)

log = logging.getLogger(__name__)


class TreeVerifier:
    """Tree-aware target verifier. Mirrors ``TargetVerifier``'s shape.

    Stateless w.r.t. KV. ``verify_tree`` writes ``N`` new columns to
    the cache (one per tree node, including the root); the caller's
    ``commit_path`` reconciles which columns survive.
    """

    def __init__(self, target_model: Any, device: Any = None) -> None:
        self.model = target_model
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(target_model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @torch.no_grad()
    def verify_tree(
        self,
        tree: DraftTree,
        past_kv: Any,
        committed_length: int,
    ) -> Tuple[torch.Tensor, Any]:
        """Run one tree-aware forward.

        ``tree``'s node 0 is the synthetic root holding the last
        committed token; it's at absolute position ``committed_length``
        (same as the flat verifier's first slot). Drafted nodes are at
        positions ``committed_length + depth[i]`` — multiple nodes can
        share a position if they're siblings.

        Returns
        -------
        per_node_logits: shape ``(N, V)`` — logits at each tree node.
            ``per_node_logits[i]`` is the distribution that should
            follow having committed the path from root through node i
            (inclusive).
        new_past_kv: target's KV cache after the forward; has
            ``committed_length + N`` columns. Caller will pass the
            accepted-path indices to ``commit_path_target_kv`` to
            collapse it to the contiguous committed prefix.
        """
        if tree.n_nodes < 1:
            raise ValueError("verify_tree called on empty tree")
        if committed_length < 0:
            raise ValueError(
                f"committed_length must be >= 0, got {committed_length}"
            )

        input_ids = flatten_tree_input_ids(tree, device=self.device)
        position_ids = build_tree_position_ids(
            tree, committed_length=committed_length, device=self.device,
        )

        # Build the additive attention mask in the model's preferred
        # dtype. Try fp16 first (matches model param dtype); fall back
        # to fp32 if anything blocks the cast.
        try:
            model_dtype = next(self.model.parameters()).dtype
        except StopIteration:
            model_dtype = torch.float32
        attn_mask_dtype = (
            model_dtype if model_dtype.is_floating_point else torch.float32
        )
        attention_mask = build_tree_attention_mask(
            tree,
            committed_length=committed_length,
            device=self.device,
            dtype=attn_mask_dtype,
        )

        out = self.model(
            input_ids=input_ids,
            past_key_values=past_kv,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        per_node_logits = out.logits[0]   # (N, V)
        return per_node_logits, out.past_key_values

    def rollback(self, past_kv: Any, keep_n: int) -> Any:
        """Drop everything past ``keep_n``. Used when the engine
        decides to abort the tree and fall back to a smaller commit
        (e.g. all-rejected sampling mode)."""
        return truncate_past_kv(past_kv, keep_n)
