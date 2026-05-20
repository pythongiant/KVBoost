# src/kvboost/speculative/bridge.py

"""Glue layer between ``KVBoost._decode_with_kv`` and ``SpeculativeEngine``.

This is the single function the engine calls when ``self.speculative_engine``
is set. Kept separate from ``KVBoost`` so the dispatch is testable in
isolation against a mock ``SpeculativeEngine``.

Cache-commit invariant
----------------------
The returned ``target_past_kv`` has length ``cached_length - 1 +
len(generated)``. KVBoost's existing chunk-commit path
(``_store_prompt_chunks`` at engine.py:806) re-encodes from token IDs
rather than reading the live ``past_kv``, so the chunk cache correctness
depends only on the *committed token sequence* — not on the precise
final KV state. Speculative rollback guarantees the token sequence is
identical to what a non-speculative run would have produced (in greedy
mode) or distributionally equivalent (in sampling mode).
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Tuple

from .engine import SpeculativeEngine


def run_speculative_decode(
    *,
    full_token_ids: List[int],
    target_past_kv: Any,
    cached_length: int,
    spec_engine: SpeculativeEngine,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    on_token: Optional[Callable[[int], None]] = None,
) -> Tuple[List[int], Any]:
    """Drive ``SpeculativeEngine.decode_from`` from a prefilled state.

    Parameters
    ----------
    full_token_ids: the full input prompt as a list of token ids (target
        has already been prefilled on these)
    target_past_kv: post-prefill (and post-CacheBlend, if applicable)
        target KV cache. Length must equal ``cached_length``.
    cached_length: ``len(full_token_ids)`` — sanity-checked.
    spec_engine: the engine to drive
    max_new_tokens: hard cap on tokens to generate this call
    eos_token_id: stop early on this id (None = no early stop)
    on_token: per-COMMITTED-token callback for SSE streaming. Never
        fires on drafted-but-rejected tokens.

    Returns
    -------
    (generated, target_past_kv): generated is the list of token ids
    appended this call. ``target_past_kv`` length matches the baseline
    decode contract: ``cached_length - 1 + len(generated)``.
    """
    if cached_length != len(full_token_ids):
        raise ValueError(
            f"cached_length ({cached_length}) != len(full_token_ids) "
            f"({len(full_token_ids)}); target must be fully prefilled"
        )

    return spec_engine.decode_from(
        prompt_ids=full_token_ids,
        target_past_kv=target_past_kv,
        cached_length=cached_length,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        on_token=on_token,
    )
