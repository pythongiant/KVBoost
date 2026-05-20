# src/kvboost/speculative/engine.py

"""Speculative decoding orchestrator.

State invariants
----------------
At the boundary of every round:
- ``committed_length`` is the number of positions covered by the target's
  ``past_kv`` (i.e. the seq_len of past_kv).
- ``last_committed`` is the token that sits at position
  ``committed_length`` — the next token to be re-fed into the model. It
  is NOT present in ``past_kv``; it's the boundary token.
- The draft's ``past_kv`` also covers exactly ``committed_length``
  positions, mirroring the target.

This matches the baseline decode contract in
:meth:`InferenceEngine._decode_with_kv`: past_kv length equals
``len(prefill) + len(generated) - 1``, with the final generated token
held in a separate Python variable (here ``last_committed``).

Per round
---------
1. ``draft_ids, draft_probs = draft.draft(last_committed, K)``
   Draft autoregressively emits K tokens, growing its past_kv by K.
2. ``logits, target_past_kv = verifier.verify(last_committed, draft_ids, ...)``
   Target runs ONE forward over [last_committed] + draft_ids (K+1 tokens).
   target_past_kv grows by K+1.
3. ``accepted, output_tok = sampler(...)``
   - ``accepted`` ∈ [0, K]: how many draft prefix tokens to keep
   - ``output_tok`` is the bonus (if accepted == K) or correction (else)
4. Commit ``accepted`` draft tokens + ``output_tok`` (total accepted+1),
   firing ``on_token`` per commit and breaking on EOS / token budget.
5. Roll back both past_kvs to length
   ``committed_length + 1 + n_committed_draft_tokens`` where
   ``n_committed_draft_tokens = min(n_committed, accepted)``. The +1 is
   the slot now occupied by what *was* ``last_committed``; the
   correction/bonus is not in KV (it was sampled, not fed) and becomes
   the next round's ``last_committed``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Tuple

import torch

from ..cache_manager import KVCacheManager
from .config import SpeculativeConfig
from .draft import DraftModel
from .sampler import verify_greedy, verify_sampling
from .stats import SpeculativeStats
from .verifier import TargetVerifier

log = logging.getLogger(__name__)


def _kv_length(past_kv: Any) -> int:
    """Get seq length of either DynamicCache or tuple-of-tuples KV."""
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return past_kv.get_seq_length()
    return KVCacheManager.kv_seq_len(past_kv)


class SpeculativeEngine:
    """Drives the speculative decode loop after the target has been prefilled.

    The engine does NOT handle prefill itself — that's still
    ``KVBoost._decode_with_kv``'s job. The engine takes the post-prefill
    target ``past_kv`` and runs the decode loop from there.
    """

    def __init__(
        self,
        cfg: SpeculativeConfig,
        target_verifier: TargetVerifier,
        draft_model: DraftModel,
        stats: Optional[SpeculativeStats] = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.verifier = target_verifier
        self.draft = draft_model
        self.stats = stats or SpeculativeStats()

    # ── Public entry point ───────────────────────────────────────────────────

    @torch.inference_mode()
    def decode_from(
        self,
        prompt_ids: List[int],
        target_past_kv: Any,
        cached_length: int,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> Tuple[List[int], Any]:
        """Run the speculative decode loop until EOS or token budget.

        Returns ``(generated, target_past_kv)``. The returned past_kv has
        length ``cached_length - 1 + len(generated)`` — the same boundary
        convention as the baseline decode loop, so ``KVBoost`` can store
        it (or pass it to the chunk-commit path) unchanged.
        """
        if len(prompt_ids) != cached_length:
            raise ValueError(
                f"cached_length={cached_length} must equal len(prompt_ids)="
                f"{len(prompt_ids)} — the target must be fully prefilled "
                "before speculative decode begins"
            )
        if max_new_tokens <= 0:
            return [], target_past_kv

        # ── Set up the boundary state ───────────────────────────────────
        # Prefill ended with past_kv covering the WHOLE prompt. To match
        # our state invariant ("last_committed is at position
        # committed_length, NOT in past_kv"), we roll back by 1 so the
        # final prompt token becomes the next-to-be-fed boundary token.
        last_committed = prompt_ids[-1]
        target_past_kv = self.verifier.rollback(
            target_past_kv, cached_length - 1
        )
        committed_length = cached_length - 1

        # Prime the draft on the prompt MINUS the last token (we'll feed
        # it via draft.draft like the target does).
        prompt_minus_one = prompt_ids[:cached_length - 1]
        if prompt_minus_one:
            prompt_t = torch.tensor(
                [prompt_minus_one], dtype=torch.long, device=self.draft.device
            )
            self.draft.prime(prompt_t)
        else:
            # Degenerate single-token prompt; draft has no prefix to prime on.
            # First draft.draft(last_committed, K) will populate KV from scratch.
            self.draft._past_kv = None
            self.draft._primed_length = 0

        generated: List[int] = []

        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            k = min(self.cfg.draft_k, remaining)
            if k < 1:
                break

            # ── 1. Draft proposes K tokens ───────────────────────────
            # If draft.past_kv is None (single-token prompt case), draft()
            # will fail; prime with an empty prefix wouldn't help either.
            # Fall back to a fresh prime here.
            if self.draft.past_kv is None:
                prompt_t = torch.tensor(
                    [[last_committed]], dtype=torch.long, device=self.draft.device
                )
                self.draft.prime(prompt_t)
                # Now past_kv covers last_committed; roll back by 1 so the
                # invariant holds (last_committed at boundary, not in KV).
                self.draft.rollback(0)

            draft_ids, draft_probs = self.draft.draft(last_committed, k=k)
            # After draft: draft.past_kv length = committed_length + k

            # ── 2. Target verifies in one multi-token forward ───────────
            target_logits, target_past_kv = self.verifier.verify(
                last_committed_token=last_committed,
                draft_ids=draft_ids,
                past_kv=target_past_kv,
                committed_length=committed_length,
            )
            # After verify: target_past_kv length = committed_length + k + 1

            # ── 3. Sampler decides ──────────────────────────────────────
            if self.cfg.mode == "greedy":
                accepted, output_tok = verify_greedy(draft_ids, target_logits)
            else:
                accepted, output_tok = verify_sampling(
                    draft_ids,
                    draft_probs,
                    target_logits,
                    temperature=self.cfg.temperature,
                )

            # ── 4. Commit accepted draft prefix + correction/bonus ──────
            # Each commit can early-stop on EOS or token budget.
            committed_this_round: List[int] = []
            stop = False
            for i in range(accepted):
                tok = int(draft_ids[i].item())
                generated.append(tok)
                committed_this_round.append(tok)
                if on_token is not None:
                    on_token(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    stop = True
                    break
                if len(generated) >= max_new_tokens:
                    stop = True
                    break
            if not stop:
                tok = int(output_tok)
                generated.append(tok)
                committed_this_round.append(tok)
                if on_token is not None:
                    on_token(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    stop = True

            n_committed = len(committed_this_round)
            # n_committed_draft_tokens: how many of the committed entries
            # were DRAFT tokens (which DID land in past_kv via verify).
            # The trailing output_tok was sampled, not fed — it never
            # landed in past_kv. So the count of draft tokens we kept is
            # min(n_committed, accepted): everything except the trailing
            # correction/bonus IF we got that far.
            n_committed_draft_tokens = min(n_committed, accepted)

            # ── 5. Rollback both KV caches to match committed state ─────
            new_kv_length = committed_length + 1 + n_committed_draft_tokens
            target_past_kv = self.verifier.rollback(target_past_kv, new_kv_length)

            # Draft KV after draft.draft has length committed_length + k.
            # We want it at new_kv_length. Since new_kv_length <=
            # committed_length + k + 1 always (accepted <= k, and there's
            # the +1 for last_committed), rollback truncates.
            # Special case: when accepted == k AND we committed all k+1
            # tokens, new_kv_length = committed_length + 1 + k, which
            # EXCEEDS the draft's current length (= committed_length + k)
            # by 1. We can't extend on rollback — but the missing slot
            # corresponds to output_tok which the draft will re-consume
            # as last_committed at the start of the next round. So we
            # only rollback if new_kv_length <= current draft length;
            # otherwise leave it (next round's draft.draft will extend
            # naturally).
            cur_draft_len = _kv_length(self.draft.past_kv)
            if new_kv_length < cur_draft_len:
                self.draft.rollback(new_kv_length)
            # else: leave draft KV as-is. The next round's draft.draft
            # will feed the new last_committed, which lands at the
            # current draft KV's tail — matching the new state.

            # ── 6. Stats + advance ──────────────────────────────────────
            self.stats.record_round(accepted_count=accepted, draft_k=k)

            committed_length = new_kv_length
            if n_committed > 0:
                last_committed = generated[-1]
            else:
                # Should be impossible — we always commit at least the
                # correction/bonus unless we early-stopped at the very
                # first draft token (which we DID append).
                log.warning("speculative round committed 0 tokens; breaking")
                break

            if stop:
                break

        return generated, target_past_kv
