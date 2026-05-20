# src/kvboost/speculative/verifier.py

"""Target-model verifier for speculative decoding.

Takes the draft's proposed tokens plus the last committed token, runs
ONE multi-token forward on the target model, and returns logits at each
of the K+1 positions. The K+1-th logit is the "bonus" — what comes after
consuming all K accepted draft tokens.

The streaming pipeline's per-layer hooks fire on every forward
regardless of seq_len, so a (1, K+1) input passes through unchanged.
This is exactly the same shape as the warm-up prefill in
``demo_partial_8b``.

KV rollback is delegated to :func:`rollback.truncate_past_kv`. The
verifier is otherwise stateless — KV ownership lives with the engine,
which threads ``past_kv`` in and out per round.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ..compat import last_logit_only
from .rollback import truncate_past_kv


class TargetVerifier:
    """Single-forward verifier over the target's streaming model.

    Stateless w.r.t. KV: ``verify(past_kv)`` returns the extended
    ``past_kv`` and the caller stores it. This keeps the engine in
    charge of rollback and prevents two writers fighting over the same
    cache reference.
    """

    def __init__(self, target_model: Any, device: Optional[str] = None) -> None:
        self.model = target_model
        # Resolve device from the model if not provided. StreamingCausalLM
        # forwards through hf_model; use the hf_model's first param.
        if device is not None:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(target_model.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

    @torch.inference_mode()
    def verify(
        self,
        last_committed_token: int,
        draft_ids: torch.Tensor,
        past_kv: Any,
        committed_length: int,
    ) -> Tuple[torch.Tensor, Any]:
        """Run one multi-token forward over [last_committed_token] + draft_ids.

        Parameters
        ----------
        last_committed_token: the most recently committed token; this is
            the position at which the *first* verification slot sits
        draft_ids: shape ``(K,)`` int64 — the K proposed tokens
        past_kv: target's current KV cache of length ``committed_length``
        committed_length: the absolute seq position of ``last_committed_token``;
            ``past_kv`` MUST already include it. We pass it explicitly to set
            ``position_ids`` correctly, which matters for RoPE-based models.

        Returns
        -------
        logits: shape ``(K+1, V)`` — logits at each of the K+1 positions.
            ``logits[i]`` is the distribution that should follow having
            committed ``[last_committed_token, draft_ids[0], ..., draft_ids[i-1]]``.
        new_past_kv: the target's KV cache after this forward, of length
            ``committed_length + K + 1``. Caller will truncate via
            ``rollback`` once the sampler decides how many to accept.
        """
        if draft_ids.dim() != 1:
            raise ValueError(
                f"draft_ids must be 1-D, got shape {tuple(draft_ids.shape)}"
            )
        k = int(draft_ids.shape[0])
        if k < 1:
            raise ValueError(f"draft_ids must have at least 1 element, got {k}")

        # Build [last_committed_token] + draft_ids as (1, K+1).
        # The K+1 positions correspond to absolute positions
        # [committed_length, committed_length+1, ..., committed_length+K].
        # ``past_kv`` already covers [0, committed_length); the new forward
        # writes positions [committed_length, committed_length+K].
        seq = torch.empty((1, k + 1), dtype=torch.long, device=self.device)
        seq[0, 0] = last_committed_token
        seq[0, 1:] = draft_ids.to(self.device)

        pos_ids = torch.arange(
            committed_length,
            committed_length + k + 1,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)  # (1, K+1)

        # IMPORTANT: do NOT wrap in last_logit_only — we need ALL K+1
        # positions' logits, not just the final one.
        out = self.model(
            input_ids=seq,
            past_key_values=past_kv,
            position_ids=pos_ids,
            use_cache=True,
        )
        # out.logits: (1, K+1, V). Strip the batch dim.
        logits = out.logits[0]  # (K+1, V)
        return logits, out.past_key_values

    def rollback(self, past_kv: Any, keep_n: int) -> Any:
        """Truncate ``past_kv`` to the first ``keep_n`` positions.

        Thin wrapper over :func:`rollback.truncate_past_kv` so the engine
        can call ``verifier.rollback(...)`` symmetrically with
        ``draft.rollback(...)``.
        """
        return truncate_past_kv(past_kv, keep_n)
