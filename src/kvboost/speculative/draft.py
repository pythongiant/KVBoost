# src/kvboost/speculative/draft.py

"""Draft model wrapper for speculative decoding.

Loads a small AWQ model through the same ``StreamingCausalLM`` pipeline
as the target — but defaults to ``residency_mode="full_resident"`` since
draft models (e.g. Qwen2.5-1.5B AWQ at ~1 GB) fit comfortably in spare
VRAM and don't benefit from streaming.

The draft owns its own KV cache; ``rollback`` truncates it after each
verifier round. Tokenizer parity with the target is asserted at
construction — divergent vocabularies would mean the target sees the
"wrong" tokens during verification and acceptance rate would collapse to
zero (or worse, produce semantic corruption silently).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F

from ..compat import default_device, last_logit_only
from .config import SpeculativeConfig
from .rollback import truncate_past_kv

log = logging.getLogger(__name__)


class DraftModel:
    """Autoregressive draft generator.

    Lifecycle::

        draft = DraftModel(cfg, target_tokenizer)
        draft.prime(prompt_ids)                   # one-shot prefill
        for round in spec_loop:
            ids, probs = draft.draft(last_token, k=cfg.draft_k)
            ... (target verifies, computes accepted_count) ...
            draft.rollback(committed_after_round)
    """

    def __init__(
        self,
        cfg: SpeculativeConfig,
        target_tokenizer: Any,
        device: Optional[str] = None,
    ) -> None:
        cfg.validate()
        self.cfg = cfg
        self.device = device or cfg.draft_device or default_device()

        # Load draft via the same StreamingCausalLM pipeline. Imported here
        # to avoid a top-level circular import (speculative imported from
        # engine.py, which itself imports streaming).
        from ..streaming import StreamingCausalLM, StreamingConfig

        # Default to partial_resident with very high keep_first/last so all
        # layers count as resident — this is functionally equivalent to
        # full_resident but takes StreamingCausalLM's from_config load path,
        # which bypasses transformers' AWQ quantizer (otherwise gptqmodel
        # intercepts and replaces our StreamingQLinear with its slow
        # TorchAtenAwqLinear; ~80x slowdown observed on Ada hardware).
        draft_streaming_cfg = cfg.draft_streaming_config or StreamingConfig(
            residency_mode="partial_resident",
            keep_first_k=1024,
            keep_last_k=1024,
        )
        log.info(
            "DraftModel loading %s (%s) on %s",
            cfg.draft_model_id,
            draft_streaming_cfg.summary(),
            self.device,
        )
        self.model = StreamingCausalLM.from_pretrained(
            cfg.draft_model_id,
            streaming_config=draft_streaming_cfg,
            dtype=torch.float16,
        )
        # full_resident requires an explicit .cuda() call per the
        # convention in demo_partial_8b.main.
        if draft_streaming_cfg.residency_mode == "full_resident":
            self.model.hf_model.to(self.device)

        # Draft tokenizer loaded separately; we still verify parity with
        # the target's tokenizer at construction.
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model_id)

        self._assert_tokenizer_parity(target_tokenizer)

        # Per-generation state.
        self._past_kv: Any = None
        self._primed_length: int = 0

    # ── Tokenizer parity ─────────────────────────────────────────────────────

    def _assert_tokenizer_parity(self, target_tokenizer: Any) -> None:
        """Strict vocab-size equality. Most same-family models share vocab,
        but a mismatch would silently corrupt verification, so fail loudly
        at construction rather than at first forward.
        """
        draft_vocab_size = self.tokenizer.vocab_size
        target_vocab_size = target_tokenizer.vocab_size
        if draft_vocab_size != target_vocab_size:
            raise ValueError(
                f"DraftModel tokenizer vocab_size ({draft_vocab_size}) "
                f"differs from target ({target_vocab_size}). Draft and "
                f"target must share a tokenizer family — e.g. both from "
                f"the Qwen2.5 line or both from Llama-3."
            )

    # ── Public API ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def prime(self, input_ids: torch.Tensor) -> None:
        """Prefill on the full prompt; populate ``_past_kv``.

        ``input_ids`` shape: ``(1, prompt_len)`` int64 on draft's device.
        Subsequent ``draft()`` calls extend from this state.
        """
        if input_ids.dim() != 2 or input_ids.size(0) != 1:
            raise ValueError(
                f"prime expects input_ids of shape (1, L), got "
                f"{tuple(input_ids.shape)}"
            )
        input_ids = input_ids.to(self.device)
        with last_logit_only(self.model):
            out = self.model(
                input_ids=input_ids,
                past_key_values=None,
                use_cache=True,
            )
        self._past_kv = out.past_key_values
        self._primed_length = int(input_ids.size(1))

    @torch.no_grad()
    def draft(
        self,
        last_token: int,
        k: int,
        *,
        return_probs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Autoregressive K-step draft starting after ``last_token``.

        Parameters
        ----------
        last_token: the most recently committed token id; draft picks up
            from this position
        k: number of tokens to propose
        return_probs: when False, skip the per-step fp32 softmax over
            ``V`` (~152K for Qwen) and return an empty probs tensor.
            Set False in greedy mode — the sampler doesn't read probs
            there. Saves ~3-6 ms per K=5 round on Ada.

        Returns
        -------
        ids: shape ``(K,)`` int64 — proposed token ids
        probs: shape ``(K, V)`` float — distribution at each step.
            All zeros (placeholder) when ``return_probs=False``.

        Side effects: ``_past_kv`` grows by K positions. Caller must
        ``rollback`` to the committed prefix after the verifier round.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if self._past_kv is None:
            raise RuntimeError("DraftModel.draft called before prime()")

        ids: list[int] = []
        probs_rows: list[torch.Tensor] = []
        cur_token = last_token

        for step in range(k):
            cur_ids = torch.tensor(
                [[cur_token]], dtype=torch.long, device=self.device
            )
            with last_logit_only(self.model):
                out = self.model(
                    input_ids=cur_ids,
                    past_key_values=self._past_kv,
                    use_cache=True,
                )
            self._past_kv = out.past_key_values
            step_logits = out.logits[0, -1, :]  # (V,)
            if return_probs:
                step_probs = F.softmax(step_logits.float(), dim=-1)
                probs_rows.append(step_probs)
            cur_token = int(step_logits.argmax(dim=-1).item())
            ids.append(cur_token)

        ids_t = torch.tensor(ids, dtype=torch.long, device=self.device)
        if return_probs:
            probs_t = torch.stack(probs_rows, dim=0)  # (K, V)
        else:
            # Return an empty placeholder so callers don't crash if they
            # try to access shape — but never read this in greedy mode.
            probs_t = torch.empty(0, device=self.device)
        return ids_t, probs_t

    def rollback(self, keep_n: int) -> None:
        """Truncate ``_past_kv`` to ``keep_n`` total positions.

        ``keep_n`` is the absolute desired length (prompt + committed
        tokens), not a delta. Caller is responsible for computing the
        correct target.
        """
        if not self.cfg.enable_kv_rollback:
            return
        if self._past_kv is None:
            return
        self._past_kv = truncate_past_kv(self._past_kv, keep_n)

    def reset(self) -> None:
        """Drop all KV state. Use before a fresh generation."""
        self._past_kv = None
        self._primed_length = 0

    # ── Introspection (for tests / debugging) ────────────────────────────────

    @property
    def past_kv(self) -> Any:
        return self._past_kv

    @property
    def primed_length(self) -> int:
        return self._primed_length
