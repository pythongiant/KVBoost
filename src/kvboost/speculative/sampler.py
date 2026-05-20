# src/kvboost/speculative/sampler.py

"""Verification samplers for speculative decoding.

Two modes:

- ``verify_greedy``: deterministic. Accept draft token i iff
  ``target.argmax(logits[i]) == draft_ids[i]``. With ``temperature=0``
  baselines this produces bit-identical output to non-speculative greedy
  decode.

- ``verify_sampling``: rejection sampling per Leviathan et al. 2023
  ("Fast Inference from Transformers via Speculative Decoding").
  ``accept_prob[i] = min(1, P_target(x_i) / P_draft(x_i))``. On the first
  reject, the correction is sampled from
  ``normalize(max(P_target - P_draft, 0))`` so the marginal output
  distribution matches the target model exactly.

Both return ``(accepted_count, output_token)`` where:
- accepted_count in [0, K] — number of draft tokens accepted from the front
- output_token is the (K+1)-th committed token, drawn from the rejected
  position's correction distribution, or — when all K were accepted — a
  bonus token sampled from the target's logits at position K
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def verify_greedy(
    draft_ids: torch.Tensor,
    target_logits: torch.Tensor,
) -> tuple[int, int]:
    """Greedy verification.

    Parameters
    ----------
    draft_ids: shape (K,) int tensor of draft-proposed token IDs
    target_logits: shape (K+1, V) target logits at positions 0..K, where
        position i is "what token should come AFTER consuming draft[:i]"

    Returns
    -------
    (accepted_count, output_token):
        - accepted_count in [0, K]
        - output_token is target.argmax at the first reject position, OR
          target.argmax at position K when all K were accepted (bonus)
    """
    if draft_ids.dim() != 1:
        raise ValueError(
            f"draft_ids must be 1-D, got shape {tuple(draft_ids.shape)}"
        )
    if target_logits.dim() != 2:
        raise ValueError(
            f"target_logits must be 2-D (K+1, V), got shape "
            f"{tuple(target_logits.shape)}"
        )
    k = draft_ids.shape[0]
    if target_logits.shape[0] != k + 1:
        raise ValueError(
            f"target_logits must have K+1 rows; got {target_logits.shape[0]} "
            f"for draft_k={k}"
        )

    target_argmax = target_logits.argmax(dim=-1)  # (K+1,)
    # Compare on the same device; cast draft to match.
    draft_ids_dev = draft_ids.to(target_argmax.device)

    # Find first mismatch in positions [0, K-1].
    mismatch_mask = target_argmax[:k] != draft_ids_dev
    if not mismatch_mask.any():
        # All K accepted; bonus token from position K.
        return k, int(target_argmax[k].item())

    first_reject = int(mismatch_mask.nonzero(as_tuple=False)[0].item())
    correction = int(target_argmax[first_reject].item())
    return first_reject, correction


def verify_sampling(
    draft_ids: torch.Tensor,
    draft_probs: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 1.0,
) -> tuple[int, int]:
    """Rejection-sampling verification (Leviathan et al. 2023).

    Parameters
    ----------
    draft_ids: shape (K,) int tensor — the tokens the draft sampled
    draft_probs: shape (K, V) — the distribution the draft sampled from at
        each position
    target_logits: shape (K+1, V) — target logits at positions 0..K
    temperature: applied to target logits before softmax

    Returns
    -------
    (accepted_count, output_token)

    Invariant: the marginal distribution of ``output_token`` exactly matches
    sampling from ``softmax(target_logits[accepted_count] / T)``. This is
    the mathematical guarantee that makes speculative sampling equivalent
    to baseline sampling — no quality loss.
    """
    if draft_ids.dim() != 1:
        raise ValueError(
            f"draft_ids must be 1-D, got shape {tuple(draft_ids.shape)}"
        )
    if draft_probs.dim() != 2:
        raise ValueError(
            f"draft_probs must be 2-D (K, V), got {tuple(draft_probs.shape)}"
        )
    if target_logits.dim() != 2:
        raise ValueError(
            f"target_logits must be 2-D (K+1, V), got "
            f"{tuple(target_logits.shape)}"
        )
    k = draft_ids.shape[0]
    if draft_probs.shape[0] != k:
        raise ValueError(
            f"draft_probs must have K rows; got {draft_probs.shape[0]} "
            f"for draft_k={k}"
        )
    if target_logits.shape[0] != k + 1:
        raise ValueError(
            f"target_logits must have K+1 rows; got {target_logits.shape[0]} "
            f"for draft_k={k}"
        )
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")

    device = target_logits.device
    draft_ids_dev = draft_ids.to(device)
    draft_probs_dev = draft_probs.to(device)

    target_probs = F.softmax(target_logits / temperature, dim=-1)  # (K+1, V)

    # Walk left-to-right; first reject ends the run.
    for i in range(k):
        token = int(draft_ids_dev[i].item())
        p_target = float(target_probs[i, token].item())
        p_draft = float(draft_probs_dev[i, token].item())
        # Guard divide-by-zero — draft shouldn't propose tokens it gave 0
        # probability, but in low-precision math it can be ~0.
        if p_draft <= 0.0:
            accept_prob = 1.0 if p_target > 0.0 else 0.0
        else:
            accept_prob = min(1.0, p_target / p_draft)

        if torch.rand((), device=device).item() < accept_prob:
            continue

        # Rejected at position i. Sample correction from
        # normalize(max(target - draft, 0)).
        residual = (target_probs[i] - draft_probs_dev[i]).clamp_min(0.0)
        total = float(residual.sum().item())
        if total <= 0.0:
            # Degenerate: target says everything in draft's support, residual
            # is all zero. Fall back to target distribution directly.
            correction_dist = target_probs[i]
        else:
            correction_dist = residual / total
        correction = int(torch.multinomial(correction_dist, num_samples=1).item())
        return i, correction

    # All K accepted; bonus token sampled from target_probs[K].
    bonus = int(torch.multinomial(target_probs[k], num_samples=1).item())
    return k, bonus
