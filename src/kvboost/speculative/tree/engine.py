"""Tree speculative decode engine.

Mirrors the flat ``SpeculativeEngine``'s decode contract exactly:

  Input:  (prompt_ids, target_past_kv, cached_length, max_new_tokens, ...)
  Output: (generated_tokens, target_past_kv) with final past_kv length
          == cached_length - 1 + len(generated)

Per-round flow:

  1. Pick a TreeShape via the cost-aware selector (or use config default).
  2. ``TreeDraftModel.draft_tree(...)`` builds a DraftTree + fork registry.
  3. ``TreeVerifier.verify_tree(...)`` runs one target forward over the
     flattened tree with custom attention mask.
  4. ``verify_tree_greedy`` / ``verify_tree_sampling`` walks the tree.
  5. ``commit_path_target_kv`` collapses target KV; ``commit_path_draft_kv``
     promotes the accepted-path's deepest fork to be the new drafter KV.
  6. Stats + EWMA update.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, List, Optional, Tuple

import torch

from ..draft import DraftModel
from ..stats import SpeculativeStats
from ..verifier import TargetVerifier
from .config import TreeSpeculativeConfig
from .draft import TreeDraftModel
from .rollback import commit_path_draft_kv, commit_path_target_kv
from .sampler import (
    TreeAcceptance,
    verify_tree_greedy,
    verify_tree_sampling,
)
from .shape import AcceptanceEWMA, TreeShape, pick_shape
from .structure import DraftTree
from .verifier import TreeVerifier

log = logging.getLogger(__name__)


def _kv_length(past_kv: Any) -> int:
    if past_kv is None:
        return 0
    if hasattr(past_kv, "get_seq_length"):
        return int(past_kv.get_seq_length())
    return int(past_kv[0][0].shape[2])


def _cuda_sync_if_needed(device: Any) -> None:
    if device is None:
        return
    try:
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    except Exception:
        pass


class TreeSpeculativeEngine:
    """Tree speculative engine. Wraps the existing draft + target models.

    Reuses the flat engine's ``DraftModel`` and ``TargetVerifier``-style
    target wrapper for model + device handling. Adds the tree-specific
    components.

    Sampling mode: ``"greedy"`` (default) gives bit-exact greedy
    decoding. ``"sampling"`` uses SpecInfer-style per-level rejection
    sampling to preserve the target distribution.
    """

    def __init__(
        self,
        cfg: TreeSpeculativeConfig,
        target_verifier: TargetVerifier,
        draft_model: DraftModel,
        *,
        cost_coefficients: Any = None,
        target_step_ms: float = 50.0,
        draft_step_ms: float = 5.0,
        mode: str = "greedy",
        temperature: float = 1.0,
        stats: Optional[SpeculativeStats] = None,
    ) -> None:
        cfg.validate()
        if mode not in ("greedy", "sampling"):
            raise ValueError(f"mode must be greedy or sampling, got {mode!r}")
        self.cfg = cfg
        self.tree_drafter = TreeDraftModel(draft_model)
        self.draft = draft_model
        self.tree_verifier = TreeVerifier(
            target_verifier.model, device=target_verifier.device,
        )
        self.target_verifier = target_verifier  # for fallback rollback
        self.mode = mode
        self.temperature = temperature
        self.stats = stats or SpeculativeStats()
        self.ewma = AcceptanceEWMA(
            alpha=cfg.ewma_alpha, cold_accept=cfg.cold_accept,
        )
        self.cc = cost_coefficients
        self.target_step_ms = target_step_ms
        self.draft_step_ms = draft_step_ms

    # ── Public entry point ───────────────────────────────────────────

    @torch.no_grad()
    def decode_from(
        self,
        prompt_ids: List[int],
        target_past_kv: Any,
        cached_length: int,
        max_new_tokens: int,
        eos_token_id: Optional[int] = None,
        on_token: Optional[Callable[[int], None]] = None,
    ) -> Tuple[List[int], Any]:
        """Run the tree-speculative decode loop. Same contract as flat."""
        if len(prompt_ids) != cached_length:
            raise ValueError(
                f"cached_length={cached_length} must equal len(prompt_ids)="
                f"{len(prompt_ids)}"
            )
        if max_new_tokens <= 0:
            return [], target_past_kv

        # Match the flat engine's boundary roll-back: the prompt's last
        # token becomes the boundary that's NOT in past_kv.
        last_committed = prompt_ids[-1]
        target_past_kv = self.target_verifier.rollback(
            target_past_kv, cached_length - 1
        )
        committed_length = cached_length - 1

        # Prime the drafter on the prompt minus the last token.
        prompt_minus_one = prompt_ids[: cached_length - 1]
        if prompt_minus_one:
            prompt_t = torch.tensor(
                [prompt_minus_one], dtype=torch.long,
                device=self.draft.device,
            )
            self.tree_drafter.prime(prompt_t)
        else:
            self.draft._past_kv = None
            self.draft._primed_length = 0

        generated: List[int] = []
        draft_device = getattr(self.draft, "device", None)
        target_device = getattr(self.target_verifier, "device", None)

        round_idx = 0
        while len(generated) < max_new_tokens:
            remaining = max_new_tokens - len(generated)
            shape = self._pick_shape(remaining_budget=remaining)

            # Defensive prime if degenerate prompt left drafter unset.
            if self.draft._past_kv is None:
                prompt_t = torch.tensor(
                    [[last_committed]], dtype=torch.long,
                    device=self.draft.device,
                )
                self.tree_drafter.prime(prompt_t)
                self.draft.rollback(0)

            _cuda_sync_if_needed(draft_device)
            t_round = time.perf_counter()

            # ── 1. Build the draft tree ──
            draft_result = self.tree_drafter.draft_tree(
                last_token=last_committed, shape=shape,
            )
            tree = draft_result.tree
            fork_registry = draft_result.fork_registry
            draft_dt = draft_result.elapsed_s

            # ── 2. Target tree-verify in ONE forward ──
            _cuda_sync_if_needed(target_device)
            t_verify0 = time.perf_counter()
            per_node_logits, target_past_kv = self.tree_verifier.verify_tree(
                tree, target_past_kv, committed_length,
            )
            _cuda_sync_if_needed(target_device)
            verify_dt = time.perf_counter() - t_verify0

            # ── 3. Acceptance walk ──
            if self.mode == "greedy":
                acc = verify_tree_greedy(tree, per_node_logits)
            else:
                acc = verify_tree_sampling(
                    tree, per_node_logits, temperature=self.temperature,
                )

            # ── 4. Commit accepted tokens (honoring EOS / budget) ──
            committed_toks = acc.committed_tokens(tree)
            n_drafted_in_path = acc.n_drafted_accepted  # excludes correction
            committed_this_round = 0
            stop = False
            for tok in committed_toks:
                generated.append(tok)
                committed_this_round += 1
                if on_token is not None:
                    on_token(tok)
                if eos_token_id is not None and tok == eos_token_id:
                    stop = True
                    break
                if len(generated) >= max_new_tokens:
                    stop = True
                    break
            # How many of the kept tokens were DRAFTED (vs correction)?
            # ``committed_toks[:n_drafted_in_path]`` are drafted nodes;
            # the rest is the trailing correction. If early-stop cut
            # before the correction, n_committed_draft == committed_this_round.
            n_committed_draft = min(committed_this_round, n_drafted_in_path)

            # ── 5. Commit target KV to the accepted path ──
            # Note: accepted_node_ids[1:] are the drafted nodes accepted;
            # if early-stop dropped some, we keep only the leading ones.
            kept_node_ids = (
                [0] + acc.accepted_node_ids[1 : 1 + n_committed_draft]
            )
            target_past_kv = commit_path_target_kv(
                target_past_kv,
                committed_length=committed_length,
                accepted_node_ids=kept_node_ids,
            )

            # ── 6. Promote the deepest accepted fork as drafter KV ──
            new_committed_length = (
                committed_length + 1 + n_committed_draft
            )
            deepest = (
                kept_node_ids[-1] if kept_node_ids else 0
            )
            new_draft_kv = commit_path_draft_kv(
                fork_registry,
                accepted_node_ids=kept_node_ids,
                deepest_node_id=deepest,
                committed_length_after_target=new_committed_length,
            )
            if new_draft_kv is not None:
                self.draft._past_kv = new_draft_kv

            wall_ms = (time.perf_counter() - t_round) * 1000.0

            # ── 7. Stats + EWMA ──
            self.stats.record_round(
                accepted_count=n_committed_draft,
                draft_k=max(1, tree.n_nodes - 1),
                draft_time_s=draft_dt,
                verify_time_s=verify_dt,
                rollback_time_s=0.0,
            )
            self.ewma.record(
                branching=shape.branching,
                depth=shape.depth,
                accepted=n_committed_draft,
                drafted_path_len=acc.n_drafted_accepted,
                committed=committed_this_round,
                wall_ms=wall_ms,
            )

            # ── 8. Advance to next round ──
            committed_length = new_committed_length
            if committed_this_round > 0:
                last_committed = generated[-1]
            else:
                log.warning(
                    "tree speculative round committed 0 tokens; breaking"
                )
                break

            round_idx += 1
            if stop:
                break

        return generated, target_past_kv

    # ── helpers ──────────────────────────────────────────────────────

    def _pick_shape(self, *, remaining_budget: int) -> TreeShape:
        """Cost-aware shape pick, capped by remaining token budget."""
        if self.cc is None:
            # No cost model → use config defaults.
            return TreeShape(
                branching=self.cfg.max_branching,
                depth=min(self.cfg.max_depth, max(2, remaining_budget)),
                node_budget=self.cfg.node_budget,
            )
        try:
            free_vram = self._free_vram_mb()
            shape, _score = pick_shape(
                cost_coefficients=self.cc,
                ewma=self.ewma,
                target_step_ms=self.target_step_ms,
                draft_step_ms=self.draft_step_ms,
                free_vram_mb=free_vram,
                max_branching=self.cfg.max_branching,
                max_depth=min(self.cfg.max_depth, max(2, remaining_budget)),
                node_budget=self.cfg.node_budget,
                verify_extra_per_node=self.cfg.verify_extra_per_node,
            )
            return shape
        except Exception as exc:
            log.warning(
                "pick_shape failed (%s); using config defaults", exc,
            )
            return TreeShape(
                branching=self.cfg.max_branching,
                depth=min(self.cfg.max_depth, max(2, remaining_budget)),
                node_budget=self.cfg.node_budget,
            )

    def _free_vram_mb(self) -> Optional[float]:
        try:
            dev = self.target_verifier.device
            if torch.device(dev).type != "cuda":
                return None
            idx = torch.device(dev).index
            if idx is None:
                idx = torch.cuda.current_device()
            free_bytes, _ = torch.cuda.mem_get_info(idx)
            return free_bytes / (1024.0 ** 2)
        except Exception:
            return None

