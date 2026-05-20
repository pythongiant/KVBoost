"""Speculative decoding for KVBoost.

Drafts ``draft_k`` tokens with a small model, verifies them in one
multi-token forward on the target, accept-samples per token, rolls back
both KV caches on rejection. Composes with prefill-side CacheBlend and
the chunk-cache commit path — speculative operates on the decode phase
only, taking the post-CacheBlend ``past_kv`` as its input state.

Public API::

    from kvboost.speculative import SpeculativeConfig

    cfg = SpeculativeConfig(
        draft_model_id="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        draft_k=5,
        mode="greedy",
    )

The engine integration lives in ``kvboost.engine.InferenceEngine``: pass
``speculative_config=cfg`` to ``KVBoost.from_pretrained`` and the engine
will route the decode loop through ``SpeculativeEngine`` automatically.
"""

from __future__ import annotations

from .bridge import run_speculative_decode
from .config import SpeculativeConfig, SpeculativeMode
from .draft import DraftModel
from .engine import SpeculativeEngine
from .rollback import truncate_past_kv
from .sampler import verify_greedy, verify_sampling
from .stats import SpeculativeStats
from .verifier import TargetVerifier

__all__ = [
    "SpeculativeConfig",
    "SpeculativeMode",
    "SpeculativeEngine",
    "SpeculativeStats",
    "DraftModel",
    "TargetVerifier",
    "verify_greedy",
    "verify_sampling",
    "truncate_past_kv",
    "run_speculative_decode",
]
